import os
from utils import logger
from typing import Dict, Tuple

import functools
import numpy as np
import pandas as pd
from redun import File, task
from scipy.stats import chi2

SUMSTAT_VARIANT_ID = "ID"
SUMSTAT_P_VALUE_COLNAME = "P"
CHROM_COLNAME = "#CHROM"

OUTPUT_TEST_STATISTIC_COLNAME = "TEST_STATISTIC"

redun_namespace = "heritability_eval"


def get_metrics(merged_df: pd.DataFrame, output_dir: str, pval_thresh: float) -> File:
    """
    Parameters:
    - merged_df: A pandas DataFrame representing a processed sumstat file.
      It is expected to have at least two columns: 'P' and 'TEST_STATISTIC'.
    - output_dir: A string representing the directory where the output file will be saved.
    - pval_thresh: A float representing the p-value threshold. Rows in the DataFrame with a 'P' value less than or equal to this
    threshold will be considered significant.

    Returns:
    - A File object representing the output file. The output file is a tab-separated CSV file named 'multivariate_metrics.csv'
    saved in the specified output directory.
      It contains three metrics calculated from the significant rows in the input DataFrame: 'no_hits' (the number of significant
      rows), 'mean_chi2' (the mean of the 'TEST_STATISTIC' values), and 'median_chi2' (the median of the 'TEST_STATISTIC'
      values).
    """

    merged_df = merged_df.dropna()
    sig_df = merged_df[merged_df.P <= pval_thresh]
    no_hits = len(sig_df)
    median_chi2 = np.median(sig_df[OUTPUT_TEST_STATISTIC_COLNAME])
    mean_chi2 = np.mean(sig_df[OUTPUT_TEST_STATISTIC_COLNAME])

    out_file_path = os.path.join(output_dir, "multivariate_metrics.csv")
    metrics = pd.DataFrame(
        {
            "metrics": ["no_hits", "mean_chi2", "median_chi2"],
            "values": [no_hits, mean_chi2, median_chi2],
        }
    )

    metrics.to_csv(out_file_path, index=False, sep="\t")
    return File(out_file_path)


def shard_sumstat_by_chrom(
    genomewide_sumstat_df: pd.DataFrame, output_path_format: str
) -> Tuple[Dict[str, File], str]:
    """
    Shard a sumstat dataframe by chromosome and write shards to disk as csv files.

    Parameters:
        genomewide_sumstat_df: A pandas DataFrame containing genome-wide summary statistics.
            It is expected to have a column named CHROM_COLNAME which contains the chromosome names.
        output_path_format: A string format for the output path. It should contain a placeholder "{CHROM}"
            which will be replaced by the chromosome name for each output file.

    Returns:
    - A tuple containing two elements:
          1. A dictionary where the keys are chromosome names (as strings) and the values are File objects representing the
          output files.
          2. The original output path format string.
    """

    results: Dict[str, File] = {}
    for chrom, chrom_df in genomewide_sumstat_df.groupby(
        genomewide_sumstat_df[CHROM_COLNAME]
    ):
        chrom_path = output_path_format.format(CHROM=chrom)
        chrom_df.to_csv(chrom_path, sep="\t", index=False)
        results[str(chrom)] = File(chrom_path)
    return results, output_path_format


@task()
def compute_multivariate_heritability(
    trait_prefix: str,
    trait_to_raw_sumstats: dict,
    output_path: str,  # Output directory for processed sumstats
    pval_thresh: float = 5.0e-8,  # p-value threshold for metrics
    max_p: float = 1e-5,  # Maximum P-value to keep in sumstat
) -> Tuple[str, File]:
    """
    Compute multivariate GWAS.

    Parameters:
        trait_prefix (str): Prefix of the multivariate traits. All multivariate traits should have the form *_PC
        trait_to_raw_sumstats (dict): Dictionary mapping traits to raw summary statistics
        output_path (str): Output directory for processed summary statistics
        pval_thresh (float, optional): p-value threshold for metrics. Defaults to 5.0e-8.
        max_p (float, optional): Maximum P-value to keep in summary statistics. Defaults to 1e-5.

    Returns:
        Tuple[str, File]: A tuple containing the chromosome pattern and the metrics file.
    """
    # read file for each PC and compute chi-sq statistic for each PC
    df_list = []
    logger.info(f"Computing heritability for {trait_prefix}")
    template_path = ""
    for trait, raw_sumstat in trait_to_raw_sumstats.items():
        variant_id_p_val_df = pd.read_csv(
            raw_sumstat["sumstat_path"],
            sep="\t",
            usecols=[SUMSTAT_VARIANT_ID, SUMSTAT_P_VALUE_COLNAME],
            index_col=SUMSTAT_VARIANT_ID,
        )
        variant_id_p_val_df[f"chi2_PC{trait.split('_PC')[-1]}"] = chi2.isf(
            variant_id_p_val_df[SUMSTAT_P_VALUE_COLNAME], df=1
        )
        variant_id_p_val_df = variant_id_p_val_df.drop(
            [SUMSTAT_P_VALUE_COLNAME], axis=1
        )
        df_list += [variant_id_p_val_df]
        # assuming all variants are the same, take the last one as a template for the other columns
        template_path = raw_sumstat["sumstat_path"]

    df_template = pd.read_table(
        template_path,
        usecols=[
            SUMSTAT_VARIANT_ID,
            CHROM_COLNAME,
            "POS",
            "A1",
            "REF",
            "ALT",
            "A1_FREQ",
        ],
    )

    df_merged = functools.reduce(
        lambda left, right: pd.merge(left, right, on=SUMSTAT_VARIANT_ID), df_list
    )

    df_merged[OUTPUT_TEST_STATISTIC_COLNAME] = df_merged.sum(axis=1)
    df_merged[SUMSTAT_P_VALUE_COLNAME] = chi2.sf(
        df_merged[OUTPUT_TEST_STATISTIC_COLNAME], df=len(trait_to_raw_sumstats)
    )
    # re-calibrating chi2 statistic
    df_merged[OUTPUT_TEST_STATISTIC_COLNAME] = chi2.isf(
        df_merged[SUMSTAT_P_VALUE_COLNAME], df=1
    )

    df_merged = pd.merge(
        df_merged,
        df_template,
        on=SUMSTAT_VARIANT_ID,
    )
    df_final = df_merged[df_merged.P <= max_p]

    metrics_file = get_metrics(df_final, output_path, pval_thresh)

    new_sumstat_file_path = os.path.join(output_path, "multivariate_genomewide.csv")

    _, chrom_pattern = shard_sumstat_by_chrom(
        df_merged, new_sumstat_file_path.replace("genomewide.csv", "chr{CHROM}.csv")
    )
    df_merged.to_csv(new_sumstat_file_path, sep="\t", index=False)

    return chrom_pattern, metrics_file
