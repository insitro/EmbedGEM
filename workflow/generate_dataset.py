import os
from typing import Dict, List, Tuple
from redun import File

import numpy as np
import pandas as pd
import redun
import yaml
from clumping import clump
from heritability import compute_multivariate_heritability
from disease_relevance import run_prs
from disease_relevance_association import get_prs_metrics
from scipy.stats import chi2


P_VALUE_COLNAME = "P"
redun_namespace = "embedgem"


@redun.task()
def get_clumped_metrics(
    chrom_to_clumped_sumstat: Dict[str, redun.File],
    output_path: str,
    pval_thresh: float,
) -> File:
    """
    This function gets clumped metrics.

    Parameters:
        chrom_to_clumped_sumstat (Dict[str, redun.File]): A dictionary mapping
          chromosomes to clumped summary statistics files.
        output_path (str): The path where the output file will be saved.
        pval_thresh (float): The p-value threshold.

    Returns:
        File: The output file containing the clumped metrics.
    """
    chi2_list = np.array([])

    for _, sumstat in chrom_to_clumped_sumstat.items():
        p_val_col = pd.read_csv(
            sumstat.path, sep="\t", usecols=[P_VALUE_COLNAME]
        ).dropna()
        chi2_list = np.append(
            chi2_list,
            chi2.isf(p_val_col[p_val_col.P <= pval_thresh][P_VALUE_COLNAME], df=1),
        )

    out_df = pd.DataFrame(
        {
            "metrics": ["no_hits", "mean_chi2", "median_chi2"],
            "values": [len(chi2_list), np.mean(chi2_list), np.median(chi2_list)],
        }
    )

    metrics_outfile_path = os.path.join(output_path, "clumped_metrics.tsv")
    out_df.to_csv(metrics_outfile_path, index=False, sep="\t")
    return redun.File(metrics_outfile_path)


def prep_and_shard_sumstats(
    sumstat_dirs: List[str], pheno_prefix: str, chroms: List[str]
) -> Dict[str, Dict]:
    """
    This function prepares and shards summary statistics.

    Parameters:
        sumstat_dirs (List[str]): A list of directories containing summary statistics.
        pheno_prefix (str): The prefix for the phenotype.
        chroms (List[str]): A list of chromosomes.

    Returns:
        Dict[str, Dict]: A dictionary mapping phenotype prefixes to dictionaries
          containing paths to summary statistics files.
    """

    # break apart genomewide sumstat if chrom sumstats do not exist
    for i, sumstat_dir in enumerate(sumstat_dirs):
        genomewide_sumstat_path = os.path.join(sumstat_dir, "genomewide.sumstat")
        assert redun.File(genomewide_sumstat_path).exists(), FileNotFoundError(
            genomewide_sumstat_path
        )
        if not all(
            [
                redun.File(os.path.join(sumstat_dir, f"chr{chrom}.sumstat")).exists()
                for chrom in chroms
            ]
        ):
            # shard
            genomewide_sumstat_df = pd.read_csv(genomewide_sumstat_path, sep="\t")
            for chrom, chrom_df in genomewide_sumstat_df.groupby(
                genomewide_sumstat_df["#CHROM"]
            ):
                chrom_path = os.path.join(sumstat_dir, f"chr{chrom}.sumstat")
                chrom_df.to_csv(chrom_path, sep="\t", index=False)

    return {
        f"{pheno_prefix}_PC{i}": {
            "sumstat_path": os.path.join(dir_path, "genomewide.sumstat"),
            "sumstat_chr_path_pattern": os.path.join(dir_path, "chr{CHROM}.sumstat"),
        }
        for i, dir_path in enumerate(sumstat_dirs)
    }


@redun.task(no_cache=True)
def main(config: redun.File) -> Tuple[File, File]:
    """
    This function is the main function.

    Parameters:
        config (redun.File): A file containing the configuration.

    Returns:
        Tuple[File, File]: A tuple containing two Files: the heritability evaluation
          metrics and the disease relevance metrics.
    """

    with open(config.path, "r") as stream:
        d = yaml.safe_load(stream)

    pheno_prefix = d["pheno_prefix"]
    output_dir = os.path.join(d["s3_output_path"], pheno_prefix)
    bfile_template = d["bfile_template"]
    chroms = d.get("chromosomes", [str(i) for i in range(1, 23)])

    clumping_params = d.get("clumping", {})
    maf_threshold = clumping_params.get("maf_threshold", 0.0001)
    max_kb = clumping_params.get("max_kb", 250)
    max_pval_all_variants = clumping_params.get("max_pval_all_variants", 1.0e-05)
    max_pval_lead_variants = clumping_params.get("max_pval_lead_variants", 5.0e-08)
    min_r2 = clumping_params.get("min_r2", 0.5)

    # Read in the sumstats file, and construct inputs
    sumstat_dirs = d.get("sumstat_dirs", [])
    trait_to_raw_sumstats = prep_and_shard_sumstats(sumstat_dirs, pheno_prefix, chroms)

    # Begin disease relevance evaluation
    disease_relevance_output_dir = os.path.join(output_dir, "disease_relevance_eval")
    # Running clumping and getting output
    trait_to_clumped_sumstat_by_chrom: Dict[str, Dict[str, redun.File]] = {
        trait_name: clump(
            chrom_sumstat_pattern=paths["sumstat_chr_path_pattern"],
            bfile_template_pattern=bfile_template,
            output_path=os.path.join(
                disease_relevance_output_dir,
                "scratch",
                "clumping",
                trait_name.rsplit("_")[-1],
            ),
            chromosomes=chroms,
            maf=maf_threshold,
            r2=min_r2,
            kb=max_kb,
            p1=max_pval_lead_variants,
            p2=max_pval_all_variants,
        )
        for trait_name, paths in trait_to_raw_sumstats.items()
    }

    prs_out = run_prs(
        chroms=chroms,
        bfile_template_pattern=d["disease_relevance_options"][
            "disease_relevance_bfile_template"
        ],
        output_path=disease_relevance_output_dir,
        clumping_mapping=trait_to_clumped_sumstat_by_chrom,
    )

    disease_relevance_metrics = get_prs_metrics(
        prs_opt_dict=d["disease_relevance_options"],
        prs_file=prs_out,
        output_path=os.path.join(disease_relevance_output_dir),
    )

    # Begin heritability evaluation

    # Running non-PRS validation
    heritability_pval_thresh = d["heritability_eval"].get("pval_thresh", 5.0e-8)
    heritability_eval_output_dir = os.path.join(output_dir, "heritability_eval")
    chrom_level_sumstat_pattern_nonprs, _ = compute_multivariate_heritability(
        trait_prefix=f"{pheno_prefix}_PC",
        trait_to_raw_sumstats=trait_to_raw_sumstats,
        output_path=os.path.join(heritability_eval_output_dir, "scratch"),
        pval_thresh=heritability_pval_thresh,
        max_p=d["heritability_eval"].get("max_p", 5.0e-8),
    )

    # Run clumping for output of non-prs
    chrom_to_clumped_sumstat = clump(
        chrom_sumstat_pattern=chrom_level_sumstat_pattern_nonprs,
        bfile_template_pattern=bfile_template,
        # Maybe update the output path given the multivariate now
        output_path=heritability_eval_output_dir,
        chromosomes=chroms,
        maf=maf_threshold,
        r2=min_r2,
        kb=max_kb,
        p1=max_pval_lead_variants,
        p2=max_pval_all_variants,
    )

    # Getting nonPRS clumped metrics
    heritability_eval_metrics = get_clumped_metrics(
        chrom_to_clumped_sumstat=chrom_to_clumped_sumstat,
        output_path=heritability_eval_output_dir,
        pval_thresh=heritability_pval_thresh,
    )

    return heritability_eval_metrics, disease_relevance_metrics
