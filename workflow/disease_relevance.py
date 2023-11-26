import os
from typing import Dict, List, Tuple
from utils import logger

import pandas as pd
from redun import File, script, task

PLINK1_EXECUTABLE = "plink"
VARIANT_ID_COLNAME = "ID"
A1_ALLELE_COLNAME = "A1"
BETA_COLNAME = "BETA"
PRS_ID_COLNAME = "IID"

redun_namespace = "disease_relevance_eval"


@task()
def compute_prs(
    bfile_prefix: str,
    clumped_sumstat: File,
    output_fileset: str,
    plink_executable: str = PLINK1_EXECUTABLE,
) -> Tuple[File, File]:
    """
    Compute polygenic risk score.

    Parameters:
        bfile_prefix (str): Prefix for the binary fileset (.bim, .bed, .fam files).
        clumped_sumstat (File): File containing clumped summary statistics.
        output_fileset (str): Path to the output fileset.
        plink_executable (str, optional): Path to the PLINK executable.

    Returns:
        Tuple[File, File]: Returns a tuple of Files: log file and PRS file.
    """
    bim_file = File(f"{bfile_prefix}.bim")
    bed_file = File(f"{bfile_prefix}.bed")
    fam_file = File(f"{bfile_prefix}.fam")

    local_plink_prefix = "plink"
    local_sumstat_path = "local.sumstat"
    local_output_prefix = "outs"

    sumstat_cols = pd.read_csv(clumped_sumstat.path, sep="\t", nrows=1).columns

    variant_col = str(sumstat_cols.get_loc(VARIANT_ID_COLNAME) + 1)
    allele_col = str(sumstat_cols.get_loc(A1_ALLELE_COLNAME) + 1)
    beta_col = str(sumstat_cols.get_loc(BETA_COLNAME) + 1)

    return script(
        f"""
        touch {local_output_prefix}.profile && {plink_executable} \
            --bfile {local_plink_prefix} \
            --score \
            {local_sumstat_path} {variant_col} {allele_col} {beta_col} header sum \
            --out {local_output_prefix}
        """,
        inputs=[
            clumped_sumstat.stage(local_sumstat_path),
            bed_file.stage(f"{local_plink_prefix}.bed"),
            bim_file.stage(f"{local_plink_prefix}.bim"),
            fam_file.stage(f"{local_plink_prefix}.fam"),
        ],
        executor="batch",
        vcpu=2,
        mem=32,
        outputs=[
            File(f"{output_fileset}.log").stage(f"{local_output_prefix}.log"),
            File(f"{output_fileset}.prs").stage(f"{local_output_prefix}.profile"),
        ],
    )


@task()
def _compute_prs_for_trait(
    bfile_template_pattern: str,
    output_dir: str,
    chroms: List[str],
    chrom_to_clumping_sumstat: Dict[str, File],
) -> File:
    """
    Compute polygenic risk score for a single trait.

    Parameters:
        bfile_template_pattern (str): Template pattern for the binary fileset.
        output_dir (str): Directory for the output files.
        chroms (List[str]): List of chromosomes.
        chrom_to_clumping_sumstat (Dict[str, File]): Mapping of chromosomes to
        clumped summary statistics files.

    Returns:
        File: Returns a File containing polygenic risk score for a single trait.
    """
    chrom_to_prs_file = {}

    for chrom in chroms:
        if not chrom_to_clumping_sumstat.get(chrom):
            logger.warning(f"Sumstat not found for chrom {chrom}, skipping...")
            continue

        chrom_to_prs_file[chrom] = compute_prs(
            bfile_prefix=bfile_template_pattern.replace("{CHROM}", chrom),
            output_fileset=os.path.join(output_dir, f"prs_chr{chrom}"),
            clumped_sumstat=chrom_to_clumping_sumstat[chrom],
        )[1]

    # merge into genomewide result
    return _merge_chromosome_results(chrom_to_prs_file, output_dir)


@task()
def _merge_chromosome_results(plink_outputs: Dict[str, File], outdir: str) -> File:
    """
    Merges per-chromosome plink outputs.

    Parameters:
        plink_outputs (Dict[str, File]): Dictionary of PLINK output files.
        outdir (str): Directory for the output file.

    Returns:
        File: Returns the merged File.
    """
    files_to_merge = list(plink_outputs.values())

    if len(files_to_merge) == 0:
        raise Exception("No results found to merge prs.")
    first_df = pd.read_csv(files_to_merge[0].path, index_col=0, delim_whitespace=True)
    scores = first_df[["CNT", "CNT2", "SCORESUM"]]
    phenos = first_df[[PRS_ID_COLNAME, "PHENO"]]

    for file in files_to_merge[1:]:
        df = pd.read_csv(file.path, index_col=0, delim_whitespace=True)
        scores = scores.add(df[["CNT", "CNT2", "SCORESUM"]], fill_value=0)

    final_df = phenos.join(scores)
    out_file = File(os.path.join(outdir, "genomewide.prs"))
    final_df.to_csv(out_file.path, sep="\t")

    return out_file


@task()
def _build_and_save_dataframe(outdir: str, traits_to_prs_file: Dict[str, File]) -> File:
    """
    Create dataframe with all results for all traits and write to a csv file.

    Parameters:
        outdir (str): Directory for the output file.
        traits_to_prs_file (Dict[str, File]): Mapping of traits to prs files.

    Returns:
        File: Returns the merged File.
    """
    dfs = []
    for trait, prs_file in traits_to_prs_file.items():
        df = pd.read_csv(
            prs_file.path,
            delim_whitespace=True,
            usecols=[PRS_ID_COLNAME, "SCORESUM"],
            index_col=PRS_ID_COLNAME,
        )

        df.rename(columns={"SCORESUM": trait}, inplace=True)
        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No PRS outputs found!")

    df = dfs[0].join(dfs[1:], how="inner")

    out_file = File(os.path.join(outdir, "prs_df.tsv"))
    logger.info(f"Saving joined PRS dataframe to {out_file.path}")
    df.to_csv(out_file.path, sep="\t")

    return out_file


@task()
def run_prs(
    bfile_template_pattern: str,
    clumping_mapping: Dict[str, Dict[str, File]],
    output_path: str,
    chroms: List,
) -> File:
    """
    Runs polygenic risk score.

    Parameters:
        bfile_template_pattern (str): Template pattern for the binary fileset.
        clumping_mapping (Dict[str, Dict[str, File]]): Mapping of traits to
        clumped summary statistics files.
        output_path (str): Path for the output file.
        chroms (List): List of chromosomes.

    Returns:
        File: Returns the output File.
    """
    logger.info("Running disease relevance eval")
    trait_to_prs_file = {
        trait_name: _compute_prs_for_trait(
            bfile_template_pattern=bfile_template_pattern,
            chrom_to_clumping_sumstat=chrom_to_clumped_sumstat,
            chroms=chroms,
            output_dir=os.path.join(
                output_path,
                "scratch",
                "trait_level_prs",
                trait_name.rsplit("_")[-1],
            ),
        )
        for trait_name, chrom_to_clumped_sumstat in clumping_mapping.items()
    }

    return _build_and_save_dataframe(output_path, trait_to_prs_file)
