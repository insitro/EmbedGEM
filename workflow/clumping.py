import os
from utils import logger
from typing import Dict, Optional
import pandas as pd
from redun import File, script, task

PLINK1_EXECUTABLE = "plink"

CLUMP_REPORT_VARIANT_ID = "SNP"
SUMSTAT_VARIANT_ID = "ID"

# ASSUMPTION is that ID column is unique
redun_namespace = "clumping"


@task()
def clump_single_chromosome(
    bfile_prefix: str,
    sumstat_path: str,
    output_path: str,
    p1: float,
    p2: float,
    r2: float,
    kb: float,
    maf: float,
    clump_field: str = "P",
    snp_field: str = SUMSTAT_VARIANT_ID,
    plink_executable: str = PLINK1_EXECUTABLE,
) -> File:
    """
    Perform clumping on a single chromosome.

    Parameters:
        bfile_prefix (str): The prefix for the .bed, .bim, and .fam files
        (binary PLINK files).
        sumstat_path (str): The path to the summary statistics file.
        output_path (str): The path where the output file will be saved.
        p1 (float): The p-value threshold for lead variant?
        p2 (float): The p-value threshold for all variants?
        r2 (float): The r-squared threshold for clumping.
        kb (float): The kilobase window for clumping.
        maf (float): The minor allele frequency threshold for clumping.
        clump_field (str, optional): The field in the summary statistics
          file to use for clumping. Defaults to "P".
        snp_field (str, optional): The field in the summary statistics
          file that contains the SNP IDs.
        plink_executable (str, optional): The path to the PLINK executable.

    Returns:
        File: The output file containing the clumped results.
    """

    bim_file = File(f"{bfile_prefix}.bim")
    bed_file = File(f"{bfile_prefix}.bed")
    fam_file = File(f"{bfile_prefix}.fam")

    local_sumstat_path = "local.sumstat"
    local_bfile_prefix = "plink"
    local_output_prefix = "plink_clump_output"
    # Touch file first to handle chromosomes without genomewide-significant results
    return script(
        f"""
        touch {local_output_prefix}.clumped &&
        {plink_executable} \
            --bfile {local_bfile_prefix} \
            --clump {local_sumstat_path} \
            --clump-p1 {p1} \
            --clump-p2 {p2} \
            --clump-r2 {r2} \
            --clump-kb {kb} \
            --clump-field {clump_field} \
            --clump-snp-field {snp_field} \
            --maf {maf} \
            --out {local_output_prefix}
        """,
        inputs=[
            File(sumstat_path).stage(local_sumstat_path),
            bed_file.stage(f"{local_bfile_prefix}.bed"),
            bim_file.stage(f"{local_bfile_prefix}.bim"),
            fam_file.stage(f"{local_bfile_prefix}.fam"),
        ],
        executor="batch",
        vcpu=16,
        mem=60,
        outputs=File(output_path).stage(f"{local_output_prefix}.clumped"),
    )


@task()
def gather(chrom_to_clumped_sumstats: Dict[str, File], output_path: str) -> File:
    """
    Gathers clumped summary statistics from multiple chromosomes into one file.

    Parameters:
        chrom_to_clumped_sumstats (Dict[str, File]): A dictionary mapping
          chromosome names to files containing clumped summary statistics
          for each chromosome.
        output_path (str): The path where the output file will be saved.

    Returns:
        File: The output file containing the gathered clumped summary statistics.

    """
    chrom_sumstats = [
        pd.read_csv(f.path, sep="\t") for f in chrom_to_clumped_sumstats.values()
    ]
    clumping_report_df = pd.concat(chrom_sumstats, axis=0).reset_index(drop=False)
    output = File(os.path.join(output_path, "genomewide_clumped.sumstat"))

    clumping_report_df.to_csv(output.path, sep="\t", index=False)
    return output


@task()
def create_clumped_sumstat(sumstat: File, clumping_report: File) -> Optional[File]:
    """
    Create a new sumstat that retains only the clumped variants.

    Parameters:
        sumstat (File): The original summary statistics file.
        clumping_report (File): The clumping report file.

    Returns:
        Optional[File]: The output file containing the filtered summary statistics.
        If the clumping report is empty (i.e., no variants passed the genome-wide
        significance threshold), the function returns None.

    """

    try:
        clumps = pd.read_csv(clumping_report.path, sep=r"\s+")
    except pd.errors.EmptyDataError:
        logger.info(
            "no clump passed genome-wide significance threshold "
            f"for sumstat {clumping_report.path}"
        )
        return None

    df_all = pd.read_csv(sumstat.path, sep="\t")

    idx = df_all[SUMSTAT_VARIANT_ID].isin(clumps[CLUMP_REPORT_VARIANT_ID])
    df_clumped = df_all.loc[idx].reset_index(drop=True)
    clumped_sumstat_path = clumping_report.path.replace(".clumped", "_clumped.sumstat")
    df_clumped.to_csv(clumped_sumstat_path, index=False, sep="\t")
    return File(clumped_sumstat_path)


@task()
def filter_empty_clumped_sumstats(
    chrom_to_clumped_sumstat: Dict[str, Optional[File]]
) -> Dict[str, File]:
    """
    Filters out chromosomes that do not have any clumped summary statistics.

    Parameters:
        chrom_to_clumped_sumstat (Dict[str, Optional[File]]): A dictionary
          mapping chromosome names to files containing clumped summary statistics
          for each chromosome. If a chromosome does not have any clumped summary
          statistics, its value in the dictionary is None.

    Returns:
        Dict[str, File]: A dictionary mapping chromosome names to files containing
        clumped summary statistics, with chromosomes that do not have any clumped
        summary statistics removed.
    """
    filtered = {k: v for k, v in chrom_to_clumped_sumstat.items() if v}
    if len(filtered) == 0:
        raise Exception("No genomewide-significant clumps found.")
    return filtered


@task()
def clump(
    chrom_sumstat_pattern: str,
    bfile_template_pattern: str,
    chromosomes: list[str],
    output_path: str,
    maf: float,
    kb: float,
    p1: float,
    p2: float,
    r2: float,
) -> Dict[str, File]:
    """
    Perform clumping on summary statistics for multiple chromosomes.

    Parameters:
        chrom_sumstat_pattern (str): A string pattern for the paths to the summary
          statistics files for each chromosome.
          This pattern should contain "{CHROM}", which will be replaced with the
          chromosome name.
        bfile_template_pattern (str): A string pattern for the paths to the
          .bed, .bim, and .fam files (binary PLINK files) for each chromosome.
          This pattern should contain "{CHROM}", which will be replaced with
          the chromosome name.
        chromosomes (list[str]): A list of chromosome names to perform clumping on.
        output_path (str): The path where the output files will be saved.
        maf (float): The minor allele frequency threshold for clumping.
        kb (float): The kilobase window for clumping.
        p1 (float): The p-value threshold for lead variant
        p2 (float): The p-value threshold for all variants
        r2 (float): The r-squared threshold for clumping.

    Returns:
        Dict[str, File]: A dictionary mapping chromosome names to files containing
          clumped summary statistics for each chromosome.
    """
    logger.info(
        f"Running clumping on {chrom_sumstat_pattern} for chromosomes {chromosomes}"
    )
    chrom_to_clumping_report: Dict[str, File] = {
        chrom: clump_single_chromosome(
            sumstat_path=chrom_sumstat_pattern.replace("{CHROM}", chrom),
            bfile_prefix=bfile_template_pattern.replace("{CHROM}", chrom),
            output_path=os.path.join(output_path, f"chr{chrom}.clumped"),
            p1=p1,
            p2=p2,
            maf=maf,
            r2=r2,
            kb=kb,
        )
        for chrom in chromosomes
        if File(chrom_sumstat_pattern.replace("{CHROM}", chrom)).exists()
    }

    chrom_to_clumped_sumstat: Dict[str, Optional[File]] = {
        chrom: create_clumped_sumstat(
            clumping_report=clumping_report,
            sumstat=File(chrom_sumstat_pattern.replace("{CHROM}", chrom)),
        )
        for chrom, clumping_report in chrom_to_clumping_report.items()
    }

    return filter_empty_clumped_sumstats(chrom_to_clumped_sumstat)
