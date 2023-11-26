import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.api as sm
from disease_relevance import PRS_ID_COLNAME
from redun import File, task

redun_namespace = "prs_association"

ID_COLNAME = "id"
PHENO_COLNAME = "pheno"


def join_data(
    prs_df: pd.DataFrame,
    prs_opt_dict: Dict,
) -> pd.DataFrame:
    """
    Create a joined dataframe with phenotype, PRS, and covariate data.
    Uses the supplied parameters to load, and verifies that ids are unique in all input files.

    Parameters:
        prs_df (pd.DataFrame): A DataFrame containing PRS data.
        prs_opt_dict:

    Returns:
        pd.DataFrame: A DataFrame containing the joined data.
    """

    # load pheno
    pheno_file = prs_opt_dict["pheno_file"]
    pheno_id_col = prs_opt_dict["pheno_id_col"]
    pheno_col = prs_opt_dict["pheno_col"]
    pheno_df = pd.read_csv(
        pheno_file, usecols=[pheno_id_col, pheno_col], sep=prs_opt_dict["pheno_sep"]
    ).rename(columns={pheno_id_col: ID_COLNAME, pheno_col: PHENO_COLNAME})

    if not pheno_df[ID_COLNAME].is_unique:
        warnings.warn(f"id column is not unique for {pheno_file}.")

    # load covar
    covar_file = prs_opt_dict["covar_file"]
    cov_id_col = prs_opt_dict["cov_id_col"]
    cov_sep = prs_opt_dict["cov_sep"]
    cov_cols = prs_opt_dict["cov_cols"]

    covar_df = pd.read_csv(
        covar_file, usecols=[cov_id_col] + cov_cols, sep=cov_sep
    ).rename(columns={cov_id_col: ID_COLNAME})
    if not covar_df[ID_COLNAME].is_unique:
        warnings.warn("id column is not unique.")

    return pheno_df.merge(prs_df, on=ID_COLNAME).merge(covar_df, on=ID_COLNAME)


def _calculate_yhat(
    df: pd.DataFrame, prs_cols: list, pheno_type: str
) -> Dict[str, np.ndarray]:
    """
    Predict the phenotype, yhat, from covariates plus/minus the polygenic score.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the data.
        prs_cols (list): A list of columns in the DataFrame that contain PRS data.
        pheno_type (str): The type of phenotype data.

    Returns:
        Dict[str, np.ndarray]: A dictionary where 'yhat_full' is the prediction of y based on the prs plus
        covariates and 'yhat_red' is the prediction of y based on covariates only.
    """
    # Phenotype.
    y = df.pheno

    red_exclude_cols = [ID_COLNAME, PHENO_COLNAME] + prs_cols

    # Predictors, full model (i.e. including PRS).
    x_full = df.drop(columns=[ID_COLNAME, PHENO_COLNAME])
    x_full.insert(0, "intercept", 1)

    # Predictors, reduced model (i.e. excluding PRS).
    x_red = df.drop(columns=red_exclude_cols)
    x_red.insert(0, "intercept", 1)

    if pheno_type == "continuous":
        # Linear regression.
        model_full = sm.OLS(y, x_full.values.astype(float))
        model_red = sm.OLS(y, x_red.values.astype(float))

    else:  # binary
        # Logistic regression.
        model_full = sm.Logit(y, x_full.values.astype(float))
        model_red = sm.Logit(y, x_red.values.astype(float))

    # Fit model.
    fit_full = model_full.fit(disp=0)
    fit_red = model_red.fit(disp=0)

    # Predict.
    yhat_full = fit_full.predict()
    yhat_red = fit_red.predict()

    return {"yhat_full": yhat_full, "yhat_red": yhat_red}


def _calculate_metrics(
    y: np.ndarray,
    yhat_full: np.ndarray,
    yhat_red: np.ndarray,
    pheno_type: str,
) -> pd.DataFrame:
    """
    Calculates the difference and ratio of metrics between the full model,
    which includes the PRS, and the reduced model, which excludes the PRS.

    Parameters:
        y (np.ndarray): The actual values of y.
        yhat_full (np.ndarray): The predicted values of y from the full model.
        yhat_red (np.ndarray): The predicted values of y from the reduced model.
        pheno_type (str): The type of phenotype data.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metrics.
    """
    out_metrics: Dict = {"metric": [], "full": [], "reduced": []}

    if pheno_type == "continuous":
        # Calculate R2.
        r2_full = metrics.r2_score(y, yhat_full)
        r2_red = metrics.r2_score(y, yhat_red)

        out_metrics["metric"].append("r2")
        out_metrics["full"].append(r2_full)
        out_metrics["reduced"].append(r2_red)

        # Calculate mean absolute error.
        mae_full = metrics.mean_absolute_error(y, yhat_full)
        mae_red = metrics.mean_absolute_error(y, yhat_red)

        out_metrics["metric"].append("mae")
        out_metrics["full"].append(mae_full)
        out_metrics["reduced"].append(mae_red)

    else:  # binary
        # Calculate AUROC.
        auroc_full = metrics.roc_auc_score(y, yhat_full)
        auroc_red = metrics.roc_auc_score(y, yhat_red)

        out_metrics["metric"].append("auroc")
        out_metrics["full"].append(auroc_full)
        out_metrics["reduced"].append(auroc_red)

        # Calculate AUPRC.
        auprc_full = metrics.average_precision_score(y, yhat_full)
        auprc_red = metrics.average_precision_score(y, yhat_red)

        out_metrics["metric"].append("auprc")
        out_metrics["full"].append(auprc_full)
        out_metrics["reduced"].append(auprc_red)

    # Calculate contrasts.
    metrics_df = pd.DataFrame(out_metrics)
    metrics_df["diff"] = metrics_df.full - metrics_df.reduced
    metrics_df["ratio"] = metrics_df.full / metrics_df.reduced

    return metrics_df


def _generate_bootstrap_dist(
    df: pd.DataFrame,
    prs_cols: list,
    pheno_type: str,
    boot_reps: int = 2000,
) -> pd.DataFrame:
    """
    Generates realizations of the bootstrap distribution under the null
    hypothesis by resampling the data then permuting the PRS.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the data.
        prs_cols (list): A list of columns in the DataFrame that contain PRS data.
        pheno_type (str): The type of phenotype data.
        boot_reps (int, optional): The number of bootstrap replicates. Defaults to 2000.

    Returns:
        pd.DataFrame: A DataFrame containing the bootstrap distribution.
    """

    boot_dist = []
    for b in range(boot_reps):
        df_boot = df.sample(frac=1.0, replace=True)
        df_boot[prs_cols] = np.random.permutation(df_boot[prs_cols].values)
        yhat_dict_boot = _calculate_yhat(
            df_boot, pheno_type=pheno_type, prs_cols=prs_cols
        )
        metrics_boot = _calculate_metrics(
            y=df_boot.pheno,
            yhat_full=yhat_dict_boot["yhat_full"],
            yhat_red=yhat_dict_boot["yhat_red"],
            pheno_type=pheno_type,
        )
        boot_dist.append(metrics_boot)
    boot_dist = pd.concat(boot_dist)
    return boot_dist


def _calculate_bootstrap_pval(
    metrics_obs: pd.DataFrame, boot_dist: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates bootstrap p-values.

    Specifically, the probability of obtaining a difference/ratio as or more
    extreme than observed under the null.

    Parameters:
        metrics_obs (pd.DataFrame): A DataFrame containing the observed metrics.
        boot_dist (pd.DataFrame): A DataFrame containing the bootstrap distribution.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated bootstrap p-values.
    """

    metrics: Dict = {"metric": [], "p_diff": [], "p_ratio": []}

    for row in metrics_obs.itertuples():
        # Subset bootstrap distribution to metric of interest.
        boot_dist_metric = boot_dist[boot_dist.metric == row.metric]
        boot_diff_vec = np.array(boot_dist_metric["diff"])
        boot_ratio_vec = np.array(boot_dist_metric["ratio"])
        b = boot_dist_metric.shape[0]
        metrics["metric"].append(row.metric)

        # Observed values for metric.
        obs_diff = row.diff
        obs_ratio = row.ratio

        # Calculate p-value for the difference.
        if obs_diff >= 0:
            p_diff = (1 + np.sum(boot_diff_vec >= obs_diff)) / (1 + b)
        else:
            p_diff = (1 + np.sum(boot_diff_vec <= obs_diff)) / (1 + b)

        # computing 2-sides p-value
        if p_diff > 0.5:
            p_diff = 1.0 - p_diff

        p_diff = min(2 * p_diff, 1)
        metrics["p_diff"].append(p_diff)

        # Calculate p-value for the ratio.
        if obs_ratio >= 1:
            p_ratio = (1 + np.sum(boot_ratio_vec >= obs_ratio)) / (1 + b)
        else:
            p_ratio = (1 + np.sum(boot_ratio_vec <= obs_ratio)) / (1 + b)

        if p_ratio > 0.5:
            p_ratio = 1.0 - p_ratio

        # converting to 2 sided p-value
        p_ratio = min(2 * p_ratio, 1)
        metrics["p_ratio"].append(p_ratio)

    return pd.DataFrame(metrics)


@task()
def eval_prs(
    df: pd.DataFrame,
    prs_cols: list,
    pheno_type: str,
    boot_reps: int,
    output_path: str,
) -> File:
    """
    Evaluate the polygenic risk score.

    Notes:
    The id_col should be a unique key.

    Parameters:
        df (pd.DataFrame): A DataFrame containing the data.
        prs_cols (list): A list of columns in the DataFrame that contain PRS data.
        boot_reps (int): The number of bootstrap replicates. Defaults to 2000.
        pheno_col (str): The column that contains phenotype data. Defaults to "pheno".
        pheno_type (str): The type of phenotype data.
        output_path (str): File path where output should be placed.

    Returns:
        File: File containing the observed value of the metrics for the full model,
        which includes the PRS, and the reduced model, which excludes the PRS.
        Calculates a p-value for the difference and ratio of the metrics between
        the full and reduced model via paired bootstrap.
    """
    assert df[ID_COLNAME].is_unique, "id must be unique."

    # Observed metrics.
    yhat_dict_obs = _calculate_yhat(df, pheno_type=pheno_type, prs_cols=prs_cols)
    metrics_obs = _calculate_metrics(
        y=df.pheno,
        yhat_full=yhat_dict_obs["yhat_full"],
        yhat_red=yhat_dict_obs["yhat_red"],
        pheno_type=pheno_type,
    )

    # Bootstrap the null distribution.
    boot_dist = _generate_bootstrap_dist(
        df=df, prs_cols=prs_cols, boot_reps=boot_reps, pheno_type=pheno_type
    )

    # Calculate the bootstrap p-value.
    boot_pval = _calculate_bootstrap_pval(metrics_obs, boot_dist)

    metrics_obs.merge(boot_pval, on="metric").to_csv(output_path, index=False, sep="\t")

    return File(output_path)


@task()
def get_prs_metrics(prs_opt_dict: Dict, prs_file: File, output_path: str) -> File:
    """
    Collect PRS metrics for all univariate and multivariate traits into one file.

    Parameters:
        prs_opt_dict (Dict): A dictionary with the following keys:
            pheno_file,
            covar_file,
            id_col,
            pheno_id_col,
            pheno_type,
            pheno_col,
            out_path,
            cov_cols
        prs_file (File): Path to prs_file.
        output_path (str): Path where the output file will be saved.

    Returns:
        File: The output file containing the PRS metrics.
    """

    pheno_type = prs_opt_dict["pheno_type"]
    if pheno_type not in ["continuous", "binary"]:
        raise ValueError(
            f"Expected pheno type to be `binary` or `continuous`. Got {pheno_type}."
        )
    boot_reps = prs_opt_dict.get("boot_reps", 2000)

    prs_df = pd.read_csv(prs_file.path, sep="\t").rename(
        columns={PRS_ID_COLNAME: ID_COLNAME}
    )

    data = join_data(prs_df=prs_df, prs_opt_dict=prs_opt_dict)

    return eval_prs(
        data,
        pheno_type=pheno_type,
        prs_cols=prs_df.columns.tolist(),
        boot_reps=boot_reps,
        output_path=os.path.join(output_path, "multivariate_associations.csv"),
    )
