from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..core import ds
from ..utils import CSV_DIR, load_tune, load_table, DATA_DIR, pca_cov, progress_iter
from .base import _load_factors, _read_portfolios, _build_portfolio_blocks, _finalize_result_table


def _process_robustness_factor(
    j: int,
    test_factor: np.ndarray,
    control_factor: np.ndarray,
    port_3x2b: np.ndarray,
    port_5x5b: np.ndarray,
    port_202: np.ndarray,
    ht_pca_final: np.ndarray,
    tune_center: np.ndarray,
    tune_center_pca: np.ndarray,
    tune_center_202: np.ndarray,
    tune_center_25: np.ndarray,
    tune_center_enet: np.ndarray,
) -> dict:
    """Process a single factor for robustness analysis."""
    gt = test_factor[:, j][None, :]
    ht = control_factor.T

    model_ds = ds(port_3x2b.T, gt, ht, -np.log(tune_center[j, 0]), -np.log(tune_center[j, 1]), 1, 100)
    model_pca = ds(port_3x2b.T, gt, ht_pca_final, -np.log(tune_center_pca[j, 0]), -np.log(tune_center_pca[j, 1]), 1, 100)

    model_202 = ds(port_202.T, gt, ht, -np.log(tune_center_202[j, 0]), -np.log(tune_center_202[j, 1]), 1, 100)
    model_ds25 = ds(port_5x5b.T, gt, ht, -np.log(tune_center_25[j, 0]), -np.log(tune_center_25[j, 1]), 1, 100)
    model_glmnet = ds(port_3x2b.T, gt, ht, -np.log(tune_center_enet[j, 0]), -np.log(tune_center_enet[j, 1]), 0.5, 100)

    return {
        "lambda_ds": float(model_ds["gamma_ds"][0]),
        "tstat_ds": float(model_ds["lambdag_ds"][0] / model_ds["se_ds"][0]),
        "lambda_glmnet": float(model_glmnet["gamma_ds"][0]),
        "tstat_glmnet": float(model_glmnet["lambdag_ds"][0] / model_glmnet["se_ds"][0]),
        "lambda_ds25": float(model_ds25["gamma_ds"][0]),
        "tstat_ds25": float(model_ds25["lambdag_ds"][0] / model_ds25["se_ds"][0]),
        "lambda_202": float(model_202["gamma_ds"][0]),
        "tstat_202": float(model_202["lambdag_ds"][0] / model_202["se_ds"][0]),
        "lambda_pca": float(model_pca["gamma_ds"][0]),
        "tstat_pca": float(model_pca["lambdag_ds"][0] / model_pca["se_ds"][0]),
    }


def run_robustness(output_dir: Path | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """Parallel robustness workflow."""
    if output_dir is None:
        output_dir = CSV_DIR

    print("Running robustness analysis...")

    _, rf, factors, summary = _load_factors()
    factorname_full = summary["Descpription"].astype(str).to_numpy()
    year_pub = summary["Year"].to_numpy()

    port_5x5 = _read_portfolios("port_5x5", rf)
    port_3x2 = _read_portfolios("port_3x2", rf)
    port_202 = _read_portfolios("port202", rf, divide_by_100=True)
    port_5x5_id = load_table(DATA_DIR / "port_5x5_id.csv")
    port_3x2_id = load_table(DATA_DIR / "port_3x2_id.csv")

    kk = 10
    include_3x2 = np.where(port_3x2_id["min_stk6"].to_numpy() >= kk)[0]
    port_3x2b = _build_portfolio_blocks(port_3x2, include_3x2, 6)

    include_5x5 = np.where(port_5x5_id["min_stk"].to_numpy() >= kk)[0]
    port_5x5b = _build_portfolio_blocks(port_5x5, include_5x5, 25)

    tune = load_tune(DATA_DIR / "tune_robustness.mat")
    tune_center = np.asarray(tune.get("tune_center"))
    tune_center_pca = np.asarray(tune.get("tune_center_pca"))
    tune_center_202 = np.asarray(tune.get("tune_center_202"))
    tune_center_25 = np.asarray(tune.get("tune_center_25"))
    tune_center_enet = np.asarray(tune.get("tune_center_enet"))

    control_factor = factors[:, year_pub < 2012]
    test_list = np.where(year_pub >= 2012)[0]
    test_factor = factors[:, test_list]

    # Pre-compute PCA components handling NaNs
    ht_pca_scores = pca_cov(control_factor)
    # Standardize scores using nanstd (matching MATLAB diag(nanstd(ht_pca)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        ht_pca_std = np.nanstd(ht_pca_scores, axis=0, ddof=1)
    ht_pca_std[ht_pca_std == 0] = 1.0
    ht_pca_final = (ht_pca_scores / ht_pca_std).T

    # Parallel processing
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_process_robustness_factor)(
            j, test_factor, control_factor, port_3x2b, port_5x5b, port_202,
            ht_pca_final,
            tune_center, tune_center_pca, tune_center_202, tune_center_25, tune_center_enet
        )
        for j in progress_iter(range(len(test_list)), desc="robustness factors", leave=False)
    )

    result = pd.DataFrame(rows)
    for col in ["lambda_ds", "lambda_glmnet", "lambda_ds25", "lambda_202", "lambda_pca"]:
        result[col] *= 10000
    # Reorder columns to match MATLAB robustness.csv
    cols = [
        "TestList", "factornames", "lambda_ds", "tstat_ds", "lambda_ds25", "tstat_ds25",
        "lambda_202", "tstat_202", "lambda_glmnet", "tstat_glmnet", "lambda_pca", "tstat_pca"
    ]
    return _finalize_result_table(result, cols, output_dir / "robustness.csv", factorname_full, test_list)
