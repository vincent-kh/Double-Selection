from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..core import ds, price_risk_ols
from ..utils import CSV_DIR, nanmean_vector, nanstd_vector, progress_iter, load_tune, load_table, DATA_DIR
from .base import _load_factors, _read_portfolios, _finalize_result_table


def _process_main_factor(
    j: int,
    test_factor: np.ndarray,
    control_factor: np.ndarray,
    port_3x2b: np.ndarray,
    ff3_factors: np.ndarray,
    tune_center: np.ndarray,
) -> dict:
    """Process a single factor for main analysis."""
    gt = test_factor[:, j][None, :]
    ht = control_factor.T

    model_ds = ds(port_3x2b.T, gt, ht, -np.log(tune_center[j, 0]), -np.log(tune_center[j, 1]), 1, 100)
    model_ss = ds(port_3x2b.T, gt, ht, -np.log(tune_center[j, 0]), -np.log(1), 1, 100)
    model_ols = price_risk_ols(port_3x2b.T, gt, ht)
    model_ff3 = price_risk_ols(port_3x2b.T, gt, ff3_factors)
    avg = nanmean_vector(gt.reshape(-1))
    tstat_avg = avg / nanstd_vector(gt.reshape(-1)) * np.sqrt(np.sum(~np.isnan(gt)))

    return {
        "tstat_ds": float(model_ds["lambdag_ds"][0] / model_ds["se_ds"][0]),
        "lambda_ds": float(model_ds["gamma_ds"][0]),
        "tstat_ss": float(model_ss["lambdag_ss"][0] / model_ss["se_ss"][0]),
        "lambda_ss": float(model_ss["gamma_ss"][0]),
        "avg": float(avg),
        "tstat_avg": float(tstat_avg),
        "lambda_ols": float(model_ols["lambda_ols"][0]),
        "tstat_ols": float(model_ols["lambdag_ols"][0] / model_ols["se_ols"][0]),
        "lambda_FF3": float(model_ff3["lambda_ols"][0]),
        "tstat_FF3": float(model_ff3["lambdag_ols"][0] / model_ff3["se_ols"][0]),
    }


def run_main(output_dir: Path | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """Parallel main workflow."""
    if output_dir is None:
        output_dir = CSV_DIR

    print("Running main analysis...")

    _, rf, factors, summary = _load_factors()
    factorname = summary["Row"].astype(str).to_numpy()
    factorname_full = summary["Descpription"].astype(str).to_numpy()
    year_pub = summary["Year"].to_numpy()

    port_3x2 = _read_portfolios("port_3x2", rf)
    port_3x2_id = load_table(DATA_DIR / "port_3x2_id.csv")

    mkt_ind = int(np.where(factorname == "MktRf")[0][0])
    smb_ind = int(np.where(factorname == "SMB")[0][0])
    hml_ind = int(np.where(factorname == "HML")[0][0])

    kk = 10
    include_3x2 = np.where(port_3x2_id["min_stk6"].to_numpy() >= kk)[0]
    blocks = [port_3x2[:, ((i - 1) * 6):(i * 6)] for i in range(1, factors.shape[1] + 1) if (i - 1) in include_3x2]
    port_3x2b = np.concatenate(blocks, axis=1) if blocks else np.empty((port_3x2.shape[0], 0))

    tune = load_tune(DATA_DIR / "tune_main.mat")
    tune_center = np.asarray(tune.get("tune_center"))

    control_factor = factors[:, year_pub < 2012]
    test_list = np.where(year_pub >= 2012)[0]
    test_factor = factors[:, test_list]
    ff3 = factors[:, [mkt_ind, smb_ind, hml_ind]].T

    # Parallel processing
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_process_main_factor)(j, test_factor, control_factor, port_3x2b, ff3, tune_center)
        for j in progress_iter(range(len(test_list)), desc="main factors", leave=False)
    )

    result = pd.DataFrame(rows)
    result["lambda_ds"] *= 10000
    result["lambda_ss"] *= 10000
    result["lambda_FF3"] *= 10000
    result["lambda_ols"] *= 10000
    result["avg"] *= 10000
    # Reorder columns to match MATLAB main.csv
    cols = [
        "TestList", "factornames", "lambda_ds", "tstat_ds", "lambda_ss", "tstat_ss",
        "lambda_FF3", "tstat_FF3", "lambda_ols", "tstat_ols", "avg", "tstat_avg"
    ]
    print(result.head())
    return _finalize_result_table(result, cols, output_dir / "main.csv", factorname_full, test_list)
