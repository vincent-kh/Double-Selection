from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..core import ds_stepwise
from ..utils import CSV_DIR, load_table, DATA_DIR, progress_iter, write_result_table
from .base import _load_factors, _read_portfolios, _build_portfolio_blocks


def _process_stepwise_factor(
    j: int,
    test_factor: np.ndarray,
    control_factor: np.ndarray,
    port_3x2b: np.ndarray,
    ff4: list,
) -> tuple[float, float]:
    """Process a single factor for stepwise analysis."""
    gt = test_factor[:, j][None, :]
    ht = control_factor.T
    model_fs = ds_stepwise(port_3x2b.T, gt, ht, ff4)
    tstat = float(model_fs["lambdag_ds"][0] / model_fs["se_ds"][0])
    lam = float(model_fs["gamma_ds"][0] * 10000)
    return tstat, lam


def run_stepwise(output_dir: Path | None = None, n_jobs: int = -1) -> pd.DataFrame:
    """Parallel stepwise robustness workflow."""
    if output_dir is None:
        output_dir = CSV_DIR

    print("Running stepwise analysis...")

    _, rf, factors, summary = _load_factors()
    factorname_full = summary["Descpription"].astype(str).to_numpy()
    year_pub = summary["Year"].to_numpy()
    factorname = summary["Row"].astype(str).to_numpy()

    port_3x2 = _read_portfolios("port_3x2", rf)
    port_3x2_id = load_table(DATA_DIR / "port_3x2_id.csv")

    mkt_ind = int(np.where(factorname == "MktRf")[0][0])
    smb_ind = int(np.where(factorname == "SMB")[0][0])
    hml_ind = int(np.where(factorname == "HML")[0][0])
    umd_ind = int(np.where(factorname == "UMD")[0][0])
    ff4 = [mkt_ind, smb_ind, hml_ind, umd_ind]

    kk = 10
    include_3x2 = np.where(port_3x2_id["min_stk6"].to_numpy() >= kk)[0]
    port_3x2b = _build_portfolio_blocks(port_3x2, include_3x2, 6)

    control_factor = factors[:, year_pub < 2012]
    test_list = np.where(year_pub >= 2012)[0]
    test_factor = factors[:, test_list]

    # Parallel processing
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_stepwise_factor)(j, test_factor, control_factor, port_3x2b, ff4)
        for j in progress_iter(range(len(test_list)), desc="stepwise factors", leave=False)
    )

    tstat_fs = [r[0] for r in results]
    lambda_fs = [r[1] for r in results]

    result = pd.DataFrame({
        "TestList": test_list + 1,
        "factornames": factorname_full[test_list],
        "lambda_fs": lambda_fs,
        "tstat_fs": tstat_fs,
    })
    write_result_table(result, output_dir / "robustness_stepwise.csv")
    return result
