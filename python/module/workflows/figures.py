from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from ..core import ds
from ..utils import progress_iter, load_tune, load_table, DATA_DIR
from .base import _load_factors, _read_portfolios


def _process_figure1_iteration(k: int, tune_sel: np.ndarray, port_3x2b: np.ndarray, gt: np.ndarray, ht: np.ndarray) -> np.ndarray:
    """Process a single iteration for Figure 1 selection rate."""
    model_ds = ds(port_3x2b.T, gt, ht, -np.log(tune_sel[k, 0]), -np.log(tune_sel[k, 1]), 1, 100)
    # model_ds['sel1'] contains indices of selected control factors
    selection = np.zeros(ht.shape[0])
    if model_ds["sel1"].size > 0:
        selection[model_ds["sel1"]] = 1
    return selection


def run_figure1(n_jobs: int = -1) -> np.ndarray:
    """Parallel computation of selection rates for Figure 1."""
    print("Computing selection rates for Figure 1...")

    _, rf, factors, summary = _load_factors()
    year_pub = summary["Year"].to_numpy()

    port_3x2 = _read_portfolios("port_3x2", rf)
    port_3x2_id = load_table(DATA_DIR / "port_3x2_id.csv")

    kk = 10
    include_3x2 = np.where(port_3x2_id["min_stk6"].to_numpy() >= kk)[0]
    port_3x2b = np.concatenate([port_3x2[:, ((i - 1) * 6):(i * 6)] for i in range(1, factors.shape[1] + 1) if (i - 1) in include_3x2], axis=1)

    tune = load_tune(DATA_DIR / "tune_figure1.mat")
    tune_sel = np.asarray(tune.get("tune_sel_150"))

    control_factor = factors[:, year_pub < 2012]
    test_list = np.where(year_pub >= 2012)[0]
    test_factor = factors[:, test_list]

    # Just use the 1st test factor as in MATLAB script
    gt = test_factor[:, 0][None, :]
    ht = control_factor.T

    # Parallel processing across 200 tuning parameter sets
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_figure1_iteration)(k, tune_sel, port_3x2b, gt, ht)
        for k in progress_iter(range(len(tune_sel)), desc="Figure 1 iterations", leave=False)
    )

    # results is list of 200 arrays, each 135-long
    selection_rates = np.mean(results, axis=0)
    return selection_rates
