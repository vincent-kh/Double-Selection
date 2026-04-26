from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .glmnet_adapter import _lambda_grid, fit_and_select, fit_glmnet_single
from .inference import infer_price_risk
from .selection import stepwise_select, tscv_third_selection
from .utils import pairwise_cov, progress_iter


def ds(
    ri: np.ndarray,
    gt: np.ndarray,
    ht: np.ndarray,
    tune1: float,
    tune2: float,
    alpha: float = 1.0,
    seednum: int | None = None,
) -> dict[str, np.ndarray]:
    """Double selection following MATLAB implementation."""
    if seednum is None:
        seednum = 100

    n = ri.shape[0]
    p = ht.shape[0]
    d = gt.shape[0]

    cov_g = pairwise_cov(np.vstack([gt, ri]).T)[d:, :d]
    cov_h = pairwise_cov(np.vstack([ht, ri]).T)[p:, :p]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        er = np.nanmean(ri, axis=1)

    beta = np.full((n, p), np.nan)
    for i in progress_iter(range(p), desc="DS first selection", leave=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            beta[:, i] = cov_h[:, i] / np.nanvar(ht[i, :], ddof=1)
    penalty = np.mean(beta**2, axis=0)
    penalty = penalty / np.mean(penalty)

    x1 = cov_h * penalty
    sel1, coeff1 = fit_and_select(x1, er, np.exp(-tune1), alpha=alpha, standardize=False, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        err1 = np.nanmean((er - (coeff1[0] + x1 @ coeff1[1:])) ** 2)

    sel2_list: list[int] = []
    err2 = np.full(d, np.nan)
    for i in progress_iter(range(d), desc="DS second selection", leave=False):
        sel2_i, coeff2 = fit_and_select(x1, cov_g[:, i], np.exp(-tune2), alpha=alpha, standardize=False, fit_intercept=True)
        sel2_list.extend(sel2_i.tolist())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            err2[i] = np.nanmean((cov_g[:, i] - (coeff2[0] + x1 @ coeff2[1:])) ** 2)
    sel2 = np.unique(np.asarray(sel2_list, dtype=int)) if sel2_list else np.array([], dtype=int)

    sel3_list: list[int] = []
    lambdas = _lambda_grid()
    for i in progress_iter(range(d), desc="DS third selection", leave=False):
        out = tscv_third_selection(ri, gt[i, :][None, :], ht, lambdas=lambdas, k_folds=10, j_rep=1, alpha=alpha, seednum=seednum)
        sel3_list.extend(out["sel3"].tolist())
    sel3 = np.unique(np.asarray(sel3_list, dtype=int)) if sel3_list else np.array([], dtype=int)

    dsout = infer_price_risk(ri, gt, ht, sel1, sel2, sel3)
    ssout = infer_price_risk(ri, gt, ht, sel1, np.array([], dtype=int), sel3)

    return {
        "lambdag_ds": dsout["lambdag"],
        "se_ds": dsout["se"],
        "gamma_ds": dsout["gamma"],
        "lambdag_ss": ssout["lambdag"],
        "se_ss": ssout["se"],
        "gamma_ss": ssout["gamma"],
        "sel1": sel1,
        "sel2": sel2,
        "sel3": sel3,
        "select": np.unique(np.concatenate([sel1, sel2])) if sel1.size or sel2.size else np.array([], dtype=int),
        "err1": err1,
        "err2": err2,
    }


def price_risk_ols(ri: np.ndarray, gt: np.ndarray, ht: np.ndarray) -> dict[str, np.ndarray]:
    n, t = ri.shape
    d = gt.shape[0]
    p = ht.shape[0]

    cov_h = np.full((n, p), np.nan)
    for nn in progress_iter(range(n), desc="OLS covariances (h)", leave=False):
        cov_h[nn, :] = pairwise_cov(np.vstack([ri[nn, :], ht]).T)[1:, :1].reshape(-1)
    cov_g = np.full((n, d), np.nan)
    for nn in progress_iter(range(n), desc="OLS covariances (g)", leave=False):
        cov_g[nn, :] = pairwise_cov(np.vstack([ri[nn, :], gt]).T)[1:, :1].reshape(-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        er = np.nanmean(ri, axis=1)
    m0 = np.eye(n) - np.ones((n, n)) / n
    x = np.concatenate([cov_g, cov_h], axis=1)
    x_zero = np.concatenate([np.ones((n, 1)), cov_g, cov_h], axis=1)
    lambda_full_zero = np.linalg.pinv(x_zero.T @ x_zero) @ (x_zero.T @ er)
    lambda_full = np.linalg.pinv(x.T @ m0 @ x) @ (x.T @ m0 @ er)
    lambdag_ols = lambda_full[:d]

    nomissing = np.where(np.sum(np.isnan(np.vstack([ht, gt])), axis=0) == 0)[0]
    lnm = len(nomissing)
    zthat3 = np.full((d, lnm), np.nan)
    for i in progress_iter(range(d), desc="OLS residuals", leave=False):
        h_sel = ht[:, nomissing]
        proj = np.eye(lnm) - h_sel.T @ np.linalg.pinv(h_sel @ h_sel.T) @ h_sel
        zthat3[i, :] = proj @ gt[i, nomissing].T
    sigma_zhat3 = zthat3 @ zthat3.T / lnm

    temp3 = np.zeros((d, d), dtype=float)
    for idx, l in enumerate(progress_iter(nomissing, desc="OLS variance", leave=False)):
        mt = 1.0 - lambda_full @ np.concatenate([gt[:, l], ht[:, l]])
        temp3 += mt**2 * (np.linalg.pinv(sigma_zhat3) @ np.outer(zthat3[:, idx], zthat3[:, idx]) @ np.linalg.pinv(sigma_zhat3))

    avar_lambdag3 = np.diag(temp3) / lnm
    se3 = np.sqrt(avar_lambdag3 / lnm)

    vt = np.vstack([gt[:, nomissing], ht[:, nomissing]])
    v_bar = vt - np.mean(vt, axis=1, keepdims=True)
    var_v = v_bar @ v_bar.T / lnm
    lambda_ols = np.diag(var_v) * lambda_full

    return {
        "lambdag_ols": lambdag_ols,
        "se_ols": se3,
        "lambda_ols": lambda_ols,
        "lambda_ols_zero": lambda_full_zero,
    }


def ds_stepwise(ri: np.ndarray, gt: np.ndarray, ht: np.ndarray, start: Sequence[int]) -> dict[str, np.ndarray]:
    n = ri.shape[0]
    p = ht.shape[0]
    d = gt.shape[0]

    cov_g = pairwise_cov(np.vstack([gt, ri]).T)[d:, :d]
    cov_h = pairwise_cov(np.vstack([ht, ri]).T)[p:, :p]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        er = np.nanmean(ri, axis=1)

    sel1 = stepwise_select(cov_h, er, start)
    sel2_list: list[int] = []
    for i in progress_iter(range(d), desc="stepwise second", leave=False):
        sel2_list.extend(stepwise_select(cov_h, cov_g[:, i], start).tolist())
    sel2 = np.unique(np.asarray(sel2_list, dtype=int)) if sel2_list else np.array([], dtype=int)
    sel3_list: list[int] = []
    for i in progress_iter(range(d), desc="stepwise third", leave=False):
        sel3_list.extend(stepwise_select(ht.T, gt[i, :], start).tolist())
    sel3 = np.unique(np.asarray(sel3_list, dtype=int)) if sel3_list else np.array([], dtype=int)

    dsout = infer_price_risk(ri, gt, ht, sel1, sel2, sel3)
    ssout = infer_price_risk(ri, gt, ht, sel1, np.array([], dtype=int), sel3)

    return {
        "lambdag_ds": dsout["lambdag"],
        "se_ds": dsout["se"],
        "gamma_ds": dsout["gamma"],
        "lambdag_ss": ssout["lambdag"],
        "se_ss": ssout["se"],
        "gamma_ss": ssout["gamma"],
        "sel1": sel1,
        "sel2": sel2,
        "sel3": sel3,
        "select": np.unique(np.concatenate([sel1, sel2])) if sel1.size or sel2.size else np.array([], dtype=int),
    }
