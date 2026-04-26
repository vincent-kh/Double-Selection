from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .utils import pairwise_cov


def infer_price_risk(
    ri: np.ndarray,
    gt: np.ndarray,
    ht: np.ndarray,
    sel1: Sequence[int],
    sel2: Sequence[int],
    sel3: Sequence[int],
) -> dict[str, np.ndarray]:
    n = ri.shape[0]
    d = gt.shape[0]
    cov_g = pairwise_cov(np.vstack([gt, ri]).T)[d:, :d]
    cov_h = pairwise_cov(np.vstack([ht, ri]).T)[ht.shape[0]:, :ht.shape[0]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        er = np.nanmean(ri, axis=1)
    m0 = np.eye(n) - np.ones((n, n)) / n

    nomissing = np.where(np.sum(np.isnan(np.vstack([ht, gt])), axis=0) == 0)[0]
    select = np.unique(np.concatenate([np.asarray(sel1, dtype=int), np.asarray(sel2, dtype=int)])) if len(sel1) or len(sel2) else np.array([], dtype=int)

    if select.size == 0:
        x = cov_g
    else:
        x = np.concatenate([cov_g, cov_h[:, select]], axis=1)
    lambda_full = np.linalg.pinv(x.T @ m0 @ x) @ (x.T @ m0 @ er)
    lambdag = lambda_full[:d]

    if len(sel3) == 0:
        zthat = gt[:, nomissing]
    else:
        h_sel = ht[np.asarray(sel3, dtype=int)[:, None], nomissing]
        proj = np.eye(len(nomissing)) - h_sel.T @ np.linalg.pinv(h_sel @ h_sel.T) @ h_sel
        zthat = proj @ gt[:, nomissing].T
        zthat = zthat.T
    sigma_zhat = zthat @ zthat.T / len(nomissing)

    temp2 = np.zeros((d, d), dtype=float)
    for pos, l in enumerate(nomissing):
        mt = 1.0 - lambda_full @ np.concatenate([gt[:d, l], ht[select, l] if select.size else np.array([], dtype=float)])
        temp2 += mt**2 * (np.linalg.pinv(sigma_zhat) @ np.outer(zthat[:, pos], zthat[:, pos]) @ np.linalg.pinv(sigma_zhat))

    avar_lambdag = np.diag(temp2) / len(nomissing)
    se = np.sqrt(avar_lambdag / len(nomissing))

    vt = np.vstack([gt[:, nomissing], ht[select][:, nomissing] if select.size else np.empty((0, len(nomissing)))])
    v_bar = vt - np.mean(vt, axis=1, keepdims=True)
    var_v = v_bar @ v_bar.T / len(nomissing)
    gamma = np.diag(var_v) * lambda_full

    return {"lambdag": lambdag, "se": se, "gamma": gamma}
