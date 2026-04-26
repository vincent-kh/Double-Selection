from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold

from .glmnet_adapter import _lambda_grid, fit_glmnet_single, get_glmnet_functions


def crossval_kfold_indices(n_samples: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.arange(n_samples))]


def tscv_third_selection(
    ri: np.ndarray,
    gt: np.ndarray,
    ht: np.ndarray,
    lambdas: np.ndarray | None = None,
    k_folds: int = 10,
    j_rep: int = 1,
    alpha: float = 1.0,
    seednum: int | None = None,
) -> dict[str, np.ndarray]:
    """Time series cross-validation for third selection (MATLAB-style)."""
    if seednum is None:
        seednum = 100
    if lambdas is None:
        lambdas = _lambda_grid()

    nomissing = np.where(np.sum(np.isnan(np.vstack([ht, gt])), axis=0) == 0)[0]
    
    # Subset to non-missing data for the entire CV process
    ht_raw = ht[:, nomissing]
    gt_raw = gt[:, nomissing]
    
    cvm = np.full((len(lambdas), k_folds * j_rep), np.nan)
    cv_idx = 0

    for j in range(j_rep):
        # Use exact random state matching MATLAB rng(seednum + j)
        splitter = KFold(n_splits=k_folds, shuffle=True, random_state=seednum + j + 1)
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(nomissing)))):
            ht_train = ht_raw[:, train_idx]
            gt_train = gt_raw[:, train_idx]
            ht_test = ht_raw[:, test_idx]
            gt_test = gt_raw[:, test_idx]
            
            # Use internal standardization to match MATLAB opts3 = struct('standardize', true)
            # This is critical for aligning the lambda grid's effect
            try:
                glmnet_fit, _ = get_glmnet_functions()
                fit = glmnet_fit(
                    x=ht_train.T,
                    y=gt_train.T,
                    family="gaussian",
                    alpha=float(alpha),
                    standardize=True,
                    intr=False,
                    lambdau=lambdas,
                )
                # Predict on test set
                _, glmnet_coef = get_glmnet_functions()
                # model.beta in MATLAB is equivalent to glmnet_coef output without intercept
                betas = glmnet_coef(fit)
                if betas.ndim > 1:
                    # betas is (p+1) x len(lambdas). Intercept is first row (all zeros since intr=False)
                    betas = betas[1:, :] 
                    preds = ht_test.T @ betas # T_test x n_lambda
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Mean of empty slice")
                        errors = np.nanmean((gt_test.T - preds) ** 2, axis=0)
                    # Align indices in case fit returned fewer lambdas
                    fit_lambdas = np.asarray(fit['lambdau']).reshape(-1)
                    idx_map = [np.where(np.abs(lambdas - l) < 1e-10)[0][0] for l in fit_lambdas]
                    cvm[idx_map, cv_idx] = errors
            except Exception:
                # Fallback to manual loop if vectorized fit fails
                for l_idx, lam in enumerate(lambdas):
                    _, coefs = fit_glmnet_single(ht_train.T, gt_train.T, lam, alpha=alpha, standardize=True, fit_intercept=False)
                    pred = ht_test.T @ coefs
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Mean of empty slice")
                        cvm[l_idx, cv_idx] = np.nanmean((gt_test.T - pred) ** 2)
            cv_idx += 1

    # Find best lambda using one-standard-error rule
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        with np.errstate(divide="ignore", invalid="ignore"):
            cvm_mean = np.nanmean(cvm, axis=1)
            # Count non-NaN values per lambda to avoid ddof warnings and div by zero
            n_valid = np.sum(~np.isnan(cvm), axis=1)
            cvm_se = np.nanstd(cvm, axis=1, ddof=1)
            cvm_se[n_valid <= 1] = 0  # Cannot compute SE with 0 or 1 point
            cvm_se = cvm_se / np.sqrt(n_valid)
        
    if np.all(np.isnan(cvm_mean)):
        # Fallback if everything failed (should be rare)
        l_best = 0
    else:
        l_best = int(np.nanargmin(cvm_mean))
        
    cvm_ub = cvm_mean[l_best] + cvm_se[l_best]
    
    # MATLAB: l3_1se = find(cvm333(1:l_sel3) >= cvm33ub, 1,'last')
    # This finds the largest index (simplest model) that is STILL ABOVE the 1SE threshold.
    with np.errstate(invalid="ignore"):
        eligible_above = np.where(cvm_mean[:l_best + 1] >= cvm_ub)[0]
    l_1se = int(eligible_above[-1]) if eligible_above.size > 0 else l_best

    # Final fit on all data
    _, coefs_final = fit_glmnet_single(ht_raw.T, gt_raw.T, lambdas[l_1se], alpha=alpha, standardize=True, fit_intercept=False)
    sel3 = np.where(np.abs(coefs_final) > 1e-10)[0]

    return {"sel3": sel3}


def stepwise_select(x: np.ndarray, y: np.ndarray, start: Sequence[int]) -> np.ndarray:
    selected = list(dict.fromkeys(int(v) for v in start))
    remaining = [idx for idx in range(x.shape[1]) if idx not in selected]

    def bic_for(columns: list[int]) -> float:
        design = sm.add_constant(x[:, columns], has_constant="add")
        fit = sm.OLS(y, design, missing="drop").fit()
        return float(fit.bic)

    current_bic = bic_for(selected) if selected else bic_for([])
    improved = True
    while improved and remaining:
        improved = False
        best_choice = None
        best_bic = current_bic
        for candidate in remaining:
            trial = selected + [candidate]
            bic = bic_for(trial)
            if bic < best_bic - 1e-8:
                best_bic = bic
                best_choice = candidate
        if best_choice is not None:
            selected.append(best_choice)
            remaining.remove(best_choice)
            current_bic = best_bic
            improved = True
    return np.asarray(sorted(selected), dtype=int)
