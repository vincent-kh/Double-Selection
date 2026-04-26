from __future__ import annotations

import ctypes
import os
import warnings

import numpy as np
import scipy
from sklearn.linear_model import ElasticNet

from .utils import PYTHON_DIR

_GLMNET_FUNCS: tuple | None = None
_GLMNET_WARNING_EMITTED = False
_GLMNET_RUNTIME_HANDLE = None


def _ensure_glmnet_python_scipy_compat() -> None:
    """Provide removed scipy top-level aliases expected by glmnet_python."""
    alias_map = {
        "empty": np.empty,
        "ones": np.ones,
        "zeros": np.zeros,
        "reshape": np.reshape,
        "tile": np.tile,
        "transpose": np.transpose,
        "array": np.array,
        "float64": np.float64,
        "int32": np.int32,
        "integer": np.int32,
        "signedinteger": np.int32,
        "ndarray": np.ndarray,
        "inf": np.inf,
        "isinf": np.isinf,
        "isnan": np.isnan,
        "isfinite": np.isfinite,
        "sum": np.sum,
        "mean": np.mean,
        "dot": np.dot,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "absolute": np.absolute,
        "amax": np.amax,
        "amin": np.amin,
        "append": np.append,
        "arange": np.arange,
        "cumsum": np.cumsum,
        "floor": np.floor,
        "ceil": np.ceil,
        "mod": np.mod,
        "median": np.median,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "vstack": np.vstack,
        "column_stack": np.column_stack,
        "row_stack": np.row_stack,
        "unique": np.unique,
        "argsort": np.argsort,
        "sort": np.sort,
        "bincount": np.bincount,
        "all": np.all,
        "any": np.any,
        "random": np.random,
        "size": np.size,
        "diff": np.diff,
        "round_": np.round,
        "NAN": np.nan,
        "NaN": np.nan,
    }

    for name, value in alias_map.items():
        if not hasattr(scipy, name):
            setattr(scipy, name, value)


def _ensure_glmnet_python_runtime() -> None:
    """Add the local libgfortran 3 to the loader search path when present."""
    global _GLMNET_RUNTIME_HANDLE
    runtime_lib = PYTHON_DIR / "libgfortran.so.3"
    if not runtime_lib.exists():
        return

    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    path_parts = [part for part in current_ld_library_path.split(os.pathsep) if part]
    python_dir_str = str(PYTHON_DIR)
    if python_dir_str not in path_parts:
        path_parts.insert(0, python_dir_str)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(path_parts)

    if _GLMNET_RUNTIME_HANDLE is None:
        _GLMNET_RUNTIME_HANDLE = ctypes.CDLL(str(runtime_lib), mode=ctypes.RTLD_GLOBAL)


def get_glmnet_functions():
    global _GLMNET_FUNCS
    if _GLMNET_FUNCS is None:
        _ensure_glmnet_python_scipy_compat()
        _ensure_glmnet_python_runtime()
        try:
            from glmnet_python import glmnet as glmnet_module
            from glmnet_python import glmnetCoef as glmnet_coef_module
        except Exception as exc:  # pragma: no cover - import guard for runtime environment
            raise ImportError(
                "Failed to import glmnet_python. Install glmnet_python from bbalasub1/glmnet_python and ensure its Fortran runtime is available."
            ) from exc
        glmnet_fit = getattr(glmnet_module, "glmnet", glmnet_module)
        glmnet_coef = getattr(glmnet_coef_module, "glmnetCoef", glmnet_coef_module)
        _GLMNET_FUNCS = (glmnet_fit, glmnet_coef)
    return _GLMNET_FUNCS


def _prepare_design_matrix(x: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, np.array([]), np.array([])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(x, axis=0)
    centered = x - mean
    if standardize:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            scale = np.nanstd(centered, axis=0, ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        return centered / scale, mean, scale
    scale = np.ones(x.shape[1], dtype=float)
    return x, mean, scale


def _fit_elasticnet_fallback(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    standardize: bool,
    fit_intercept: bool,
) -> tuple[float, np.ndarray]:
    x_use, x_mean, x_scale = _prepare_design_matrix(x, standardize=standardize)
    y_use = np.asarray(y, dtype=float).reshape(-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = ElasticNet(
            alpha=float(lam),
            l1_ratio=float(alpha),
            fit_intercept=fit_intercept,
            max_iter=20000,
            tol=1e-7,
            warm_start=False,
            selection="cyclic",
            random_state=0,
        )
        model.fit(x_use, y_use)

    coef = model.coef_.copy() / x_scale
    intercept = float(model.intercept_)
    if standardize:
        intercept = intercept - np.sum((x_mean / x_scale) * model.coef_)
    return intercept, coef


def _single_glmnet_coef_vector(coef_output: np.ndarray) -> np.ndarray:
    """Reduce glmnetCoef output to a single coefficient vector for one lambda."""
    coef_array = np.asarray(coef_output, dtype=np.float64)
    while coef_array.ndim > 1:
        index = (slice(None),) + (0,) * (coef_array.ndim - 1)
        coef_array = coef_array[index]
    return coef_array.reshape(-1)


def fit_glmnet_single(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    standardize: bool = False,
    fit_intercept: bool = True,
) -> tuple[float, np.ndarray]:
    """Fit elastic net at a single lambda value with glmnet_python (MATLAB-style)."""
    x_use = np.asarray(x, dtype=np.float64)
    y_use = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    lambda_vec = np.asarray([float(lam)], dtype=np.float64)
    try:
        glmnet_fit, glmnet_coef = get_glmnet_functions()
        fit = glmnet_fit(
            x=x_use,
            y=y_use,
            family="gaussian",
            alpha=float(alpha),
            standardize=bool(standardize),
            intr=bool(fit_intercept),
            lambdau=lambda_vec,
        )

        coef_vec = _single_glmnet_coef_vector(glmnet_coef(fit, s=lambda_vec))
        intercept = float(coef_vec[0])
        coef = coef_vec[1:]
        return intercept, coef
    except Exception as exc:
        global _GLMNET_WARNING_EMITTED
        if not _GLMNET_WARNING_EMITTED:
            warnings.warn(
                f"glmnet_python unavailable at runtime ({exc}); falling back to sklearn ElasticNet.",
                RuntimeWarning,
                stacklevel=2,
            )
            _GLMNET_WARNING_EMITTED = True
        return _fit_elasticnet_fallback(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float).reshape(-1),
            lam=lam,
            alpha=alpha,
            standardize=standardize,
            fit_intercept=fit_intercept,
        )


def _lambda_grid(num_lambda: int = 100) -> np.ndarray:
    return np.exp(np.linspace(0.0, -35.0, num_lambda))


def fit_and_select(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    alpha: float,
    standardize: bool,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit at single lambda and extract nonzero coefficients (MATLAB-style)."""
    intercept, coefs = fit_glmnet_single(x, y, lam, alpha=alpha, standardize=standardize, fit_intercept=fit_intercept)
    coeff = coef_combine(intercept, coefs)
    sel = _select_nonzero(coeff)
    return sel, coeff


def coef_combine(intercept: float, coefs: np.ndarray) -> np.ndarray:
    """Combine intercept and coefficients into single vector."""
    return np.concatenate(([intercept], coefs))


def _select_nonzero(coef_vector: np.ndarray) -> np.ndarray:
    return np.where(np.abs(coef_vector[1:]) > 1e-10)[0]
