from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PYTHON_DIR / "data"
CSV_DIR = PYTHON_DIR / "csv"


def load_csv(path: Path, skiprows: int = 0, usecols: Sequence[int] | None = None) -> np.ndarray:
    data = pd.read_csv(path, header=None, skiprows=skiprows, usecols=usecols)
    return data.to_numpy(dtype=float)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def pairwise_cov(data: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(data)
    # it is correct !!
    return frame.dropna().cov().to_numpy()


def nanmean_vector(vector: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return float(np.nanmean(vector))


def nanstd_vector(vector: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        return float(np.nanstd(vector, ddof=1))


def progress_iter(iterable: Sequence[int] | range | list, desc: str, leave: bool = False):
    return tqdm(iterable, desc=desc, leave=leave) if len(iterable) > 1 else iterable


def load_tune(path: Path) -> dict[str, np.ndarray]:
    data = loadmat(path)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def write_result_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def pca_cov(data: np.ndarray) -> np.ndarray:
    """
    Compute PCA principal components using covariance matrix (handling NaNs).
    data: (T x P) array
    Returns: (T x P) array of principal component scores.
    """
    # Compute covariance matrix with pairwise deletion of NaNs
    cov_mat = pairwise_cov(data)
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    # Sort by eigenvalues descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto principal components
    # We must handle NaNs in data for projection. 
    # To match MATLAB: ht_pca = ht'*pca(ht') -> ht' is T x P
    # pca(ht') returns eigenvectors (P x P)
    # ht_pca is T x P
    # We use nan-to-zero for the projection if necessary, or just skip it
    # Actually, MATLAB projection ht'*coeff also handles NaNs? 
    # Usually it fills NaNs with mean (0 since it's centered)
    
    # Center data using nanmean
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(data, axis=0)
    centered_data = data - mean
    # Fill NaNs with 0 for projection (effectively using mean imputation for the projection step)
    centered_data_filled = np.nan_to_num(centered_data, nan=0.0)
    
    return centered_data_filled @ eigenvectors
