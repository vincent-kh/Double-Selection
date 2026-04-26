from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import DATA_DIR, load_csv, load_table, write_result_table


def _load_factors() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    allfactors = load_csv(DATA_DIR / "factors.csv", skiprows=1)
    date = allfactors[:, 0]
    rf = allfactors[:, 1]
    factors = allfactors[:, 2:]
    summary = load_table(DATA_DIR / "summary.csv")
    return date, rf, factors, summary


def _build_portfolio_blocks(portfolios: np.ndarray, include_indices: np.ndarray, block_width: int) -> np.ndarray:
    blocks = [
        portfolios[:, ((i - 1) * block_width):(i * block_width)]
        for i in range(1, portfolios.shape[1] // block_width + 1)
        if (i - 1) in include_indices
    ]
    return np.concatenate(blocks, axis=1) if blocks else np.empty((portfolios.shape[0], 0))


def _read_portfolios(name: str, rf: np.ndarray, divide_by_100: bool = False) -> np.ndarray:
    port = pd.read_csv(DATA_DIR / f"{name}.csv", header=None).iloc[:, 1:].to_numpy(dtype=float)
    if divide_by_100:
        port = port / 100.0
    return port - rf[:, None]


def _finalize_result_table(frame: pd.DataFrame, columns: list[str], output_path: Path, factor_names: np.ndarray, test_list: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "TestList", test_list + 1)
    result.insert(1, "factornames", factor_names[test_list])
    result = result[columns]
    write_result_table(result, output_path)
    return result
