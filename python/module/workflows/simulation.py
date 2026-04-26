from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..utils import DATA_DIR, load_tune


def run_simulation(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2]

    sim_name = "_T_480_n_300_p_100"
    mat = load_tune(DATA_DIR / f"TS_CMA_Simu{sim_name}.mat")
    lambdag_ds = np.asarray(mat.get("lambdag_ds"))
    lambdag_ss = np.asarray(mat.get("lambdag_ss"))
    lambdag_ns = np.asarray(mat.get("lambdag_ns"))
    lambdag = np.asarray(mat.get("lambdag")).reshape(-1)
    k = int(np.asarray(mat.get("K")).reshape(-1)[0])
    lambdag_ds_std = np.asarray(mat.get("lambdag_ds_std"))
    lambdag_ss_std = np.asarray(mat.get("lambdag_ss_std"))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    bin_ctrs = np.linspace(-6, 6, 50)
    bin_width = bin_ctrs[1] - bin_ctrs[0]
    normal_x = np.linspace(-4, 4, 81)
    normal_pdf = np.exp(-0.5 * normal_x**2) / np.sqrt(2 * np.pi)

    titles = [(0, 0, "DS: Uselful"), (1, 0, "DS: Redundant"), (2, 0, "DS: Useless"), (0, 1, "SS: Uselful"), (1, 1, "SS: Redundant"), (2, 1, "SS: Useless")]
    series = [lambdag_ds_std[:, 0], lambdag_ds_std[:, 1], lambdag_ds_std[:, 2], lambdag_ss_std[:, 0], lambdag_ss_std[:, 1], lambdag_ss_std[:, 2]]

    for (row, col, title), values in zip(titles, series):
        counts, _ = np.histogram(values, bins=np.concatenate([bin_ctrs, [bin_ctrs[-1] + bin_width]]))
        prob = counts / (k * bin_width)
        ax = axes[row, col]
        ax.bar(bin_ctrs, prob, width=bin_width, color="0.6", edgecolor="0.6", align="center")
        ax.plot(normal_x, normal_pdf, color="k", linestyle="--", linewidth=1)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 0.5)
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "Figure11.jpg", format="jpg", dpi=300)
    plt.close(fig)

    print("MC bias")
    print(["useful", "redundant", "useless"])
    print(np.mean(lambdag_ds, axis=0) - lambdag)
    print(np.mean(lambdag_ss, axis=0) - lambdag)
    print(np.mean(lambdag_ns, axis=0) - lambdag)

    print("MC RMSE")
    print(["useful", "redundant", "useless"])
    print(np.sqrt(np.mean((lambdag_ds - np.ones((k, 1)) * lambdag) ** 2, axis=0)))
    print(np.sqrt(np.mean((lambdag_ss - np.ones((k, 1)) * lambdag) ** 2, axis=0)))
    print(np.sqrt(np.mean((lambdag_ns - np.ones((k, 1)) * lambdag) ** 2, axis=0)))
