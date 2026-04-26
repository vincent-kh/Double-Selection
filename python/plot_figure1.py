#!/usr/bin/env python3
"""
Replicate Figure 1 from the paper using Python.
Computes selection rates across 200 random seeds/tuning parameters
and plots them with factor labels.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from module.workflows import run_figure1

def main():
    print("Starting Figure 1 replication...")
    
    # Compute selection rates (parallelized)
    # n_jobs=-1 uses all CPUs
    rates = run_figure1(n_jobs=-1)
    
    # Factor IDs (1-indexed for the plot to match MATLAB)
    factor_ids = np.arange(1, len(rates) + 1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar plot (gray color as in MATLAB)
    bars = ax.bar(factor_ids, rates, width=0.4, color='0.6', edgecolor='0.6')
    
    ax.set_xlim(0, 136)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('Factor ID', fontsize=12)
    ax.set_ylabel('Selection rate', fontsize=12)
    ax.tick_params(labelsize=10)
    
    # Key factor labels from Figure1_add.jpg
    # Mapped IDs (1-indexed):
    # MktRf: 1
    # Earnings to price: 3
    # SMB: 21
    # Number of earnings increases: 45
    # Financial statements score: 47 (ps)
    # Industry-adjusted book to market: 48
    # Volatility of liquidity (dollar volume): 53
    # Change in Net Non-current Operating Assets: 84
    # Financial statements score (Mohanram): 90
    # Net external finance: 99
    # Change in shares outstanding: 109
    # Profit margin: 117
    # IA change in profit margin: 120
    
    labels = [
        (1, "MktRf", (5, 0.15)),
        (3, "Earnings\nto price", (10, 0.55)),
        (21, "SMB", (30, 0.7)),
        (45, "Number of\nearnings\nincreases", (35, 0.3)),
        (47, "Financial\nstatements\nscore", (35, 0.55)),
        (48, "Industry-adjusted\nbook to market", (50, 0.65)),
        (53, "Volatility\nof liquidity", (70, 0.55)),
        (84, "Change in Net\nNon-current\nOperating Assets", (110, 0.45)),
        (99, "Net external\nfinance", (115, 0.7)),
        (109, "Change in\nShares\nOutstanding", (105, 0.2)),
        (117, "Profit Margin", (125, 0.35)),
    ]
    
    for fid, text, text_pos in labels:
        # Draw arrow
        ax.annotate(
            text,
            xy=(fid, rates[fid-1]), 
            xytext=text_pos,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10,
            color='blue',
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='red', lw=1),
            ha='center'
        )

    plt.tight_layout()
    output_path = project_root / "Figure1_python.png"
    plt.savefig(output_path, dpi=300)
    print(f"Figure 1 saved to: {output_path}")
    
    # Also save the rates as CSV for reference
    rates_df = pd.DataFrame({
        'FactorID': factor_ids,
        'SelectionRate': rates
    })
    csv_path = project_root / "csv" / "figure1_rates.csv"
    rates_df.to_csv(csv_path, index=False)
    print(f"Selection rates saved to: {csv_path}")

if __name__ == "__main__":
    main()
