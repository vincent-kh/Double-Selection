import argparse
import sys
import time
import warnings
from pathlib import Path

from module.workflows import run_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main analysis.")
    parser.add_argument(
        "-n", "--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all CPUs)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress runtime warnings."
    )
    args = parser.parse_args()

    if args.quiet:
        # Ignore specific runtime warnings often caused by empty slices or NaN operations
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        print("Warnings suppressed.")

    print(f"Starting main analysis (n_jobs={args.n_jobs})...")
    st = time.time()
    run_main(n_jobs=args.n_jobs)
    ed = time.time()
    print(f"Main analysis completed in {ed - st:.2f} seconds.")
    print("Main analysis finished.")
    print("Results saved to: csv/main.csv")


if __name__ == "__main__":
    main()
