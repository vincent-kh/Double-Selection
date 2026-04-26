import argparse
from pathlib import Path

from module.workflows import run_robustness, run_stepwise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness analysis.")
    parser.add_argument(
        "-n", "--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all CPUs)"
    )
    args = parser.parse_args()

    print(f"Starting robustness analysis (n_jobs={args.n_jobs})...")
    run_robustness(n_jobs=args.n_jobs)
    print("Robustness analysis finished. Results: csv/robustness.csv")

    print(f"Starting stepwise analysis (n_jobs={args.n_jobs})...")
    run_stepwise(n_jobs=args.n_jobs)
    print("Stepwise analysis finished. Results: csv/robustness_stepwise.csv")


if __name__ == "__main__":
    main()
