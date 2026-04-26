from pathlib import Path

from module.workflows import run_simulation


def main() -> None:
    print("Starting simulation analysis...")
    run_simulation(Path(__file__).resolve().parent)
    print("Simulation analysis finished.")


if __name__ == "__main__":
    main()
