import os
import json
import argparse
from pathlib import Path

from backend.benchmark_engine import run_model_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run GridGuard AI benchmark")
    parser.add_argument("--mode", type=str, default="kaggle_historical", 
                        help="Data mode (e.g. kaggle_historical, synthetic)")
    args = parser.parse_args()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    json_path = artifacts_dir / "gridguard_benchmark.json"
    csv_path = artifacts_dir / "gridguard_benchmark.csv"

    payload, predictions = run_model_benchmark(data_mode=args.mode)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved benchmark metadata and metrics to {json_path}")

    # Save CSV
    predictions.to_csv(csv_path, index=False)
    print(f"Saved holdout predictions to {csv_path}")

    print("\nBenchmark completed successfully.")
    print("-" * 40)
    print("Portfolio Summary:")
    print(json.dumps(payload["portfolio_summary"], indent=2))
    print("-" * 40)

if __name__ == "__main__":
    main()
