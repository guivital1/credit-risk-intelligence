from __future__ import annotations

import argparse
import json

from credit_risk.config import DASHBOARD_DATA_PATH, METRICS_PATH, RAW_DATA_PATH
from credit_risk.dashboard import generate_dashboard_data
from credit_risk.data import fetch_dataset, load_dataset
from credit_risk.modeling import train_and_evaluate


def command_fetch() -> None:
    frame = fetch_dataset()
    print(f"Fetched {len(frame):,} rows to {RAW_DATA_PATH}")
    print(f"Default rate: {frame['default_next_month'].mean():.2%}")


def command_train() -> None:
    result = train_and_evaluate(load_dataset())
    print(f"Selected model: {result.selected_model}")
    print(f"Test rows: {result.test_rows:,}")
    print(json.dumps(result.metrics[result.selected_model], indent=2))
    print(f"Metrics saved to {METRICS_PATH}")


def command_dashboard() -> None:
    payload = generate_dashboard_data()
    print(f"Dashboard snapshot saved to {DASHBOARD_DATA_PATH}")
    print(f"Scored rows: {payload['summary']['scored_rows']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit Risk Intelligence pipeline")
    parser.add_argument("command", choices=("fetch", "train", "dashboard", "all"))
    args = parser.parse_args()

    if args.command in {"fetch", "all"}:
        command_fetch()
    if args.command in {"train", "all"}:
        command_train()
    if args.command in {"dashboard", "all"}:
        command_dashboard()


if __name__ == "__main__":
    main()
