from __future__ import annotations

import argparse
import json

from credit_risk.aws_pipeline import cleanup, deploy_and_run, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled SageMaker workflow")
    parser.add_argument("command", choices=("plan", "deploy", "cleanup"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for commands that create or remove AWS resources",
    )
    args = parser.parse_args()

    if args.command == "plan":
        result = plan()
    elif not args.execute:
        parser.error(f"{args.command} requires --execute")
    elif args.command == "deploy":
        result = deploy_and_run()
    else:
        result = cleanup()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

