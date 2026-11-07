"""Verify pinned DVC data, schema, labels, and absence of train/eval overlap."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_utils import check_datasets, load_pipeline_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/data_validation.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    try:
        report = check_datasets(load_pipeline_config(args.overrides))
    except Exception as error:
        write_json(args.output, {"passed": False, "reasons": [str(error)]})
        raise
    write_json(args.output, report)
    print(f"DATA PASS: {report['n_training']} training rows, {report['n_evaluation']} evaluation rows, no overlap")


if __name__ == "__main__":
    main()
