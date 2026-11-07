"""Render report evidence into the GitHub Actions job summary (stdlib only)."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    reports = {}
    for name in ("data_validation", "training_manifest", "validation", "registration"):
        path = directory / f"{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## MLOps pipeline evidence", "", "| Stage | Result |", "|---|---|"]
    for name in ("data_validation", "training_manifest", "validation", "registration"):
        report = reports.get(name)
        result = "NOT RUN"
        if report:
            result = "PASS" if report.get("passed", bool(report.get("run_id"))) else "FAIL"
        lines.append(f"| {name} | {result} |")
    candidate = reports.get("training_manifest", {})
    for key in ("git_commit", "run_id", "model_uri"):
        if key in candidate:
            lines.extend(["", f"**{key}:** `{candidate[key]}`"])
    validation = reports.get("validation", {})
    if "metrics" in validation:
        lines.extend(["", "PR-AUC definition: Average Precision.", "",
                      "| Metric | Measured | Minimum |", "|---|---:|---:|"])
        for name, threshold in validation["thresholds"].items():
            lines.append(f"| {name} | {validation['metrics'][name]:.6f} | {threshold:.6f} |")
    registry = reports.get("registration", {})
    if registry.get("passed"):
        lines.extend(["", f"**Registry:** `{registry['model_name']}` v{registry['version']} → {registry['stage']} (@{registry['alias']})"])
    for name, report in reports.items():
        for reason in report.get("reasons", []):
            lines.extend(["", f"{name}: {reason}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True)
    print(summarize(parser.parse_args().reports), end="")
