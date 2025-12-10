"""
Lightweight parser for NEMO run logs.

Usage:
  python analyze_log.py --log data/nemo_run.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


KEY_PATTERNS = {
    "task_failed": re.compile(r"TaskFailedException", re.IGNORECASE),
    "key_error": re.compile(r"KeyError", re.IGNORECASE),
    "infeasible": re.compile(r"infeasible", re.IGNORECASE),
    "unbounded": re.compile(r"unbounded", re.IGNORECASE),
    "dual_infeasible": re.compile(r"DualInfeasible|Dual infeasible", re.IGNORECASE),
    "primal_inf": re.compile(r"Primal inf", re.IGNORECASE),
    "solver_status": re.compile(r"NEMO termination status:\s*(.+)", re.IGNORECASE),
}


def analyze_log(log_path: Path):
    text = log_path.read_text(errors="ignore")
    lines = text.splitlines()

    findings: list[str] = []

    status_matches = KEY_PATTERNS["solver_status"].findall(text)
    if status_matches:
        findings.append(f"Solver status: {status_matches[-1].strip()}")

    for name, pattern in KEY_PATTERNS.items():
        if name == "solver_status":
            continue
        matches = pattern.findall(text)
        if matches:
            findings.append(f"Found {len(matches)} occurrences of '{name}'.")

    if not findings:
        findings.append("No obvious errors found.")

    tail = "\n".join(lines[-30:]) if lines else ""

    return findings, tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to nemo_run.log")
    args = parser.parse_args()
    log_path = Path(args.log)

    findings, tail = analyze_log(log_path)
    print("Findings:")
    for f in findings:
        print("  -", f)
    print("\nTail of log (last 30 lines):")
    print(tail)


if __name__ == "__main__":
    main()
