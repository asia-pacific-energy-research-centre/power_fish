from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Path to Julia executable.
# If you're using the NEMO-installed Julia, it's often something like:
# JULIA_EXE = r"C:\NemoMod\Julia\bin\julia.exe"
# Adjust this to whatever works on your system.
DEFAULT_JULIA_EXE = r"C:\NemoMod\Julia\bin\julia.exe"  # <-- change if needed

# Path to the Julia wrapper script in this folder
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_NEMO_SCRIPT = SCRIPT_DIR / "nemo_process.jl"


def _resolve_julia_exe(julia_exe: str | Path | None = None) -> Path:
    """
    Resolve a usable Julia executable from:
    - explicit argument
    - env JULIA_EXE or NEMO_JULIA_EXE
    - DEFAULT_JULIA_EXE constant
    - julia found on PATH
    """
    candidates: list[Path] = []
    if julia_exe:
        candidates.append(Path(julia_exe))
    env_julia = os.environ.get("JULIA_EXE") or os.environ.get("NEMO_JULIA_EXE")
    if env_julia:
        candidates.append(Path(env_julia))
    candidates.append(Path(DEFAULT_JULIA_EXE))
    which_julia = shutil.which("julia")
    if which_julia:
        candidates.append(Path(which_julia))

    for cand in candidates:
        if cand and cand.exists():
            return cand

    raise FileNotFoundError(
        "Julia executable not found. Set env JULIA_EXE (or NEMO_JULIA_EXE) "
        "or update DEFAULT_JULIA_EXE in run_nemo_via_julia.py."
    )


def run_nemo_on_db(
    db_path: Path,
    julia_exe: str | Path | None = None,
    log_path: str | Path | None = None,
):
    """
    Call Julia + NEMO to solve the scenario in db_path.
    Set log_path to capture stdout/stderr into a file for debugging.
    """
    db_path = Path(db_path)
    if not RUN_NEMO_SCRIPT.exists():
        raise FileNotFoundError(f"NEMO Julia script not found at '{RUN_NEMO_SCRIPT}'.")

    resolved_julia = _resolve_julia_exe(julia_exe)
    log_path = Path(log_path) if log_path else None

    cmd = [
        str(resolved_julia),
        str(RUN_NEMO_SCRIPT),
        str(db_path),
    ]
    print("\nRunning NEMO via Julia:")
    print("  Command:", " ".join(cmd))

    # If log_path is given, capture output and write it; otherwise stream live.
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=bool(log_path),
    )

    if log_path:
        log_path.write_text(
            (proc.stdout or "")
            + "\n--- STDERR ---\n"
            + (proc.stderr or "")
        )
        print(f"  Julia output written to '{log_path}'")
        # Surface any lines containing 'error' to help debugging.
        try:
            text = log_path.read_text()
            error_lines = [
                line for line in text.splitlines() if "error" in line.lower()
            ]
            if error_lines:
                print("  Errors found in log:")
                for line in error_lines[:10]:
                    print("   ", line)
        except Exception:
            pass
    else:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"NEMO run failed with exit code {proc.returncode}. "
            f"See {log_path if log_path else 'stdout/stderr above'} for details."
        )
    print("NEMO run completed.")
