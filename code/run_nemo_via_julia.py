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


def create_template_db(template_path: Path, julia_exe: str | Path | None = None) -> Path:
    """
    Create a blank NEMO template database via NemoMod.createnemodb if it is missing.
    """
    template_path = Path(template_path)
    if template_path.exists():
        print(f"Template DB already exists at '{template_path}'. Skipping creation.")
        return template_path

    template_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_julia = _resolve_julia_exe(julia_exe)

    db_dir = template_path.parent.as_posix()
    db_name = template_path.name
    julia_cmd = [
        str(resolved_julia),
        "-e",
        (
            "using NemoMod; "
            f'cd(raw"{db_dir}"); '
            f'NemoMod.createnemodb(raw"{db_name}")'
        ),
    ]

    print(f"Template DB not found. Creating via Julia at '{template_path}'.")
    proc = subprocess.run(julia_cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to create template DB at '{template_path}'. "
            "Ensure NemoMod.jl is installed and accessible to Julia."
        )

    if not template_path.exists():
        raise RuntimeError(
            f"Julia reported success but '{template_path}' was not created."
        )

    print(f"Created template DB at '{template_path}'.")
    return template_path


def run_nemo_on_db(
    db_path: Path,
    julia_exe: str | Path | None = None,
    log_path: str | Path | None = None,
    stream_output: bool = True,
):
    """
    Call Julia + NEMO to solve the scenario in db_path.
    Set log_path to also write output to a file for debugging.
    If stream_output is True, stdout/stderr is streamed live to the console (and optionally tee'd to log_path).
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

    stdout_text = ""
    stderr_text = ""
    if stream_output:
        # Stream combined stdout/stderr live; optionally tee to log_path.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fh = log_path.open("w", encoding="utf-8") if log_path else None
        collected: list[str] = []
        try:
            assert proc.stdout is not None  # for type checkers
            for line in proc.stdout:
                print(line, end="")
                collected.append(line)
                if log_fh:
                    log_fh.write(line)
        finally:
            if log_fh:
                log_fh.flush()
                log_fh.close()
        proc.wait()
        stdout_text = "".join(collected)
        print()  # newline after stream
        if log_path:
            print(f"  Julia output streamed and written to '{log_path}'")
        # Surface any lines containing 'error' to help debugging.
        try:
            text = log_path.read_text() if log_path else stdout_text
            error_lines = [
                line for line in text.splitlines() if "error" in line.lower()
            ]
            if error_lines:
                print("  Errors found in output:")
                for line in error_lines[:10]:
                    print("   ", line)
        except Exception:
            pass
    else:
        # If not streaming, fall back to buffered capture (old behavior).
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=bool(log_path),
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        if log_path:
            log_path.write_text(
                stdout_text
                + "\n--- STDERR ---\n"
                + stderr_text
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

    if proc.returncode != 0:
        breakpoint()
        raise RuntimeError(
            f"NEMO run failed with exit code {proc.returncode}. "
            f"See {log_path if log_path else 'stdout/stderr above'} for details."
        )
    print("NEMO run completed.")
