from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Path to Julia executable.
# If you're using the NEMO-installed Julia, it's often something like:
# JULIA_EXE = r"C:\NemoMod\Julia\bin\julia.exe"
# Adjust this to whatever works on your system.
DEFAULT_JULIA_EXE = r"C:\NemoMod\Julia\bin\julia.exe"  # <-- change if needed

# Path to the Julia wrapper script in this folder
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_NEMO_SCRIPT = SCRIPT_DIR / "nemo_process.jl"
# Default LP dump target (mirrors the default in nemo_process.jl)
DEFAULT_LP_PATH = SCRIPT_DIR.parent / "intermediate_data" / "nemo_model_dump.lp"


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
    config_path: str | Path | None = None,
    varstosave: list[str] | None = None,
):
    """
    Call Julia + NEMO to solve the scenario in db_path.
    Set log_path to also write output to a file for debugging.
    If stream_output is True, stdout/stderr is streamed live to the console (and optionally tee'd to log_path).
    When config_path is provided, Julia runs from that file's parent directory so NEMO can auto-read nemo.cfg/nemo.ini.
    """
    db_path = Path(db_path)
    if not RUN_NEMO_SCRIPT.exists():
        raise FileNotFoundError(f"NEMO Julia script not found at '{RUN_NEMO_SCRIPT}'.")

    resolved_julia = _resolve_julia_exe(julia_exe)
    log_path = Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    lp_path = resolve_lp_dump_path()
    workdir = None
    if config_path:
        cfg = Path(config_path)
        workdir = cfg.parent
        if cfg.exists():
            print(f"  Config file: {cfg}")
        else:
            print(f"  Config file not found at '{cfg}' (will still run from its parent directory)")

    cmd = [
        str(resolved_julia),
        str(RUN_NEMO_SCRIPT),
        str(db_path),
    ]
    print("\nRunning NEMO via Julia:")
    print("  Command:", " ".join(cmd))
    print("  Input DB:", db_path)
    print("  Log file:", log_path if log_path else "<stdout only>")
    if os.environ.get("NEMO_WRITE_LP"):
        print(f"  LP dump: {lp_path} (from env NEMO_WRITE_LP)")
    else:
        print(f"  LP dump: {lp_path} (default if status != OPTIMAL or NEMO_WRITE_LP is set)")

    env = os.environ.copy()
    if varstosave is not None:
        cleaned = [str(v).strip() for v in varstosave if str(v).strip()]
        env["NEMO_VARSTOSAVE"] = ",".join(cleaned)

    stdout_text = ""
    stderr_text = ""
    if stream_output:
        # Stream combined stdout/stderr live; optionally tee to log_path.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=workdir,
            env=env,
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
            cwd=workdir,
            env=env,
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


def run_solver_test_script(
    script_path: Path,
    db_dir: Path,
    julia_exe: str | Path | None = None,
    log_path: str | Path | None = None,
    stream_output: bool = True,
):
    """
    Execute an upstream solver-specific NEMO test script (e.g., cbc_tests.jl) after
    ensuring it can find the bundled test DBs in db_dir.
    """
    script_path = Path(script_path)
    db_dir = Path(db_dir)
    if not script_path.exists():
        raise FileNotFoundError(f"NEMO solver test script not found at '{script_path}'.")
    if not db_dir.exists():
        raise FileNotFoundError(f"DB directory for solver tests not found at '{db_dir}'.")

    resolved_julia = _resolve_julia_exe(julia_exe)
    log_path = Path(log_path) if log_path else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Use POSIX-style paths in Julia strings to avoid Windows escape issues.
    db_dir_str = Path(db_dir).as_posix()
    script_path_str = Path(script_path).as_posix()

    harness = textwrap.dedent(
        f"""
        using NemoMod, SQLite, DataFrames, Test, JuMP
        const TOL = 0.5
        compilation = false
        reg_jumpmode = true
        calculatescenario_quiet = true
        dbfile_path = raw"{db_dir_str}"
        @info "Running solver test script {script_path.name} against DB dir {db_dir_str}"
        include(raw"{script_path_str}")
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False, encoding="utf-8") as tmp:
        tmp.write(harness)
        harness_path = Path(tmp.name)

    cmd = [
        str(resolved_julia),
        str(harness_path),
    ]
    print("\nRunning NEMO solver test via Julia:")
    print("  Command:", " ".join(cmd))

    stdout_text = ""
    stderr_text = ""
    proc = None
    try:
        if stream_output:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=db_dir,
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
            try:
                text = log_path.read_text() if log_path else stdout_text
                error_lines = [line for line in text.splitlines() if "error" in line.lower()]
                if error_lines:
                    print("  Errors found in output:")
                    for line in error_lines[:10]:
                        print("   ", line)
            except Exception:
                pass
        else:
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=bool(log_path),
                cwd=db_dir,
            )
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            if log_path:
                log_path.write_text(stdout_text + "\n--- STDERR ---\n" + stderr_text)
                print(f"  Julia output written to '{log_path}'")
                try:
                    text = log_path.read_text()
                    error_lines = [line for line in text.splitlines() if "error" in line.lower()]
                    if error_lines:
                        print("  Errors found in log:")
                        for line in error_lines[:10]:
                            print("   ", line)
                except Exception:
                    pass
    finally:
        try:
            harness_path.unlink()
        except Exception:
            pass

    if proc is None:
        raise RuntimeError("Failed to start Julia process for solver test.")
    if proc.returncode != 0:
        raise RuntimeError(
            f"NEMO solver test '{script_path.name}' failed with exit code {proc.returncode}. "
            f"See {log_path if log_path else 'stdout/stderr above'} for details."
        )
    print("Solver test run completed.")


def resolve_lp_dump_path() -> Path:
    """
    Mirror the Julia-side LP resolution:
    - Use env NEMO_WRITE_LP when set
    - Otherwise default to ./intermediate_data/nemo_model_dump.lp
    """
    env_lp = os.environ.get("NEMO_WRITE_LP")
    if env_lp:
        return Path(env_lp)
    return DEFAULT_LP_PATH
