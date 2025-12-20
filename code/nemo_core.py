from __future__ import annotations

"""
Core utilities for the NEMO pipeline: template DB helpers, NEMO test flow,
log analysis, diagnostics, dummy workbook generation, and shared defaults.
"""

from pathlib import Path
import sqlite3
import shutil
import re
import sys
import os
import urllib.request
from typing import Iterable, Sequence

import pandas as pd

from convert_osemosys_input_to_nemo import (
    dump_db_to_entry_excel,
    PARAM_SPECS,
    convert_osemosys_input_to_nemo,
)
from run_nemo_via_julia import create_template_db, run_nemo_on_db, run_solver_test_script


# ---------------------------------------------------------------------------
# Template DB helpers
# ---------------------------------------------------------------------------
def ensure_template_db(template_path: Path, auto_create: bool, julia_exe: str | Path | None):
    """Make sure the NEMO template DB exists; optionally create it with Julia if missing."""
    template_path = Path(template_path)
    if template_path.exists():
        return
    if not auto_create:
        raise FileNotFoundError(
            f"Template DB '{template_path}' not found. "
            "Set AUTO_CREATE_TEMPLATE_DB=True to build it automatically."
        )
    create_template_db(template_path, julia_exe=julia_exe)


def trim_db_years_in_place(db_path: Path, years: list[int]):
    """Remove rows from all tables that have a 'y' column for years not in the list."""
    years = sorted({int(y) for y in years})
    if not years:
        return
    years_param = ",".join("?" * len(years))
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for tbl in tables:
            info = cur.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            cols = [c[1] for c in info]
            if "y" not in cols:
                continue
            cur.execute(
                f'DELETE FROM "{tbl}" WHERE y NOT IN ({years_param})',
                tuple(years),
            )
        if "YEAR" in tables:
            cur.execute(f'DELETE FROM "YEAR" WHERE val NOT IN ({years_param})', tuple(years))
        conn.commit()


# ---------------------------------------------------------------------------
# Storage / NEMO test DB flow
# ---------------------------------------------------------------------------
# Raw GitHub URL pointing at the upstream NEMO test folder
NEMO_TEST_BASE_URL = "https://raw.githubusercontent.com/sei-international/NemoMod.jl/master/test"
NEMO_TEST_DB_MAP = {
    "storage_test": "storage_test.sqlite",
    "storage_transmission_test": "storage_transmission_test.sqlite",
    "ramp_test": "ramp_test.sqlite",
}
# Solver-specific test scripts that exist upstream; used for convenience downloads
NEMO_SOLVER_TEST_SCRIPTS = {
    "cbc": "cbc_tests.jl",
    "glpk": "glpk_tests.jl",
    "gurobi": "gurobi_tests.jl",
    "cplex": "cplex_tests.jl",
    "mosek": "mosek_tests.jl",
    "xpress": "xpress_tests.jl",
    "highs": "highs_tests.jl",
}


def detect_solver_preference(vars_cfg: dict) -> str:
    """Best-effort detection of which solver the user asked NEMO to use (env or config)."""
    return str(vars_cfg.get("NEMO_SOLVER") or os.environ.get("NEMO_SOLVER") or "cbc").lower()


def resolve_solver_from_test_name(test_name: str | None) -> str | None:
    """
    Map a NEMO_TEST_NAME like 'cbc_tests' (or 'cbc') to the corresponding solver key.
    Returns None when the name is not one of the solver-specific upstream test scripts.
    """
    if not test_name:
        return None
    normalized = Path(test_name).stem.lower()
    for solver, file_name in NEMO_SOLVER_TEST_SCRIPTS.items():
        if normalized in {solver, Path(file_name).stem.lower()}:
            return solver
    return None


def _download_to_path(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest_path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_nemo_test_db(test_name: str, dest_dir: Path) -> Path:
    """
    Ensure the requested NEMO test DB exists locally; download from upstream if missing.
    """
    dest_dir = Path(dest_dir)
    file_name = NEMO_TEST_DB_MAP.get(test_name, f"{test_name}.sqlite")
    dest_path = dest_dir / file_name
    if dest_path.exists():
        return dest_path

    url = f"{NEMO_TEST_BASE_URL}/{file_name}"
    print(f"Downloading NEMO test DB '{test_name}' to '{dest_path}'")
    try:
        _download_to_path(url, dest_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download NEMO test DB '{test_name}' from {url}: {exc}") from exc
    return dest_path


def ensure_solver_test_script(solver: str, dest_dir: Path, *, strict: bool = False) -> Path | None:
    """
    Optionally fetch the upstream solver-specific NEMO test script (e.g., cbc_tests.jl) for reference.
    When strict=True, failures raise instead of printing a warning.
    """
    solver_key = solver.lower()
    file_name = NEMO_SOLVER_TEST_SCRIPTS.get(solver_key)
    if not file_name:
        for cand_solver, cand_file in NEMO_SOLVER_TEST_SCRIPTS.items():
            cand_stem = Path(cand_file).stem.lower()
            if solver_key in {cand_file.lower(), cand_stem}:
                solver_key = cand_solver
                file_name = cand_file
                break
    if not file_name:
        if strict:
            raise ValueError(f"Unknown solver test '{solver}'.")
        return None
    dest_path = Path(dest_dir) / file_name
    if dest_path.exists():
        return dest_path
    url = f"{NEMO_TEST_BASE_URL}/{file_name}"
    print(f"Downloading NEMO {solver_key} test script to '{dest_path}'")
    try:
        _download_to_path(url, dest_path)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"Failed to download NEMO {solver_key} test script from {url}: {exc}") from exc
        print(f"Warning: failed to download {file_name} from {url}: {exc}")
        return None
    return dest_path


def handle_test_run(vars_cfg: dict, data_dir: Path, log_dir: Path, run_nemo: bool) -> bool:
    """
    Handle the NEMO test shortcut (skip main conversion) if enabled.
    Supports:
      - Built-in upstream NEMO test DB names (auto-download)
      - Local sqlite DB path
      - Local Excel path (nemo_entry or osemosys) converted to DB on the fly
    Returns True if the flow was handled and the caller should exit early.
    """
    nemo_test_name = vars_cfg.get("NEMO_TEST_NAME")

    def maybe_handle_solver_specific_test(dest_dir: Path, test_name: str | None = None) -> bool:
        """Run solver-specific upstream tests (e.g., cbc_tests.jl) when requested."""
        solver_key = resolve_solver_from_test_name(test_name or nemo_test_name)
        if not solver_key:
            return False
        dest_dir = Path(dest_dir)
        script_path = ensure_solver_test_script(solver_key, dest_dir, strict=True)
        # Solver test scripts expect the standard trio of upstream DBs to be present alongside them.
        for base_test in NEMO_TEST_DB_MAP:
            ensure_nemo_test_db(base_test, dest_dir)
        if run_nemo:
            run_solver_test_script(
                script_path,
                db_dir=dest_dir,
                julia_exe=vars_cfg.get("JULIA_EXE"),
                log_path=log_dir / f"{solver_key}_tests.log",
                stream_output=True,
            )
        else:
            print(
                f"Solver-specific NEMO test script downloaded to '{script_path}', "
                "but run_nemo=False so execution was skipped."
            )
        return True

    test_input = (
        vars_cfg.get("TEST_INPUT_PATH")
        or vars_cfg.get("NEMO_TEST_DB_PATH")
        or vars_cfg.get("TEST_DB_PATH")
        or vars_cfg.get("STORAGE_TEST_DB")  # backward compat
    )
    source_kind = "sqlite"
    excel_source: Path | None = None
    db_path: Path | None = None
    if test_input:
        test_input_path = Path(test_input)
        if not test_input_path.exists():
            dest_dir = (
                test_input_path.parent
                if test_input_path.parent != Path("")
                else Path(vars_cfg.get("NEMO_TEST_DB_DIR", data_dir))
            )
            if maybe_handle_solver_specific_test(dest_dir, nemo_test_name):
                return True
            if nemo_test_name:
                db_path = ensure_nemo_test_db(nemo_test_name, dest_dir)
                test_input_path = db_path
            else:
                raise FileNotFoundError(f"Test input not found at '{test_input_path}'")
        ext = test_input_path.suffix.lower()
        if ext in {".xlsx", ".xlsm", ".xls"}:
            source_kind = "excel"
            excel_source = test_input_path
        else:
            db_path = test_input_path
    else:
        test_name = (
            nemo_test_name
            or ("storage_test" if vars_cfg.get("USE_STORAGE_TEST_DB") else None)
            or "storage_test"
        )
        if maybe_handle_solver_specific_test(Path(vars_cfg.get("NEMO_TEST_DB_DIR", data_dir)), test_name):
            return True
        db_dir = Path(vars_cfg.get("NEMO_TEST_DB_DIR", data_dir))
        db_path = ensure_nemo_test_db(test_name, db_dir)

    if source_kind == "excel" and excel_source:
        output_db = Path(vars_cfg.get("TEST_OUTPUT_DB") or data_dir / "nemo_test.sqlite")
        input_mode = str(vars_cfg.get("TEST_INPUT_MODE") or "nemo_entry").lower()
        # Ensure template exists if conversion is needed
        ensure_template_db(
            vars_cfg["TEMPLATE_DB"],
            auto_create=bool(vars_cfg.get("AUTO_CREATE_TEMPLATE_DB", False)),
            julia_exe=vars_cfg.get("JULIA_EXE"),
        )
        convert_cfg = dict(vars_cfg)
        convert_cfg.update(
            {
                "INPUT_MODE": input_mode,
                "OSEMOSYS_EXCEL_PATH": excel_source,
                "NEMO_ENTRY_EXCEL_PATH": excel_source,
                "OUTPUT_DB": output_db,
            }
        )
        convert_osemosys_input_to_nemo(convert_cfg)
        db_path = output_db
    elif source_kind == "sqlite":
        # db_path was set either from explicit path or download
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Test DB not found at '{db_path}'")
    else:
        raise ValueError(f"Unsupported test source type: {source_kind}")

    solver = detect_solver_preference(vars_cfg)
    solver_script = ensure_solver_test_script(solver, db_path.parent)
    if solver_script:
        print(f"Detected solver '{solver}'. Solver-specific test script available at '{solver_script}'.")
    else:
        print(f"Detected solver '{solver}'. No solver-specific test script downloaded.")

    if vars_cfg.get("EXPORT_DB_TO_EXCEL"):
        dump_db_to_entry_excel(
            db_path=db_path,
            excel_path=Path(
                vars_cfg.get("TEST_EXPORT_DB_TO_EXCEL_PATH")
                or vars_cfg.get("TEST_EXPORT_EXCEL_PATH")
                or vars_cfg.get("NEMO_TEST_EXCEL_PATH")
                or vars_cfg.get("STORAGE_TEST_EXCEL_PATH")  # backward compat
                or data_dir / "test_db_dump.xlsx"
            ),
            specs=PARAM_SPECS,
            tables=None,
        )

    if run_nemo:
        if vars_cfg.get("NEMO_WRITE_LP"):
            os.environ["NEMO_WRITE_LP"] = str(vars_cfg["NEMO_WRITE_LP"])
        run_nemo_on_db(
            db_path,
            julia_exe=vars_cfg.get("JULIA_EXE"),
            log_path=log_dir / "nemo_run.log",
            stream_output=True,
            config_path=vars_cfg.get("NEMO_CONFIG_PATH"),
        )
    return True


# Backward compatibility alias
def handle_storage_test(vars_cfg: dict, data_dir: Path, log_dir: Path, run_nemo: bool) -> bool:
    return handle_test_run(vars_cfg, data_dir, log_dir, run_nemo)


# ---------------------------------------------------------------------------
# Main flow preparation
# ---------------------------------------------------------------------------
def prepare_run_context(vars_cfg: dict, data_dir: Path, mode: str, log_dir: Path) -> dict:
    """
    Ensure template DB exists and configure mode-specific paths/inputs.
    Returns an updated vars dict.
    """
    ensure_template_db(
        vars_cfg["TEMPLATE_DB"],
        auto_create=bool(vars_cfg.get("AUTO_CREATE_TEMPLATE_DB", False)),
        julia_exe=vars_cfg.get("JULIA_EXE"),
    )

    updated = dict(vars_cfg)
    if mode == "dummy":
        updated["INPUT_MODE"] = "osemosys"
        updated["OSEMOSYS_EXCEL_PATH"] = data_dir / "dummy_osemosys.xlsx"
        updated["NEMO_ENTRY_EXCEL_PATH"] = data_dir / "dummy_nemo.xlsx"
        updated["EXPORT_DB_TO_EXCEL_PATH"] = data_dir / "dummy_nemo.xlsx"
        updated["OUTPUT_DB"] = data_dir / "dummy_nemo.sqlite"
        make_dummy_workbook(updated["OSEMOSYS_EXCEL_PATH"])
    elif mode == "nemo_input":
        updated["INPUT_MODE"] = "nemo_entry"
    elif mode == "osemosys_input":
        updated["INPUT_MODE"] = "osemosys"
    else:
        raise ValueError(f"Unknown MODE '{mode}'")

    log_dir.mkdir(parents=True, exist_ok=True)
    return updated


# ---------------------------------------------------------------------------
# Log analysis
# ---------------------------------------------------------------------------
KEY_PATTERNS = {
    "task_failed": re.compile(r"TaskFailedException", re.IGNORECASE),
    "key_error": re.compile(r"KeyError", re.IGNORECASE),
    "infeasible": re.compile(r"infeasible", re.IGNORECASE),
    "unbounded": re.compile(r"unbounded", re.IGNORECASE),
    "dual_infeasible": re.compile(r"DualInfeasible|Dual infeasible", re.IGNORECASE),
    "primal_inf": re.compile(r"Primal inf", re.IGNORECASE),
    "solver_status": re.compile(r"NEMO termination status:\\s*(.+)", re.IGNORECASE),
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def parse_years(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def load_df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    except Exception:
        return pd.DataFrame()


def trim_years(conn: sqlite3.Connection, years: Sequence[int], tables: Iterable[str]):
    years_set = set(int(y) for y in years)
    cur = conn.cursor()
    for table in tables:
        try:
            info = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
        except Exception:
            continue
        cols = [c[1] for c in info]
        if "y" not in cols:
            continue
        cur.execute(f'DELETE FROM "{table}" WHERE y NOT IN ({",".join("?"*len(years_set))})', tuple(years_set))
    # YEAR set table
    cur.execute('DELETE FROM "YEAR" WHERE val NOT IN ({})'.format(",".join("?"*len(years_set))), tuple(years_set))
    conn.commit()


def _print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def run_diagnostics(db_path: str | Path, years: list[int] | None = None, write_trimmed: str | Path | None = None):
    db_path = Path(db_path)
    work_db = db_path
    if write_trimmed and years:
        work_db = Path(write_trimmed)
        shutil.copyfile(db_path, work_db)
        with sqlite3.connect(work_db) as conn:
            trim_years(conn, years, tables=[])
        print(f"Wrote trimmed DB to {work_db}")

    with sqlite3.connect(work_db) as conn:
        # Sets
        sets = {}
        for tbl in ["REGION", "TECHNOLOGY", "FUEL", "TIMESLICE", "EMISSION", "YEAR", "MODE_OF_OPERATION"]:
            df_set = load_df(conn, tbl)
            sets[tbl] = set(df_set["val"].astype(str)) if not df_set.empty else set()

        _print_section("Years and demand coverage")
        sad = load_df(conn, "SpecifiedAnnualDemand")
        if sad.empty:
            print("SpecifiedAnnualDemand: empty")
        else:
            if years:
                sad = sad[sad["y"].astype(int).isin(years)]
            print(f"Demand rows: {len(sad)}; regions: {sorted(sad['r'].unique())}; fuels: {sorted(sad['f'].unique())}; years: {sorted(sad['y'].unique())[:10]}")

        _print_section("Supply tech presence")
        af = load_df(conn, "AvailabilityFactor")
        if af.empty:
            print("AvailabilityFactor: empty")
        else:
            if years:
                af = af[af["y"].astype(int).isin(years)]
            print(f"Techs with availability factors: {len(sorted(af['t'].unique()))}; sample: {sorted(af['t'].unique())[:5]}")

        _print_section("Input/Output activity ratios")
        iar = load_df(conn, "InputActivityRatio")
        oar = load_df(conn, "OutputActivityRatio")
        if years:
            if not iar.empty and "y" in iar.columns:
                iar = iar[iar["y"].astype(int).isin(years)]
            if not oar.empty and "y" in oar.columns:
                oar = oar[oar["y"].astype(int).isin(years)]
        if iar.empty:
            print("InputActivityRatio: empty (no fuel inputs defined!)")
        else:
            modes = sorted(set(str(m) for m in iar["m"].dropna().unique()))
            fuels = sorted(set(str(f) for f in iar["f"].dropna().unique()))
            print(f"IAR rows: {len(iar)}; techs: {len(set(iar['t']))}; fuels: {fuels[:5]}; modes: {modes}")
        if oar.empty:
            print("OutputActivityRatio: empty (no fuel outputs defined!)")
        else:
            modes = sorted(set(str(m) for m in oar["m"].dropna().unique()))
            fuels = sorted(set(str(f) for f in oar["f"].dropna().unique()))
            print(f"OAR rows: {len(oar)}; techs: {len(set(oar['t']))}; fuels: {fuels[:5]}; modes: {modes}")

        _print_section("Demand fuels coverage by output ratios")
        if not sad.empty and not oar.empty:
            demand_fuels = set(sad["f"])
            oar_fuels = set(oar["f"])
            missing_fuels = sorted(demand_fuels - oar_fuels)
            if missing_fuels:
                print(f"Demand fuels with no OutputActivityRatio: {missing_fuels[:10]}")
            else:
                print("All demand fuels appear in OutputActivityRatio.")

        _print_section("Reserve margin status")
        rm = load_df(conn, "ReserveMargin")
        if rm.empty:
            print("ReserveMargin: empty (or disabled).")
        else:
            if years:
                rm = rm[rm["y"].astype(int).isin(years)]
            fuels = sorted(set(str(f) for f in rm["f"].unique()))
            print(f"ReserveMargin rows: {len(rm)}; fuels tags: {fuels[:10]}")

        _print_section("Emission limits and ratios")
        ael = load_df(conn, "AnnualEmissionLimit")
        if ael.empty:
            print("AnnualEmissionLimit: empty (or disabled).")
        else:
            if years:
                ael = ael[ael["y"].astype(int).isin(years)]
            print(f"AnnualEmissionLimit rows: {len(ael)}; emissions: {sorted(ael['e'].unique())[:5]}")
        ear = load_df(conn, "EmissionActivityRatio")
        if ear.empty:
            print("EmissionActivityRatio: empty!")
        else:
            modes = sorted(set(str(m) for m in ear["m"].dropna().unique()))
            print(f"EmissionActivityRatio rows: {len(ear)}; modes present: {modes}")

        _print_section("Basic conflict checks")
        # Demand years vs availability years
        if not sad.empty and not af.empty:
            demand_years = set(int(y) for y in sad["y"].unique())
            avail_years = set(int(y) for y in af["y"].unique())
            missing_years = sorted(demand_years - avail_years)
            if missing_years:
                print(f"No availability factors for demand years: {missing_years[:10]}")
            else:
                print("Availability factors cover all demand years.")

        # Demand fuels vs output fuels
        if not sad.empty and not oar.empty:
            demand_fuels = set(sad["f"])
            oar_fuels = set(oar["f"])
            missing_fuels = sorted(demand_fuels - oar_fuels)
            if missing_fuels:
                print(f"WARNING: Demand fuels missing in OutputActivityRatio: {missing_fuels[:10]}")

        # Emission coverage: if limits exist but no emission ratios
        if not ael.empty and ear.empty:
            print("WARNING: Emission limits present but no EmissionActivityRatio rows.")

        # Modes coverage
        if not ear.empty:
            modes_set = sets.get("MODE_OF_OPERATION", set())
            if modes_set and not set(modes_set) >= set(str(m) for m in ear["m"].dropna().unique()):
                print("WARNING: EmissionActivityRatio modes not all in MODE_OF_OPERATION set.")


# ---------------------------------------------------------------------------
# Dummy workbook generator
# ---------------------------------------------------------------------------
def make_dummy_workbook(
    out_path: Path,
    *,
    scenario: str = "Reference",
    years: list[int] | None = None,
    region: str = "R1",
    timeslice: str = "ANNUAL",
    gen_tech: str = "GEN_GAS",
    supply_tech: str = "GAS_SUPPLY",  # unused in simplified dummy
    input_fuel: str = "GAS",  # unused in simplified dummy
    output_fuel: str = "ELEC",
    mode: str = "1",
):
    years = years or [2017, 2018]
    techs = [gen_tech]
    fuels = [output_fuel]

    # Set sheets (VALUE column)
    set_frames = {
        "REGION": pd.DataFrame({"VALUE": [region]}),
        "TECHNOLOGY": pd.DataFrame({"VALUE": techs}),
        "FUEL": pd.DataFrame({"VALUE": fuels}),
        "TIMESLICE": pd.DataFrame({"VALUE": [timeslice]}),
        "EMISSION": pd.DataFrame({"VALUE": []}),  # empty but present
        "MODE_OF_OPERATION": pd.DataFrame({"VALUE": [mode]}),
        "YEAR": pd.DataFrame({"VALUE": years}),
    }

    def wide(df_dict: dict[str, list], value_per_year):
        """Attach per-year columns. Accepts either a scalar or a list/tuple with one value per year."""
        df = pd.DataFrame(df_dict)
        for idx, y in enumerate(years):
            if isinstance(value_per_year, (list, tuple)) and len(value_per_year) == len(years):
                val = value_per_year[idx]
            else:
                val = value_per_year
            df[str(y)] = val
        return df

    # Demand for output fuel
    specified_demand = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "FUEL": [output_fuel],
            "UNITS": ["GWh"],
        },
        [0, 0],  # zero demand to keep dummy trivially feasible
    )

    # InputActivityRatio omitted (fuel-free generator)
    iar = pd.DataFrame(columns=["SCENARIO", "REGION", "TECHNOLOGY", "FUEL", "MODE_OF_OPERATION"])
    oar = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "FUEL": [output_fuel],
            "MODE_OF_OPERATION": [mode],
            "UNITS": ["PJ/PJ"],
        },
        [1.0, 1.0],
    )

    # Simple capacity/availability and costs
    availability_factor = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "TIMESLICE": [timeslice],
            "UNITS": ["fraction"],
        },
        [0.9, 0.9],
    )
    capital_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["$/kW"],
        },
        [1200, 1100],
    )
    fixed_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["$/kW-yr"],
        },
        [30, 30],
    )
    variable_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "MODE_OF_OPERATION": [mode],
            "UNITS": ["$/MWh"],
        },
        [10, 10],
    )

    # Tables without year columns (use VALUE)
    capacity_per_unit = pd.DataFrame(
        {
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW/unit"],
            "VALUE": [200],
        }
    )
    capacity_to_activity = pd.DataFrame(
        {
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["GWh/MWyr"],
            "VALUE": [8.76],
        }
    )

    # YearSplit: single timeslice sums to 1
    yearsplit = wide(
        {
            "TIMESLICE": [timeslice],
            "UNITS": ["fraction"],
        },
        [1.0, 1.0],
    )

    total_annual_max_capacity = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW"],
        },
        [1_000_000_000, 1_000_000_000],  # effectively unconstrained
    )
    residual_capacity = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW"],
        },
        [1000, 1000],  # seed enough capacity to satisfy demand
    )

    sheets: dict[str, pd.DataFrame] = {
        **set_frames,
        "SpecifiedAnnualDemand": specified_demand,
        "InputActivityRatio": iar,
        "OutputActivityRatio": oar,
        # Converter expects this sheet name to populate AvailabilityFactor table
        "AvailabilityFactor": availability_factor,
        "VariableCost": variable_cost,
        "CapitalCost": capital_cost,
        "FixedCost": fixed_cost,
        # Seed residual capacity and give a finite max cap so capacity binds correctly.
        "ResidualCapacity": residual_capacity,
        "TotalAnnualMaxCapacity": total_annual_max_capacity,
        "CapacityOfOneTechnologyUnit": capacity_per_unit,
        "CapacityToActivityUnit": capacity_to_activity,
        "YearSplit": yearsplit,
    }
    # # Include both names to avoid sheet-name mismatches
    # sheets["AvailabilityFactor"] = availability_factor.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Wrote dummy workbook to {out_path}")


def dummy_main(out_path: str | Path | None = None):
    """
    Entrypoint usable both from CLI and notebooks.
    - If a path argument is provided, use it.
    - Otherwise, default to ../data/dummy_osemosys.xlsx relative to cwd (works in notebooks).
    """
    default_path = Path("../data/dummy_osemosys.xlsx")

    # If caller passed a notebook/kernel flag like "--f=...", ignore it.
    if out_path is not None and str(out_path).startswith("-"):
        out_path = None

    if out_path is None:
        cli_paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
        out_path = cli_paths[0] if cli_paths else default_path

    make_dummy_workbook(Path(out_path))


# ---------------------------------------------------------------------------
# Default configuration (less frequently tweaked)
# ---------------------------------------------------------------------------
DEFAULTS = {
    #defaults for main.py
    "EXPORT_DB_TO_EXCEL": True,
    "EXPORT_RESULTS_TO_EXCEL": True,
    "EXPORT_RESULTS_TO_EXCEL_PATH": None,
    "EXPORT_RESULTS_WIDE_TO_EXCEL": True,
    "EXPORT_RESULTS_WIDE_TO_EXCEL_PATH": None,
    "PLOT_RESULTS": True,
    "PLOT_RESULTS_DIR": None,
    "PLOT_RESULTS_TABLES": None,
    "PLOT_RESULTS_MAX_SERIES": 12,
    # Paths (override in main if needed)
    "TEMPLATE_DB": "nemo_template.sqlite",
    "OUTPUT_DB": "nemo.sqlite",
    "NEMO_CONFIG_PATH": "nemo.cfg",
    "NEMO_TEST_DB_DIR": "nemo_tests",
    "NEMO_TEST_NAME": "storage_test",
    "TEST_OUTPUT_DB": "nemo_test.sqlite",
    "TEST_INPUT_MODE": "nemo_entry",
    # Units
    "TARGET_UNITS": {"energy": "PJ", "power": "GW"},
    "USE_UNIT_CONVERSION": False,
    # Transmission / demand fuel handling
    "ENABLE_NEMO_TRANSMISSION_METHODS": False,
    "REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS": False,
    "TEST_DB_PATH": None,
    "TEST_INPUT_PATH": None,
    "TEST_EXPORT_EXCEL_PATH": "nemo_entry_dump.xlsx",
    "TEST_EXPORT_DB_TO_EXCEL_PATH": "nemo_entry_dump.xlsx",
    # Template creation
    "AUTO_CREATE_TEMPLATE_DB": True,
    # Diagnostics
    "RUN_DIAGNOSTICS": True,
    "AUTO_FILL_MISSING_MODES": True,
    "STRICT_ERRORS": True,
    # NEMO / Julia
    "JULIA_EXE": r"C:\\ProgramData\\Julia\\Julia-1.9.3\\bin\\julia.exe",
    "NEMO_WRITE_LP": "intermediate_data/nemo_model_dump.lp"
}


def apply_defaults(user_vars: dict, data_dir: Path) -> dict:
    """Fill in less-frequently changed defaults into user_vars, resolving paths via data_dir where needed."""
    out = dict(user_vars)
    out.setdefault("EXPORT_DB_TO_EXCEL", DEFAULTS["EXPORT_DB_TO_EXCEL"])
    out.setdefault("TEMPLATE_DB", data_dir / DEFAULTS["TEMPLATE_DB"])
    out.setdefault("OUTPUT_DB", data_dir / DEFAULTS["OUTPUT_DB"])
    out.setdefault("NEMO_CONFIG_PATH", data_dir / DEFAULTS["NEMO_CONFIG_PATH"])
    out.setdefault("NEMO_TEST_DB_DIR", data_dir / DEFAULTS["NEMO_TEST_DB_DIR"])
    out.setdefault("NEMO_TEST_DB_PATH", None)
    out.setdefault("NEMO_TEST_NAME", DEFAULTS["NEMO_TEST_NAME"])
    out.setdefault("TEST_OUTPUT_DB", data_dir / DEFAULTS["TEST_OUTPUT_DB"])
    out.setdefault("TEST_INPUT_MODE", DEFAULTS["TEST_INPUT_MODE"])
    out.setdefault("TARGET_UNITS", DEFAULTS["TARGET_UNITS"])
    out.setdefault("USE_UNIT_CONVERSION", DEFAULTS["USE_UNIT_CONVERSION"])
    out.setdefault("ENABLE_NEMO_TRANSMISSION_METHODS", DEFAULTS["ENABLE_NEMO_TRANSMISSION_METHODS"])
    out.setdefault("REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS", DEFAULTS["REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS"])
    out.setdefault("TEST_INPUT_PATH", DEFAULTS["TEST_INPUT_PATH"])
    out.setdefault("TEST_DB_PATH", DEFAULTS["TEST_DB_PATH"] if DEFAULTS["TEST_DB_PATH"] is None else data_dir / DEFAULTS["TEST_DB_PATH"])
    out.setdefault("TEST_EXPORT_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_EXCEL_PATH"])
    out.setdefault("TEST_EXPORT_DB_TO_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_DB_TO_EXCEL_PATH"])
    out.setdefault("EXPORT_RESULTS_TO_EXCEL", DEFAULTS["EXPORT_RESULTS_TO_EXCEL"])
    out.setdefault("EXPORT_RESULTS_TO_EXCEL_PATH", DEFAULTS["EXPORT_RESULTS_TO_EXCEL_PATH"])
    out.setdefault("EXPORT_RESULTS_WIDE_TO_EXCEL", DEFAULTS["EXPORT_RESULTS_WIDE_TO_EXCEL"])
    out.setdefault("EXPORT_RESULTS_WIDE_TO_EXCEL_PATH", DEFAULTS["EXPORT_RESULTS_WIDE_TO_EXCEL_PATH"])
    out.setdefault("PLOT_RESULTS", DEFAULTS["PLOT_RESULTS"])
    out.setdefault("PLOT_RESULTS_DIR", DEFAULTS["PLOT_RESULTS_DIR"])
    out.setdefault("PLOT_RESULTS_TABLES", DEFAULTS["PLOT_RESULTS_TABLES"])
    out.setdefault("PLOT_RESULTS_MAX_SERIES", DEFAULTS["PLOT_RESULTS_MAX_SERIES"])
    out.setdefault("NEMO_TEST_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_EXCEL_PATH"])  # backward compat name
    out.setdefault("AUTO_CREATE_TEMPLATE_DB", DEFAULTS["AUTO_CREATE_TEMPLATE_DB"])
    out.setdefault("JULIA_EXE", DEFAULTS["JULIA_EXE"])
    out.setdefault("RUN_DIAGNOSTICS", DEFAULTS["RUN_DIAGNOSTICS"])
    out.setdefault("AUTO_FILL_MISSING_MODES", DEFAULTS["AUTO_FILL_MISSING_MODES"])
    out.setdefault("STRICT_ERRORS", DEFAULTS["STRICT_ERRORS"])
    out.setdefault("NEMO_WRITE_LP", DEFAULTS["NEMO_WRITE_LP"])
    return out


__all__ = [
    "ensure_template_db",
    "trim_db_years_in_place",
    "handle_test_run",
    "handle_storage_test",
    "ensure_nemo_test_db",
    "ensure_solver_test_script",
    "detect_solver_preference",
    "resolve_solver_from_test_name",
    "prepare_run_context",
    "analyze_log",
    "parse_years",
    "run_diagnostics",
    "make_dummy_workbook",
    "dummy_main",
    "DEFAULTS",
    "apply_defaults",
]
