#%%
from pathlib import Path
import json
import os
import shutil
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from convert_osemosys_input_to_nemo import (
    convert_osemosys_input_to_nemo,
    dump_db_to_entry_excel,
    export_results_to_excel,
    export_results_to_excel_wide,
    PARAM_SPECS,
)
from build_leap_import_template import (
    generate_leap_template,
    apply_leap_template_defaults,
)
from run_nemo_via_julia import run_nemo_on_db
from nemo_core import (
    ensure_template_db,
    trim_db_years_in_place,
    handle_test_run,
    prepare_run_context,
    run_diagnostics,
    make_dummy_workbook,
)
from config_defaults import apply_defaults
from utils.pipeline_helpers import (
    print_run_summary,
    _as_bool,
    run_postprocess_steps,
)
from plotting.plotly_dashboard import generate_plotly_dashboard

TESTS_ROOT = PROJECT_ROOT / "tests"
DATA_DIR = TESTS_ROOT / "data"
TEST_DIR = TESTS_ROOT / "data" / "tests"
LOG_DIR = TESTS_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# AGENT TEST SETTINGS
# -------------------------------------------------------------------
# Edit these to trade off speed vs coverage for quick checks.
AGENT_TEST_MODE = "osemosys_input_xlsx"  # options: results, test, dummy, db_only, osemosys_input_xlsx, nemo_input_xlsx
AGENT_TEST_RUN_NEMO = True  # keep False to avoid running Julia by default
AGENT_TEST_USE_LAST_RUN = False
AGENT_TEST_LAST_RUN_CONFIG = TESTS_ROOT / "last_run" / "config.json"
AGENT_TEST_RUNTIME_STATE = TESTS_ROOT / "last_run" / "runtime.json"
AGENT_TEST_OVERRIDES = {
    "RUN_POSTPROCESS_ONLY": False,  # True for mode="results" to skip conversion/NEMO
    "NO_DATEID": True,
    "ECONOMY": "TEST",
    "RUN_DIAGNOSTICS": False,
    "EXPORT_DB_TO_EXCEL": False,
    "EXPORT_RESULTS_TO_EXCEL": True,
    "EXPORT_RESULTS_WIDE_TO_EXCEL": True,
    "PLOTLY_DASHBOARD": True,
    "PLOTLY_DASHBOARD_PATH": TESTS_ROOT / "plotting_output" / "agent_dashboard.html",
    "PLOTLY_CONFIG_YAML": PROJECT_ROOT / "config" / "plotly_charts.yml",
    "PLOTLY_PNG_DIR": TESTS_ROOT / "plotting_output" / "png",
    # Keep None to avoid trim costs for quick tests; set to a small list when needed.
    "YEARS_TO_USE": None,
}
AGENT_TEST_DB_CANDIDATES = [
    TESTS_ROOT / "last_run" / "output_db.sqlite",
    TESTS_ROOT / "results" / "nemo_final.sqlite",
    TESTS_ROOT / "data" / "nemo.sqlite",
    TESTS_ROOT / "data" / "nemo_test.sqlite",
]


# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------
USER_VARS = {
    ################################
    # Input paths
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "POWER 20_USA_data_REF9_S3_test.xlsx",  # - no heat.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_reflection.xlsx",  # - storage test 2.xlsx",#nemo_entry_dump.xlsx",
    # Optional NEMO config (nemo.cfg / nemo.ini). When set, Julia runs from the parent dir
    # so NEMO can pick it up; leave as None to skip.
    "NEMO_CONFIG_PATH": DATA_DIR / "nemo.cfg",
    ################################
    # Scenario/name
    "SCENARIO": "Reference",
    # Export populated NEMO DB to Excel
    "EXPORT_DB_TO_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_reflection2.xlsx",
    "EXPORT_RESULTS_TO_EXCEL_PATH": TESTS_ROOT / "results" / "results.xlsx",
    "EXPORT_RESULTS_WIDE_TO_EXCEL_PATH": TESTS_ROOT / "results" / "results_wide.xlsx",
    # Years to use (None keeps all)
    "YEARS_TO_USE": [y for y in range(2020, 2050+1)],
    # LEAP template export
    "GENERATE_LEAP_TEMPLATE": False,
    "LEAP_TEMPLATE_OUTPUT": DATA_DIR / "leap_import_template.xlsx",
    "LEAP_IMPORT_ID_SOURCE": DATA_DIR / "import_files/USA_power_leap_import_REF.xlsx", # Path to an existing LEAP export to copy IDs from (set to None to skip). # e.g. Path("../data/import_files/USA_power_leap_import_REF.xlsx")
    
    ################################
    # test run configuration - only runs if mode=='test'
    # skip conversion and run a test DB
    #   - Point to a local .sqlite or .xlsx via TEST_INPUT_PATH (Excel will be converted)
    #   - Or auto-download an upstream NEMO test DB via NEMO_TEST_NAME (stored in data/nemo_tests/)
    "TEST_INPUT_PATH": TEST_DIR / "nemo_entry_dump - storage test.xlsx",
    "NEMO_TEST_NAME": "storage_test",  # options: storage_test, storage_transmission_test, ramp_test, or solver test names like cbc_tests/glpk_tests to auto-download and run the upstream solver test script
    "TEST_EXPORT_DB_TO_EXCEL_PATH": TEST_DIR / "test_output_dump.xlsx",
    ################################  
}

# Merge in less-frequently changed defaults (paths resolved relative to DATA_DIR where applicable)
VARS = apply_defaults(USER_VARS, DATA_DIR)
VARS = apply_leap_template_defaults(VARS, DATA_DIR)

_SNAPSHOT_PATH_KEYS = {
    "OSEMOSYS_EXCEL_PATH",
    "NEMO_ENTRY_EXCEL_PATH",
    "NEMO_CONFIG_PATH",
    "LEAP_IMPORT_ID_SOURCE",
    "OUTPUT_DB",
    "PLOTLY_CONFIG_YAML",
}


def _load_last_run_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Failed to load last-run snapshot '{path}': {exc}")
        return None


def _load_runtime_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Failed to load runtime state '{path}': {exc}")
        return {}


def _save_runtime_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _apply_last_run_snapshot(cfg: dict, snapshot: dict | None) -> dict:
    if not snapshot:
        return dict(cfg)
    updated = dict(cfg)
    for key, val in snapshot.get("config", {}).items():
        if val is not None:
            updated[key] = val
    for key, info in snapshot.get("files", {}).items():
        if not info:
            continue
        snap_path = info.get("snapshot") if isinstance(info, dict) else info
        if snap_path:
            updated[key] = Path(snap_path) if key in _SNAPSHOT_PATH_KEYS else snap_path
    return updated


def _pick_existing_db(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _apply_agent_test_overrides(cfg: dict) -> dict:
    updated = dict(cfg)
    if AGENT_TEST_USE_LAST_RUN:
        snapshot = _load_last_run_snapshot(AGENT_TEST_LAST_RUN_CONFIG)
        updated = _apply_last_run_snapshot(updated, snapshot)
    if AGENT_TEST_OVERRIDES:
        updated.update(AGENT_TEST_OVERRIDES)
    if _as_bool(updated.get("RUN_POSTPROCESS_ONLY")):
        chosen = _pick_existing_db(AGENT_TEST_DB_CANDIDATES)
        if chosen:
            updated["OUTPUT_DB"] = chosen
        else:
            print(
                "No candidate DB found for postprocess-only test. "
                "Set OUTPUT_DB in AGENT_TEST_OVERRIDES or switch AGENT_TEST_MODE."
            )
    return updated


def _run_with_cfg(cfg: dict, mode: str | None = None, run_nemo: bool = True):
    """
    Flow summary:
      1) Determine mode (osemosys_input / osemosys_input_xlsx / nemo_input / nemo_input_xlsx / dummy / test / db_only)
      2) Build or reuse the template DB and (unless test or db_only) run the conversion pipeline
      3) Optionally trim years, run diagnostics, and export supporting artifacts (Excel, LEAP template)
      4) Run NEMO once when run_nemo=True
      5) When GENERATE_LEAP_TEMPLATE is true, build the LEAP import template after NEMO finishes
    """
    cfg = dict(cfg)
    # Allow env override for postprocess flag (e.g., RUN_POSTPROCESS_ONLY=true)
    env_rpo = os.environ.get("RUN_POSTPROCESS_ONLY")
    if env_rpo is not None:
        cfg["RUN_POSTPROCESS_ONLY"] = env_rpo
    # Mode override: mode='results' forces postprocess-only
    if mode and mode.lower() == "results":
        cfg["RUN_POSTPROCESS_ONLY"] = True
    rpo_val = _as_bool(cfg.get("RUN_POSTPROCESS_ONLY"))

    # Skip conversion/NEMO and just run exports/plots on an existing DB.
    if rpo_val:
        print("RUN_POSTPROCESS_ONLY=True: skipping conversion and NEMO run, proceeding to exports/plots only.")
        db_path = Path(cfg["OUTPUT_DB"])
        cfg["RUN_MODE"] = "postprocess"
        if not db_path.exists():
            raise FileNotFoundError(f"RUN_POSTPROCESS_ONLY=True but OUTPUT_DB not found at '{db_path}'")
        print_run_summary(cfg, LOG_DIR, postprocess_only=True)
        # Ensure we never run Julia when postprocessing
        run_nemo = False
        run_postprocess_steps(cfg, db_path, PROJECT_ROOT)
        return
    # Default mode if not provided
    mode = (mode or "nemo_entry").lower()
    mode_aliases = {
        "osemosys_input_xlsx": "osemosys_input",
        "nemo_input_xlsx": "nemo_input",
    }
    mode = mode_aliases.get(mode, mode)

    # Handle test shortcut (skip conversion) only when mode=='test'
    if mode == "test":
        cfg["RUN_MODE"] = mode
        if handle_test_run(cfg, DATA_DIR, LOG_DIR, run_nemo):
            return
        raise RuntimeError("Test mode selected but no test input configured.")

    db_path: Path
    cfg["RUN_MODE"] = mode
    if mode == "db_only":
        db_path = Path(cfg["OUTPUT_DB"])
        if not db_path.exists():
            raise FileNotFoundError(
                f"db_only mode selected but OUTPUT_DB not found at '{db_path}'. "
                "Point OUTPUT_DB to an existing NEMO database."
            )
        print_run_summary(cfg, LOG_DIR)
    else:
        # Configure mode specifics and ensure template DB
        cfg = prepare_run_context(cfg, DATA_DIR, mode, LOG_DIR)
        print_run_summary(cfg, LOG_DIR)

        # Convert (handles both osemosys and nemo_entry)
        convert_osemosys_input_to_nemo(cfg, VERBOSE_ERRORS=True)
        db_path = Path(cfg["OUTPUT_DB"])

    # Optional year trim and diagnostics
    if cfg.get("YEARS_TO_USE"):
        trim_db_years_in_place(db_path, cfg["YEARS_TO_USE"])
    if cfg.get("RUN_DIAGNOSTICS"):
        run_diagnostics(
            db_path,
            years=cfg.get("YEARS_TO_USE"),
            write_trimmed=None,
        )

    # Run NEMO once
    if run_nemo:
        if cfg.get("NEMO_WRITE_LP"):
            os.environ["NEMO_WRITE_LP"] = str(cfg["NEMO_WRITE_LP"])
        run_nemo_on_db(
            db_path,
            julia_exe=cfg.get("JULIA_EXE"),
            log_path=LOG_DIR / "nemo_run.log",
            stream_output=True,
            config_path=cfg.get("NEMO_CONFIG_PATH"),
            varstosave=cfg.get("VARS_TO_SAVE"),
        )
        # Save a copy of the solved DB to results for plotting/postprocessing
        final_db_path = PROJECT_ROOT / "results" / "nemo_final.sqlite"
        final_db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, final_db_path)
        cfg["OUTPUT_DB"] = final_db_path
        db_path = final_db_path

    # Postprocess outputs
    run_postprocess_steps(cfg, db_path, PROJECT_ROOT)


def main(mode: str | None = None, run_nemo: bool = True):
    return _run_with_cfg(VARS, mode=mode, run_nemo=run_nemo)


def main_agent_test():
    start = time.time()
    runtime_state = _load_runtime_state(AGENT_TEST_RUNTIME_STATE)
    cfg = _apply_agent_test_overrides(VARS)
    result = _run_with_cfg(cfg, mode=AGENT_TEST_MODE, run_nemo=AGENT_TEST_RUN_NEMO)
    elapsed = time.time() - start
    runtime_state[AGENT_TEST_MODE] = {
        "seconds": int(elapsed),
        "updated_at": time.strftime("%Y-%m-%d"),
        "run_nemo": bool(AGENT_TEST_RUN_NEMO),
    }
    _save_runtime_state(AGENT_TEST_RUNTIME_STATE, runtime_state)
    if AGENT_TEST_RUN_NEMO:
        suggested = int(elapsed * 5)
        mins, secs = divmod(int(elapsed), 60)
        print(
            "Suggested timeout for NEMO runs: "
            f"~{suggested}s (5x last run: {mins}m{secs:02d}s)"
        )
    else:
        mins, secs = divmod(int(elapsed), 60)
        print(f"Run time: {mins}m{secs:02d}s")
    return result

#%%
# modes:
#     "osemosys_input_xlsx",
#     "nemo_input_xlsx",
#     "db_only",
#     "dummy",
#     # "test",  # test a DB flow set by NEMO_TEST_NAME
#    "results"  # postprocess-only mode (skip conversion and NEMO run)
# 
# print('to do let daniel know that the nemo_entry_dump - storage test.xlsx file may be misspelled as nemo_entry_dump - storage_test.xlsx within the data/tests folder. If you wanted to  run it thorugh the regular process, you also need to chang the years')
# print('todo add an error for when the yers we want arent in the input data. since thats A COMOMN ACCIDENT. also we actually dont even have a way of processing the results...')
# i tink if we create a default params sheet itll be good.
#will also need to change the leap template creator so it doesnt accidntlaly pull results tables from hte sqlite db - but not Variablecosts!
#%%
if __name__ == "__main__":
    main_agent_test()

# %%

## Daniel's notes - what changes were made
## Gave 2019 values of 99999 for CHP oil and other - to deal with infeasibilities
