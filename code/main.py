#%%
from pathlib import Path
import os

from convert_osemosys_input_to_nemo import (
    convert_osemosys_input_to_nemo,
    dump_db_to_entry_excel,
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
    apply_defaults,
)
from utils.pipeline_helpers import print_run_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = PROJECT_ROOT / "data" / "tests"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------
USER_VARS = {
    ################################
    # Input paths
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "ORIGINAL_OSEMOSYS_INPUT_SHEET_DO_NOT_MOD.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_dan.xlsx",
    ################################
    # Scenario/name
    "SCENARIO": "Reference",
    # Export populated NEMO DB to Excel
    "EXPORT_DB_TO_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_dan.xlsx",
    # Years to use (None keeps all)
    "YEARS_TO_USE": [y for y in range(2017, 2019+1)],
    # LEAP template export
    "GENERATE_LEAP_TEMPLATE": False,
    "LEAP_TEMPLATE_OUTPUT": DATA_DIR / "leap_import_template.xlsx",
    "LEAP_IMPORT_ID_SOURCE": DATA_DIR / "import_files/USA_power_leap_import_REF.xlsx", # Path to an existing LEAP export to copy IDs from (set to None to skip). # e.g. Path("../data/import_files/USA_power_leap_import_REF.xlsx")
    
    ################################
    # test run configuration - only runs if mode=='test'
    # skip conversion and run a test DB
    #   - Point to a local .sqlite or .xlsx via TEST_INPUT_PATH (Excel will be converted)
    #   - Or auto-download an upstream NEMO test DB via NEMO_TEST_NAME (stored in data/nemo_tests/)
<<<<<<< HEAD
    "USE_TEST_DB": False,
    "TEST_INPUT_PATH": DATA_DIR / TEST_DIR /"nemo_entry_dump - cbc_tests.xlsx",
    "NEMO_TEST_NAME": "cbc_tests",  # options: storage_test, storage_transmission_test, ramp_test, or solver test names like cbc_tests/glpk_tests to auto-download and run the upstream solver test script
=======
    "TEST_INPUT_PATH": DATA_DIR / TEST_DIR /"nemo_entry_dump - storage_test.xlsx",
    "NEMO_TEST_NAME": "storage_test",  # options: storage_test, storage_transmission_test, ramp_test, or solver test names like cbc_tests/glpk_tests to auto-download and run the upstream solver test script
>>>>>>> a50a5401f3bd05a396d61f2cddd4259251c1d084
    "TEST_EXPORT_DB_TO_EXCEL_PATH": DATA_DIR / TEST_DIR / "test_output_dump.xlsx",
    ################################  
}

# Merge in less-frequently changed defaults (paths resolved relative to DATA_DIR where applicable)
VARS = apply_defaults(USER_VARS, DATA_DIR)
VARS = apply_leap_template_defaults(VARS, DATA_DIR)

def main(mode: str | None = None, run_nemo: bool = True):
    """
    Flow summary:
      1) Determine mode (osemosys_input / osemosys_input_xlsx / nemo_input / nemo_input_xlsx / dummy / test / db_only)
      2) Build or reuse the template DB and (unless test or db_only) run the conversion pipeline
      3) Optionally trim years, run diagnostics, and export supporting artifacts (Excel, LEAP template)
      4) Run NEMO once when run_nemo=True
      5) When GENERATE_LEAP_TEMPLATE is true, build the LEAP import template after NEMO finishes
    """
    cfg = dict(VARS)
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
        )
    if cfg.get("GENERATE_LEAP_TEMPLATE"):
        generate_leap_template(
            scenario=cfg.get("SCENARIO"),
            region=cfg.get("LEAP_TEMPLATE_REGION"),
            nemo_db_path=db_path,
            output_path=cfg.get("LEAP_TEMPLATE_OUTPUT"),
            import_id_source=cfg.get("LEAP_IMPORT_ID_SOURCE"),
        )

#%%
# modes:
#     "osemosys_input_xlsx",
#     "nemo_input_xlsx",
#     "db_only",
#     "dummy",
#     # "test",  # test a DB flow set by NEMO_TEST_NAME
# 
if __name__ == "__main__":
    main('nemo_input_xlsx')

# %%


## Daniel's notes - what changes were made
## Gave 2019 values of 99999 for CHP oil and other - to deal with infeasibilities
