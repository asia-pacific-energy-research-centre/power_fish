#%%
from pathlib import Path
import os
import shutil

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
from utils.pipeline_helpers import print_run_summary, _as_bool, run_postprocess_steps
from plotting.plotly_dashboard import generate_plotly_dashboard

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
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "POWER 20_USA_data_REF9_S3_test.xlsx",# - no heat.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_reflection.xlsx",# - storage test 2.xlsx",#nemo_entry_dump.xlsx",
    # Optional NEMO config (nemo.cfg / nemo.ini). When set, Julia runs from the parent dir
    # so NEMO can pick it up; leave as None to skip.
    "NEMO_CONFIG_PATH": DATA_DIR / "nemo.cfg",
    ################################
    # Scenario/name
    "SCENARIO": "Reference",
    # Export populated NEMO DB to Excel
    "EXPORT_DB_TO_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_reflection2.xlsx",
    "EXPORT_RESULTS_TO_EXCEL_PATH": PROJECT_ROOT / "results" / "results.xlsx",
    "EXPORT_RESULTS_WIDE_TO_EXCEL_PATH": PROJECT_ROOT / "results" / "results_wide.xlsx",
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
    "TEST_INPUT_PATH": DATA_DIR / TEST_DIR /"nemo_entry_dump - storage test.xlsx",
    "NEMO_TEST_NAME": "storage_test",  # options: storage_test, storage_transmission_test, ramp_test, or solver test names like cbc_tests/glpk_tests to auto-download and run the upstream solver test script
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
        )
        # Save a copy of the solved DB to results for plotting/postprocessing
        final_db_path = PROJECT_ROOT / "results" / "nemo_final.sqlite"
        final_db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, final_db_path)
        cfg["OUTPUT_DB"] = final_db_path
        db_path = final_db_path

    # Postprocess outputs
    run_postprocess_steps(cfg, db_path, PROJECT_ROOT)

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
    main('results')#results')

# %%

## Daniel's notes - what changes were made
## Gave 2019 values of 99999 for CHP oil and other - to deal with infeasibilities
