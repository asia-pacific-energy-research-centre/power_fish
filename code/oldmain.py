#%%
from pathlib import Path

from convert_osemosys_input_to_nemo import (
    convert_osemosys_input_to_nemo,
    dump_db_to_entry_excel,
    PARAM_SPECS,
)
from build_leap_import_template import generate_leap_template
from run_nemo_via_julia import run_nemo_on_db
from nemo_core import (
    ensure_template_db,
    trim_db_years_in_place,
    handle_storage_test,
    prepare_run_context,
    run_diagnostics,
    make_dummy_workbook,
    apply_defaults,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------
USER_VARS = {
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "ORIGINAL_OSEMOSYS_INPUT_SHEET_DO_NOT_MOD.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_dan.xlsx",
    # Scenario/name
    "SCENARIO": "Reference",
    # Export populated NEMO DB to Excel
    "EXPORT_DB_TO_EXCEL": True,
    "EXPORT_DB_TO_EXCEL_PATH": DATA_DIR / "nemo_entry_dump_dan.xlsx",
    # Years to use (None keeps all)
    "YEARS_TO_USE": [y for y in range(2017, 2020)],
    # LEAP template export
    "GENERATE_LEAP_TEMPLATE": False,
    "LEAP_TEMPLATE_OUTPUT": DATA_DIR / "leap_import_template.xlsx",
    "LEAP_TEMPLATE_REGION": None,  # Override region for LEAP export; None uses defaults in builder.
    "LEAP_IMPORT_ID_SOURCE": None,  # Optional path to existing LEAP import for ID reuse.
}

# Merge in less-frequently changed defaults (paths resolved relative to DATA_DIR where applicable)
VARS = apply_defaults(USER_VARS, DATA_DIR)

def main(mode: str | None = None, run_nemo: bool = True):
    """
    Flow summary:
      1) Determine mode (osemosys_input / nemo_input / dummy / storage_test)
      2) Build or reuse the template DB and (unless storage_test) run the conversion pipeline
      3) Optionally trim years, run diagnostics, and export supporting artifacts (Excel, LEAP template)
      4) Run NEMO once when run_nemo=True
      5) When GENERATE_LEAP_TEMPLATE is true, build the LEAP import template after NEMO finishes
    """
    cfg = dict(VARS)
    # Default mode if not provided
    mode = (mode or "nemo_entry").lower()

    # Handle storage test (skip conversion)
    if handle_storage_test(cfg, DATA_DIR, LOG_DIR, run_nemo):
        return

    # Configure mode specifics and ensure template DB
    cfg = prepare_run_context(cfg, DATA_DIR, mode, LOG_DIR)

    # Convert (handles both osemosys and nemo_entry)
    convert_osemosys_input_to_nemo(cfg)
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
# modes: list[str] = [
#     "osemosys_input",
#     "nemo_input",
#     "dummy",
#     # "storage_test",  # Uncomment to test storage test DB flow
# ]
if __name__ == "__main__":
    main('osemosys_input')

# %%
