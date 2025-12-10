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
    maybe_handle_storage_test,
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
    "INPUT_MODE": "nemo_entry",  # "osemosys" or "nemo_entry".. Note that this is normally set manually as an override at the bottom in the main() function call.
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "POWER 20_USA_data_REF9_S3_test - new file.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump.xlsx",
    # Scenario/name
    "SCENARIO": "Reference",
    # Export populated NEMO DB to Excel
    "EXPORT_DB_TO_EXCEL": True,
    "EXPORT_EXCEL_PATH": DATA_DIR / "nemo_entry_dump.xlsx",
    # Years to use (None keeps all)
    "YEARS_TO_USE": [y for y in range(2017, 2020)],
    # LEAP template export
    "GENERATE_LEAP_TEMPLATE": True,
    "LEAP_TEMPLATE_OUTPUT": DATA_DIR / "leap_import_template.xlsx",
    "LEAP_TEMPLATE_REGION": None,  # Override region for LEAP export; None uses defaults in builder.
    "LEAP_IMPORT_ID_SOURCE": None,  # Optional path to existing LEAP import for ID reuse.
}

# Merge in less-frequently changed defaults (paths resolved relative to DATA_DIR where applicable)
VARS = apply_defaults(USER_VARS, DATA_DIR)

def main(mode: str | None = None, run_nemo: bool = True):
    """
    Simple flow:
      1) Pick mode (osemosys_input / nemo_input / dummy / storage_test)
      2) Optional conversion (skipped for storage_test)
      3) Optional year trim and diagnostics
      4) Run NEMO once
    """
    mode = (mode or "osemosys_input").lower()

    # Handle storage test (skip conversion)
    if maybe_handle_storage_test(VARS, DATA_DIR, LOG_DIR, run_nemo):
        return

    # Ensure template DB exists
    ensure_template_db(
        VARS["TEMPLATE_DB"],
        auto_create=bool(VARS.get("AUTO_CREATE_TEMPLATE_DB", False)),
        julia_exe=VARS.get("JULIA_EXE"),
    )

    # Configure mode specifics
    if mode == "dummy":
        VARS["INPUT_MODE"] = "osemosys"
        VARS["OSEMOSYS_EXCEL_PATH"] = DATA_DIR / "dummy_osemosys.xlsx"
        VARS["NEMO_ENTRY_EXCEL_PATH"] = DATA_DIR / "dummy_nemo.xlsx"
        VARS["EXPORT_EXCEL_PATH"] = DATA_DIR / "dummy_nemo.xlsx"
        VARS["OUTPUT_DB"] = DATA_DIR / "dummy_nemo.sqlite"
        make_dummy_workbook(VARS["OSEMOSYS_EXCEL_PATH"])
    elif mode == "nemo_input":
        VARS["INPUT_MODE"] = "nemo_entry"
    elif mode == "osemosys_input":
        VARS["INPUT_MODE"] = "osemosys"
    else:
        raise ValueError(f"Unknown MODE '{mode}'")

    # Convert (handles both osemosys and nemo_entry)
    convert_osemosys_input_to_nemo(VARS)
    db_path = Path(VARS["OUTPUT_DB"])

    # Optional year trim and diagnostics
    if VARS.get("YEARS_TO_USE"):
        trim_db_years_in_place(db_path, VARS["YEARS_TO_USE"])
    if VARS.get("RUN_DIAGNOSTICS"):
        run_diagnostics(
            db_path,
            years=VARS.get("YEARS_TO_USE"),
            write_trimmed=None,
        )

    # Run NEMO once, at the end
    if run_nemo:
        run_nemo_on_db(
            db_path,
            julia_exe=VARS.get("JULIA_EXE"),
            log_path=LOG_DIR / "nemo_run.log",
            stream_output=True,
        )
    if VARS.get("GENERATE_LEAP_TEMPLATE"):
        breakpoint()#how is this going to work>
        generate_leap_template(
            scenario=VARS.get("SCENARIO"),
            region=VARS.get("LEAP_TEMPLATE_REGION"),
            nemo_db_path=db_path,
            output_path=VARS.get("LEAP_TEMPLATE_OUTPUT"),
            import_id_source=VARS.get("LEAP_IMPORT_ID_SOURCE"),
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
