#%%
from pathlib import Path
import sqlite3

from convert_osemosys_input_to_nemo import (
    convert_osemosys_input_to_nemo,
    dump_db_to_entry_excel,
    PARAM_SPECS,
)
from run_nemo_via_julia import create_template_db, run_nemo_on_db
from diagnostics import run_diagnostics
from make_dummy_nemo_input import make_dummy_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------
VARS = {
    "INPUT_MODE": "nemo_entry",  # "osemosys" or "nemo_entry"
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "POWER 20_USA_data_REF9_S3_test.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "nemo_entry_dump.xlsx",
    "TEMPLATE_DB": DATA_DIR / "nemo_template.sqlite",
    "OUTPUT_DB": DATA_DIR / "nemo.sqlite",
    # Target units for automatic scaling (energy defaults to PJ to keep magnitudes moderate)
    "TARGET_UNITS": {
        "energy": "PJ",
        "power": "MW",
    },
    # Optional: point directly at a ready-made NEMO DB (e.g., NEMO storage test).
    "USE_STORAGE_TEST_DB": False,
    "STORAGE_TEST_DB": DATA_DIR / "storage_test.sqlite",
    "EXPORT_STORAGE_TEST_TO_EXCEL": True,
    "STORAGE_TEST_EXCEL_PATH": DATA_DIR / "nemo_entry_dump.xlsx",
    ########

    "SCENARIO": "Reference",
    "AUTO_CREATE_TEMPLATE_DB": True,
    "USE_ADVANCED": {
        "ReserveMargin": True,
        "AnnualEmissionLimit": True,
    },
    # Export the populated NEMO DB to an Excel workbook in the NEMO format (one sheet per table).
    "EXPORT_DB_TO_EXCEL": True,
    "EXPORT_EXCEL_PATH": DATA_DIR / "nemo_entry_dump.xlsx",
    # Restrict export to a subset of tables/sheets (None -> all in PARAM_SPECS).
    "EXPORT_TABLE_FILTER": None,
    # Limit rows per table when exporting (None -> all rows).
    "EXPORT_MAX_ROWS": None,
    # Optional: set to the Julia executable path; if None, will look at env JULIA_EXE/NEMO_JULIA_EXE or PATH.
    "JULIA_EXE": r"C:\ProgramData\Julia\Julia-1.9.3\bin\julia.exe",
    # Diagnostics
    "RUN_DIAGNOSTICS": True,  # run a read-only health report on the DB
    # YEARS_TO_USE trims the DB in-place to these years after conversion and before running NEMO/diagnostics.
    # Set to None to keep all years. Example: [2017, 2018]
    "YEARS_TO_USE": [y for y in range(2017, 2031)],
    "AUTO_FILL_MISSING_MODES": True,
    "STRICT_ERRORS": True,
}


def ensure_template_db(template_path: Path, auto_create: bool, julia_exe: str | Path | None):
    """
    Make sure the NEMO template DB exists; optionally create it with Julia if missing.
    """
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
    """
    Remove rows from all tables that have a 'y' column for years not in the list.
    Updates YEAR table to match. Operates in-place.
    """
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
    if VARS.get("USE_STORAGE_TEST_DB"):
        db_path = Path(VARS.get("STORAGE_TEST_DB", DATA_DIR / "storage_test.sqlite"))
        if not db_path.exists():
            raise FileNotFoundError(f"Storage test DB not found at '{db_path}'")
        if VARS.get("EXPORT_STORAGE_TEST_TO_EXCEL"):
            dump_db_to_entry_excel(
                db_path=db_path,
                excel_path=Path(
                    VARS.get("STORAGE_TEST_EXCEL_PATH", DATA_DIR / "storage_test_dump.xlsx")
                ),
                specs=PARAM_SPECS,
                tables=VARS.get("EXPORT_TABLE_FILTER"),
                max_rows=VARS.get("EXPORT_MAX_ROWS"),
                use_advanced=VARS.get("USE_ADVANCED"),
            )
        if run_nemo:
            run_nemo_on_db(
                db_path,
                julia_exe=VARS.get("JULIA_EXE"),
                log_path=DATA_DIR / "nemo_run.log",
                stream_output=True,
            )
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
            log_path=DATA_DIR / "nemo_run.log",
            stream_output=True,
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
