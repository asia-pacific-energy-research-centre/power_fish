"""
Build and load a minimal dummy OSeMOSYS workbook into a NEMO DB for smoke tests.
Does not call Julia/NEMO; it only generates the Excel and populates the SQLite DB.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from make_dummy_nemo_input import make_dummy_workbook
from convert_osemosys_input_to_nemo import convert_osemosys_input_to_nemo


def ensure_template_exists(template_db: Path):
    if not template_db.exists():
        raise FileNotFoundError(
            f"Template DB not found at {template_db}. Create it once via Julia "
            "(NemoMod.createnemodb) or run main.py with AUTO_CREATE_TEMPLATE_DB=True."
        )


def sanity_check(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        def count(table: str) -> int:
            try:
                return cur.execute(f"select count(*) from \"{table}\"").fetchone()[0]
            except Exception:
                return -1
        af = count("AvailabilityFactor")
        cf = count("CapacityFactor")
        ys = count("YearSplit")
        iar = count("InputActivityRatio")
        oar = count("OutputActivityRatio")
        print(f"Rows - AvailabilityFactor: {af}, CapacityFactor: {cf}, YearSplit: {ys}, IAR: {iar}, OAR: {oar}")
        if af <= 0:
            raise RuntimeError("AvailabilityFactor is empty; check the dummy workbook generation.")
        if ys <= 0:
            raise RuntimeError("YearSplit is empty; dummy workbook should populate it.")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    excel_path = data_dir / "dummy_osemosys.xlsx"
    output_db = data_dir / "dummy_nemo.sqlite"
    template_db = data_dir / "nemo_template.sqlite"

    make_dummy_workbook(
        excel_path,
        scenario="Reference",
        years=[2017, 2018],
        region="R1",
        timeslice="ANNUAL",
        gen_tech="GEN_GAS",
        input_fuel="GAS",
        output_fuel="ELEC",
        mode="1",
    )

    ensure_template_exists(template_db)

    config = {
        "INPUT_MODE": "osemosys",
        "OSEMOSYS_EXCEL_PATH": excel_path,
        "NEMO_ENTRY_EXCEL_PATH": excel_path,  # unused in this mode
        "TEMPLATE_DB": template_db,
        "OUTPUT_DB": output_db,
        "SCENARIO": "Reference",
        "AUTO_CREATE_TEMPLATE_DB": False,
        "USE_ADVANCED": {"ReserveMargin": False, "AnnualEmissionLimit": False},
        "EXPORT_DB_TO_EXCEL": False,
        "STRICT_ERRORS": True,
    }

    convert_osemosys_input_to_nemo(config)
    sanity_check(output_db)
    print(f"Dummy pipeline complete. DB written to {output_db}")


if __name__ == "__main__":
    main()
