from __future__ import annotations

from pathlib import Path

from convert_osemosys_input_to_nemo import dump_db_to_entry_excel, PARAM_SPECS
from run_nemo_via_julia import run_nemo_on_db


def handle_storage_test(vars_cfg: dict, data_dir: Path, run_nemo: bool) -> bool:
    """
    Handle the storage test shortcut (skip conversion) if enabled.
    Returns True if the flow was handled and the caller should exit early.
    """
    if not vars_cfg.get("USE_STORAGE_TEST_DB"):
        return False

    db_path = Path(vars_cfg.get("STORAGE_TEST_DB", data_dir / "storage_test.sqlite"))
    if not db_path.exists():
        raise FileNotFoundError(f"Storage test DB not found at '{db_path}'")

    if vars_cfg.get("EXPORT_DB_TO_EXCEL"):
        dump_db_to_entry_excel(
            db_path=db_path,
            excel_path=Path(vars_cfg.get("STORAGE_TEST_EXCEL_PATH", data_dir / "storage_test_dump.xlsx")),
            specs=PARAM_SPECS,
            tables=None,
            max_rows=None,
            use_advanced=None,
        )

    if run_nemo:
        run_nemo_on_db(
            db_path,
            julia_exe=vars_cfg.get("JULIA_EXE"),
            log_path=data_dir / "nemo_run.log",
            stream_output=True,
        )
    return True

