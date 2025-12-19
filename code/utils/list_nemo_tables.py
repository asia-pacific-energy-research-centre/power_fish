#%%
"""
Jupyter-friendly helper to list tables in the NEMO SQLite DB and show which ones
are already referenced in build_leap_import_template.py mappings.

Notebook usage:
    from list_nemo_tables import mapping_summary
    summary = mapping_summary()  # prints summary and returns a dict
    summary["unmapped_tables"]    # access pieces programmatically

CLI usage still works:
    python3 code/list_nemo_tables.py
"""

from __future__ import annotations

import sys
import sqlite3
import importlib.util
import types
import csv
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "data").exists():
            return parent
    return Path.cwd()


def list_db_tables(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return sorted([row[0] for row in cur.fetchall()])
    finally:
        conn.close()


def _ensure_pandas_stub() -> None:
    """
    Provide a minimal pandas stub so we can import build_leap_import_template.py
    in lightweight environments (e.g., Jupyter kernel without pandas installed).
    """
    if "pandas" in sys.modules:
        return
    try:
        import pandas  # noqa: F401
        return
    except Exception:
        pass

    def _unavailable(*args, **kwargs):
        raise ImportError("pandas is not installed; install it to use DataFrame utilities.")

    pd_stub = types.SimpleNamespace(
        DataFrame=object,
        Series=object,
        NA=None,
        notna=lambda x: True,
        read_excel=_unavailable,
    )
    sys.modules["pandas"] = pd_stub


def load_build_module(template_path: Path):
    """
    Import build_leap_import_template.py with a pandas stub if needed so we can
    access BASE_MAPPINGS/PROCESS_MAPPING_TEMPLATES/MAPPINGS without requiring pandas.
    """
    _ensure_pandas_stub()
    spec = importlib.util.spec_from_file_location("build_leap_import_template", template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {template_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_unmapped_leap_vars(error_csv_path: Path | None = None) -> list[str]:
    """
    Look for the latest leap_id_errors.csv (created by validate_ids) and extract
    Variables where _merge == 'left_only'. Returns a sorted unique list.
    """
    if error_csv_path is None:
        return []
    if not error_csv_path.exists():
        return []

    vars_set = set()
    try:
        with error_csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("_merge") == "left_only":
                    var = row.get("Variable")
                    if var:
                        vars_set.add(var)
    except Exception:
        return []
    return sorted(vars_set)


def mapping_summary(db_path: Path | None = None, template_path: Path | None = None, verbose: bool = True) -> dict:
    """
    Return a summary dict with mapped/unmapped table sets and referenced mapping keys.
    If verbose, also prints a compact human-readable summary (useful in notebooks).
    """
    repo_root = find_repo_root()
    template_path = template_path or (repo_root / "code" / "build_leap_import_template.py")
    db_path = db_path or (repo_root / "data" / "nemo.sqlite")

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found at {db_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found at {template_path}")

    tables = list_db_tables(db_path)

    # Gather referenced tables from the actual mapping dicts.
    build_mod = load_build_module(template_path)
    mapping_dicts = []
    mapping_tables = set()
    mapping_keys = set()
    transform_keys = set()

    base = getattr(build_mod, "BASE_MAPPINGS", {})
    mapping_dicts.append(base)

    proc_templates = getattr(build_mod, "PROCESS_MAPPING_TEMPLATES", [])
    for _, md in proc_templates:
        if isinstance(md, dict):
            t = md.get("table")
            if t:
                mapping_tables.add(t)
            tf = md.get("transform")
            if tf:
                transform_keys.add(tf)

    expanded = getattr(build_mod, "MAPPINGS", {})
    mapping_dicts.append(expanded)

    for mdict in mapping_dicts:
        if not isinstance(mdict, dict):
            continue
        for key, md in mdict.items():
            mapping_keys.add(key)
            if isinstance(md, dict):
                t = md.get("table")
                if t:
                    mapping_tables.add(t)
                tf = md.get("transform")
                if tf:
                    transform_keys.add(tf)

    referenced_tables = set(mapping_tables)

    mapped_present = sorted(set(tables) & referenced_tables)
    unmapped = sorted(set(tables) - referenced_tables)
    mapped_missing = sorted(referenced_tables - set(tables))

    summary = {
        "db_path": db_path,
        "template_path": template_path,
        "all_tables": tables,
        "mapping_keys": sorted(mapping_keys),
        "transform_keys": sorted(transform_keys),
        "mapped_tables_from_dicts": sorted(mapping_tables),
        "referenced_tables_combined": sorted(referenced_tables),
        "mapped_tables_present_in_db": mapped_present,
        "mapped_tables_missing_from_db": mapped_missing,
        "unmapped_tables": unmapped,
        "unmapped_leap_vars_from_errors": load_unmapped_leap_vars(repo_root / "data" / "errors" / "leap_id_errors.csv"),
    }

    if verbose:
        print(f"DB path: {db_path}")
        print(f"Tables in DB: {len(tables)}")
        print(f"Mapping keys (Branch/Variable pairs): {len(mapping_keys)}")
        print(f"Transforms referenced: {', '.join(transform_keys) if transform_keys else '(none)'}")
        print(f"Mapped tables referenced in mapping dicts: {len(mapping_tables)}")
        print(f"Referenced tables (combined): {len(referenced_tables)}")
        print(f"Mapped tables that exist in DB: {len(mapped_present)}")
        print(f"Mapped tables that are NOT in DB: {len(mapped_missing)}")
        print(f"Unmapped tables remaining in DB: {len(unmapped)}")
        unmapped_leap_vars = summary["unmapped_leap_vars_from_errors"]
        print(f"Unmapped LEAP variables from latest error file: {len(unmapped_leap_vars)}")
        if unmapped:
            print("\nUnmapped table names:")
            for name in unmapped:
                print(f"  {name}")
        if mapped_missing:
            print("\nMapped names not found in DB (possible typos?):")
            for name in mapped_missing:
                print(f"  {name}")
        if unmapped_leap_vars:
            print("\nUnmapped LEAP Variables (_merge=left_only in leap_id_errors.csv):")
            for name in unmapped_leap_vars:
                print(f"  {name}")
    return summary



if __name__ == "__main__":
    summary = mapping_summary()  # prints and returns a dict
    summary["unmapped_tables"][:5]

#%%
