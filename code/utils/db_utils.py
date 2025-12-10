from __future__ import annotations

from pathlib import Path
import sqlite3

from run_nemo_via_julia import create_template_db


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
                f'DELETE FROM \"{tbl}\" WHERE y NOT IN ({years_param})',
                tuple(years),
            )
        if "YEAR" in tables:
            cur.execute(f'DELETE FROM \"YEAR\" WHERE val NOT IN ({years_param})', tuple(years))
        conn.commit()

