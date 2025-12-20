from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable, Sequence

import pandas as pd
import matplotlib.pyplot as plt

# Map short column codes to verbose labels for readability in legends/axes.
INDEX_NAME_MAP_REV = {
    "r": "REGION",
    "l": "NODE",
    "t": "TECHNOLOGY",
    "f": "FUEL",
    "e": "EMISSION",
    "s": "STORAGE",
    "m": "MODE_OF_OPERATION",
    "ts": "TIMESLICE",
    "y": "YEAR",
}


def _list_result_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'v%' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    skip = {"Version", "VariableCost"}
    return [row[0] for row in cur.fetchall() if row[0] not in skip]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: INDEX_NAME_MAP_REV.get(col, col) for col in df.columns}
    if "val" in df.columns:
        rename_map["val"] = "VALUE"
    if "solvedtm" in df.columns:
        rename_map["solvedtm"] = "SOLVEDTM"
    return df.rename(columns=rename_map)


def plot_result_tables(
    db_path: Path,
    output_dir: Path,
    tables: Iterable[str] | None = None,
    max_series: int = 12,
    show: bool = False,
) -> list[Path]:
    """
    Create basic matplotlib line plots for result tables (v*) with YEAR on X.
    Categorical columns become legend labels; saves one PNG per table.
    """
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    all_tables = _list_result_tables(conn)
    allowed = set(tables) if tables else None

    saved: list[Path] = []
    for tbl in all_tables:
        if allowed and tbl not in allowed:
            continue
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn)
        except Exception as exc:
            print(f"  Skipping plot for '{tbl}': {exc}")
            continue

        if df.empty:
            print(f"  Skipping plot for '{tbl}': empty table")
            continue

        df = _rename_columns(df)
        if "YEAR" not in df.columns or "VALUE" not in df.columns:
            print(f"  Skipping plot for '{tbl}': missing YEAR/VALUE columns")
            continue

        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df = df.dropna(subset=["YEAR"])
        categorical_cols = [
            c for c in df.columns if c not in {"YEAR", "VALUE", "SOLVEDTM"}
        ]
        if not categorical_cols:
            df["_group"] = "total"
            categorical_cols = ["_group"]

        groups = df.groupby(categorical_cols)
        # Limit number of series to avoid unreadable plots.
        groups_to_plot = list(groups)[:max_series]

        plt.figure(figsize=(8, 5))
        for keys, g in groups_to_plot:
            label = keys if isinstance(keys, str) else ", ".join(map(str, keys))
            g = g.sort_values("YEAR")
            plt.plot(g["YEAR"], g["VALUE"], marker="o", label=label)

        if len(groups) > max_series:
            plt.title(f"{tbl} (showing {max_series} of {len(groups)} series)")
        else:
            plt.title(tbl)
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend(loc="best")
        plt.tight_layout()

        out_path = output_dir / f"{tbl}.png"
        plt.savefig(out_path)
        if show:
            plt.show()
        plt.close()
        saved.append(out_path)
        print(f"  Saved plot: {out_path}")

    conn.close()
    return saved
