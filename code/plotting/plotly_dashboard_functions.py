from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd
import plotly.express as px

# Helpers shared with dashboard for cleaning/mapping
def apply_common_cleaning(df: pd.DataFrame, category_col: str, opts: dict) -> pd.DataFrame:
    """Drop/aggregate categories based on options."""
    drops = opts.get("drop_categories") or []
    if drops:
        df = df[~df[category_col].isin(drops)]
    if opts.get("drop_zero_categories"):
        totals = df.groupby(category_col)["VALUE"].sum()
        keep = totals[totals != 0].index
        df = df[df[category_col].isin(keep)]
    if opts.get("aggregate_all"):
        total = df.copy()
        total[category_col] = "Total"
        df = pd.concat([df, total], ignore_index=True)
    return df


def update_missing_colors(labels: Iterable, color_map: dict, missing: set[str]):
    if not color_map:
        return
    known = {str(k) for k in color_map.keys()}
    for label in labels:
        if label is None:
            continue
        if str(label) not in known:
            missing.add(str(label))


def normalize_function_figs(fn_cfg) -> list[tuple[str, dict]]:
    """
    Acceptable forms:
      - list/tuple/set of names -> [(name, {})...]
      - dict of name -> opts
      - None -> []
    """
    if not fn_cfg:
        return []
    if isinstance(fn_cfg, (list, tuple, set)):
        return [(name, {}) for name in fn_cfg]
    if isinstance(fn_cfg, dict):
        return [(name, opts or {}) for name, opts in fn_cfg.items()]
    return []


def build_generation_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    gen = pd.read_sql_query('SELECT t, f, y, val FROM "vproductionbytechnologyannual"', conn)
    gen = gen.rename(columns={"t": "TECHNOLOGY", "f": "FUEL", "y": "YEAR", "val": "VALUE"})
    gen["TECHNOLOGY"] = map_name(gen["TECHNOLOGY"], powerplant_mapping)
    gen = gen[~gen["TECHNOLOGY"].str.contains("Storage", case=False, na=False)]
    gen = apply_common_cleaning(gen, "TECHNOLOGY", opts or {})
    gen = gen.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(gen["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Generation by technology"
    return px.area(gen, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_capacity_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    cap = pd.read_sql_query('SELECT t, y, val FROM "vtotalcapacityannual"', conn)
    cap = cap.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    cap["TECHNOLOGY"] = map_name(cap["TECHNOLOGY"], powerplant_mapping)
    cap = apply_common_cleaning(cap, "TECHNOLOGY", opts or {})
    cap = cap.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(cap["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Capacity by technology"
    return px.area(cap, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_input_use_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    use = pd.read_sql_query('SELECT t, y, val FROM "vusebytechnologyannual"', conn)
    use = use.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    use["TECHNOLOGY"] = map_name(use["TECHNOLOGY"], powerplant_mapping)
    use = apply_common_cleaning(use, "TECHNOLOGY", opts or {})
    use = use.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(use["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Input use by technology"
    return px.line(use, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_emissions_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    fuel_mapping = mapping.get("fuel")
    df = pd.read_sql_query('SELECT t, y, val FROM "AnnualTechnologyEmission"', conn)
    if df.empty:
        return None
    df = df.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], fuel_mapping)
    df = apply_common_cleaning(df, "TECHNOLOGY", opts or {})
    df = df.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(df["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Emissions by technology"
    return px.area(df, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_new_capacity_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    newcap = pd.read_sql_query('SELECT t, y, val FROM "vnewcapacity"', conn)
    newcap = newcap.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    newcap["TECHNOLOGY"] = map_name(newcap["TECHNOLOGY"], powerplant_mapping)
    newcap = apply_common_cleaning(newcap, "TECHNOLOGY", opts or {})
    newcap = newcap.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(newcap["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "New capacity by technology"
    return px.bar(newcap, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_total_cost_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    """Example extra function: total discounted cost over years."""
    df = pd.read_sql_query('SELECT y, val FROM "vtotaldiscountedcost"', conn)
    if df.empty:
        return None
    df = df.rename(columns={"y": "YEAR", "val": "VALUE"})
    title = "Total discounted cost"
    return px.line(df, x="YEAR", y="VALUE", title=title)


FUNCTION_BUILDERS = {
    "generation": build_generation_fig,
    "capacity": build_capacity_fig,
    "input_use": build_input_use_fig,
    "emissions": build_emissions_fig,
    "new_capacity": build_new_capacity_fig,
    "total_cost": build_total_cost_fig,
}


def map_name(series: pd.Series, df_map: pd.DataFrame | None) -> pd.Series:
    if df_map is None or "long_name" not in df_map or "plotting_name" not in df_map:
        return series
    df_map = df_map[["long_name", "plotting_name"]].dropna()
    return series.map(dict(zip(df_map["long_name"], df_map["plotting_name"]))).fillna(series)


def build_function_figs(
    db_path: Path,
    mapping: dict | None = None,
    include: list[tuple[str, dict]] | None = None,
    missing_colors: set[str] | None = None,
) -> list:
    """
    Build plots via dedicated functions (keys map to functions in FUNCTION_BUILDERS).
    include:
      - None: include all known function plots
      - []: include none
      - list of (name, opts) tuples to include
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    mapping = mapping or {}
    include_list = include if include is not None else [(name, {}) for name in FUNCTION_BUILDERS.keys()]
    if include is not None and len(include_list) == 0:
        conn.close()
        return []

    figs: list = []
    missing_colors = missing_colors if missing_colors is not None else set()
    for key, opts in include_list:
        builder = FUNCTION_BUILDERS.get(key)
        if not builder:
            continue
        try:
            fig = builder(conn, mapping, missing_colors, opts or {})
            if fig is not None:
                figs.append(fig)
        except Exception as exc:
            print(f"Failed to build plot '{key}': {exc}")
            continue

    conn.close()
    return figs
