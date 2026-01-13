from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd
import plotly.express as px

DEFAULT_CLEANING_OPTIONS = {
    "drop_zero_categories": True,
    "top_n_categories": 12,
    "top_n_include_other": True,
    "min_total": None,
    "min_share": None,
    "map_unmapped_to_other": True,
}

# Helpers shared with dashboard for cleaning/mapping
def apply_common_cleaning(df: pd.DataFrame, category_col: str, opts: dict) -> pd.DataFrame:
    """Drop/aggregate categories based on options."""
    merged = dict(DEFAULT_CLEANING_OPTIONS)
    merged.update(opts or {})
    drops = merged.get("drop_categories") or []
    if drops:
        df = df[~df[category_col].isin(drops)]
    if merged.get("drop_zero_categories"):
        totals = df.groupby(category_col)["VALUE"].sum()
        keep = totals[totals != 0].index
        df = df[df[category_col].isin(keep)]
    min_total = merged.get("min_total")
    min_share = merged.get("min_share")
    if min_total is not None or min_share is not None:
        totals = df.groupby(category_col)["VALUE"].sum().abs()
        if min_share is not None and totals.sum() > 0:
            min_total = max(min_total or 0, totals.sum() * float(min_share))
        if min_total:
            df = df[df[category_col].isin(totals[totals >= float(min_total)].index)]
    top_n = merged.get("top_n_categories")
    if top_n:
        totals = df.groupby(category_col)["VALUE"].sum().abs().sort_values(ascending=False)
        keep = set(totals.head(int(top_n)).index)
        if merged.get("top_n_include_other", True):
            df = df.copy()
            df[category_col] = df[category_col].where(df[category_col].isin(keep), "Other")
            df = df.groupby([category_col] + [c for c in df.columns if c not in {category_col, "VALUE"}], as_index=False)["VALUE"].sum()
        else:
            df = df[df[category_col].isin(keep)]
    if merged.get("aggregate_all"):
        total = df.copy()
        total[category_col] = "Total"
        df = pd.concat([df, total], ignore_index=True)
    return df


def _normalize_color_label(label) -> str | None:
    if label is None:
        return None
    return str(label).strip() or None


def update_missing_colors(labels: Iterable, color_map: dict, missing: set[str]):
    if not color_map:
        return
    known = {v for k in color_map.keys() if (v := _normalize_color_label(k))}
    for label in labels:
        normalized = _normalize_color_label(label)
        if normalized is None:
            continue
        if normalized not in known:
            missing.add(normalized)


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
    gen["TECHNOLOGY"] = map_name(gen["TECHNOLOGY"], powerplant_mapping, opts or {})
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
    cap["TECHNOLOGY"] = map_name(cap["TECHNOLOGY"], powerplant_mapping, opts or {})
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
    use["TECHNOLOGY"] = map_name(use["TECHNOLOGY"], powerplant_mapping, opts or {})
    use = apply_common_cleaning(use, "TECHNOLOGY", opts or {})
    use = use.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(use["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Input use by technology"
    return px.line(use, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_emissions_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    df = None
    try:
        df = pd.read_sql_query('SELECT t, y, val FROM "AnnualTechnologyEmission"', conn)
    except Exception:
        df = None

    if df is None or df.empty:
        # Fallback: compute emissions from vtotaltechnologyannualactivity * EmissionActivityRatio.
        try:
            activity = pd.read_sql_query(
                'SELECT t, y, val FROM "vtotaltechnologyannualactivity"',
                conn,
            )
            ear = pd.read_sql_query(
                'SELECT t, y, m, val FROM "EmissionActivityRatio"',
                conn,
            )
        except Exception:
            activity = pd.DataFrame()
            ear = pd.DataFrame()

        if activity.empty or ear.empty:
            return None

        activity = activity.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "ACTIVITY"})
        ear = ear.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "m": "MODE_OF_OPERATION", "val": "EAR"})
        ear = ear.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["EAR"].sum()
        df = activity.merge(ear, on=["TECHNOLOGY", "YEAR"], how="inner")
        if df.empty:
            return None
        df["VALUE"] = df["ACTIVITY"] * df["EAR"]
        df = df[["TECHNOLOGY", "YEAR", "VALUE"]]
    else:
        df = df.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})

    df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, opts or {})
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
    newcap["TECHNOLOGY"] = map_name(newcap["TECHNOLOGY"], powerplant_mapping, opts or {})
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


DEFAULT_Y_UNITS = {
    "generation": "PJ",
    "capacity": "GW",
    "input_use": "PJ",
    "emissions": "MtCO2",
    "new_capacity": "GW",
    "total_cost": "USD",
}

FUNCTION_BUILDERS = {
    "generation": build_generation_fig,
    "capacity": build_capacity_fig,
    "input_use": build_input_use_fig,
    "emissions": build_emissions_fig,
    "new_capacity": build_new_capacity_fig,
    "total_cost": build_total_cost_fig,
}


def map_name(series: pd.Series, df_map: pd.DataFrame | None, opts: dict | None = None) -> pd.Series:
    if df_map is None or "long_name" not in df_map or "plotting_name" not in df_map:
        return series
    df_map = df_map[["long_name", "plotting_name"]].dropna()
    mapping = dict(zip(df_map["long_name"], df_map["plotting_name"]))
    mapped = series.map(mapping)
    merged = dict(DEFAULT_CLEANING_OPTIONS)
    merged.update(opts or {})
    if merged.get("map_unmapped_to_other"):
        return mapped.fillna("Other (unmapped)")
    return mapped.fillna(series)


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
                y_units = (opts or {}).get("y_units") or DEFAULT_Y_UNITS.get(key)
                if y_units:
                    fig.update_yaxes(title_text=f"Value ({y_units})")
                figs.append(fig)
        except Exception as exc:
            print(f"Failed to build plot '{key}': {exc}")
            continue

    conn.close()
    return figs
