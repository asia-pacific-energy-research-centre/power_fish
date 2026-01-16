# Summary: Plotly figure builders and shared helpers for the dashboard.
from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotting.mappings import EMISSIONS_FACTOR_FUEL_MAP

DEFAULT_CLEANING_OPTIONS = {
    "drop_zero_categories": True,
    "top_n_categories": 12,
    "top_n_include_other": True,
    "min_total": None,
    "min_share": None,
    "map_unmapped_to_other": True,
    "drop_category_substrings": None,
}

DEFAULT_GENERATION_TIMESLICE_FUELS = {
    "01_05_lignite",
    "01_x_thermal_coal",
    "02_coal_products",
    "07_07_gas_diesel_oil",
    "07_08_fuel_oil",
    "07_x_other_petroleum_products",
    "08_01_natural_gas",
    "08_01_natural_gas_CCS",
    "09_nuclear",
    "10_hydro",
    "11_geothermal",
    "12_01_of_which_photovoltaics",
    "12_solar",
    "14_wind",
    "15_solid_biomass",
    "16_others",
    "16_x_hydrogen",
}
DEFAULT_CAPACITY_TECH_INCLUDE_SUBSTRINGS = ("_CHP", "_PP")

# Helpers shared with dashboard for cleaning/mapping
def apply_common_cleaning(df: pd.DataFrame, category_col: str, opts: dict) -> pd.DataFrame:
    """Drop/aggregate categories based on options."""
    merged = dict(DEFAULT_CLEANING_OPTIONS)
    merged.update(opts or {})
    drops = merged.get("drop_categories") or []
    if drops:
        df = df[~df[category_col].isin(drops)]
    substrings = merged.get("drop_category_substrings") or []
    if substrings:
        parts = [re.escape(str(item)) for item in substrings if item not in (None, "")]
        if parts:
            pattern = "|".join(parts)
            df = df[~df[category_col].astype(str).str.contains(pattern, na=False)]
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
        normalized: list[tuple[str, dict]] = []
        for entry in fn_cfg:
            if isinstance(entry, str):
                normalized.append((entry, {}))
                continue
            if isinstance(entry, dict):
                name = entry.get("name")
                opts = entry.get("opts") or entry.get("options") or {}
                if name:
                    normalized.append((name, opts))
                    continue
                if len(entry) == 1:
                    name, opts = next(iter(entry.items()))
                    normalized.append((name, opts or {}))
        return normalized
    if isinstance(fn_cfg, dict):
        normalized: list[tuple[str, dict]] = []
        for name, opts in fn_cfg.items():
            if isinstance(opts, list):
                for entry in opts:
                    if isinstance(entry, dict):
                        normalized.append((name, entry))
                    else:
                        normalized.append((name, {}))
                continue
            normalized.append((name, opts or {}))
        return normalized
    return []


def build_generation_fig(conn: sqlite3.Connection, mapping: dict, missing_colors: set[str] | None = None, opts: dict | None = None):
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    gen = pd.read_sql_query('SELECT t, f, y, val FROM "vproductionbytechnologyannual"', conn)
    gen = gen.rename(columns={"t": "TECHNOLOGY", "f": "FUEL", "y": "YEAR", "val": "VALUE"})
    gen["TECHNOLOGY"] = map_name(gen["TECHNOLOGY"], powerplant_mapping, merged_opts)
    gen = gen[~gen["TECHNOLOGY"].str.contains("Storage", case=False, na=False)]
    gen = apply_common_cleaning(gen, "TECHNOLOGY", merged_opts)
    gen = gen.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    update_missing_colors(gen["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Generation by technology"
    fig = px.area(gen, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)
    if merged_opts.get("add_demand_line", True):
        try:
            demand = pd.read_sql_query('SELECT f, y, val FROM "SpecifiedAnnualDemand"', conn)
            demand = demand.rename(columns={"f": "FUEL", "y": "YEAR", "val": "VALUE"})
            demand["YEAR"] = pd.to_numeric(demand["YEAR"], errors="coerce")
            demand["VALUE"] = pd.to_numeric(demand["VALUE"], errors="coerce")
            fuel_filter = merged_opts.get("demand_fuel_filter")
            if fuel_filter:
                if isinstance(fuel_filter, (list, tuple, set)):
                    demand = demand[demand["FUEL"].isin(fuel_filter)]
                else:
                    demand = demand[demand["FUEL"] == fuel_filter]
            demand = demand.dropna(subset=["YEAR", "VALUE"])
            demand = demand.groupby(["YEAR"], as_index=False)["VALUE"].sum()
            if not demand.empty:
                fig.add_trace(
                    go.Scatter(
                        x=demand["YEAR"],
                        y=demand["VALUE"],
                        name=merged_opts.get("demand_line_label", "Demand"),
                        mode="lines+markers",
                        line=dict(
                            color=merged_opts.get("demand_line_color", "#000000"),
                            width=merged_opts.get("demand_line_width", 2),
                        ),
                        marker=dict(
                            color=merged_opts.get("demand_line_color", "#000000"),
                            size=merged_opts.get("demand_marker_size", 6),
                        ),
                    )
                )
        except Exception as exc:
            print(f"Failed to add demand line: {exc}")
    return fig


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


def build_capacity_factor_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build capacity factor by technology over time.

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, powerplant mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (hours_per_year, energy_unit_multiplier, as_percent, min_capacity)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    hours_per_year = float(merged_opts.get("hours_per_year", 8760))
    energy_unit_multiplier = float(merged_opts.get("energy_unit_multiplier", 0.0036))
    as_percent = merged_opts.get("as_percent", True)
    min_capacity = merged_opts.get("min_capacity")

    try:
        gen = pd.read_sql_query('SELECT t, y, val FROM "vproductionbytechnologyannual"', conn)
        cap = pd.read_sql_query('SELECT t, y, val FROM "vtotalcapacityannual"', conn)
    except Exception as exc:
        print(f"Failed to load capacity factor inputs: {exc}")
        return None

    if gen.empty or cap.empty:
        return None

    gen = gen.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    cap = cap.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    gen["TECHNOLOGY"] = map_name(gen["TECHNOLOGY"], powerplant_mapping, merged_opts)
    cap["TECHNOLOGY"] = map_name(cap["TECHNOLOGY"], powerplant_mapping, merged_opts)
    gen["YEAR"] = pd.to_numeric(gen["YEAR"], errors="coerce")
    cap["YEAR"] = pd.to_numeric(cap["YEAR"], errors="coerce")
    gen["VALUE"] = pd.to_numeric(gen["VALUE"], errors="coerce")
    cap["VALUE"] = pd.to_numeric(cap["VALUE"], errors="coerce")

    gen = gen.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    cap = cap.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    merged = gen.merge(cap, on=["TECHNOLOGY", "YEAR"], how="inner", suffixes=("_GEN", "_CAP"))

    if merged.empty:
        return None

    if min_capacity is not None:
        merged = merged[merged["VALUE_CAP"] >= float(min_capacity)]

    denominator = merged["VALUE_CAP"] * hours_per_year * energy_unit_multiplier
    merged["VALUE"] = merged["VALUE_GEN"] / denominator
    merged = merged.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["VALUE"])

    if as_percent:
        merged["VALUE"] = merged["VALUE"] * 100.0

    merged = merged[["TECHNOLOGY", "YEAR", "VALUE"]]
    merged = apply_common_cleaning(merged, "TECHNOLOGY", merged_opts)

    update_missing_colors(merged["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Capacity factor by technology"
    return px.line(merged, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_capacity_factor_timeslice_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build capacity factor (AvailabilityFactor) by timeslice for a given year or year-aggregate.

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, powerplant mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (year, year_agg, as_percent, timeslice_order)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    year_opt = merged_opts.get("year", "latest")
    year_agg = merged_opts.get("year_agg")
    as_percent = merged_opts.get("as_percent", True)
    timeslice_order = merged_opts.get("timeslice_order")

    try:
        df = pd.read_sql_query('SELECT r, t, l, y, val FROM "AvailabilityFactor"', conn)
    except Exception as exc:
        print(f"Failed to load AvailabilityFactor: {exc}")
        return None

    if df.empty:
        return None

    df = df.rename(columns={"r": "REGION", "t": "TECHNOLOGY", "l": "TIMESLICE", "y": "YEAR", "val": "VALUE"})
    df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, merged_opts)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["YEAR", "VALUE"])

    if year_agg:
        agg_func = str(year_agg).lower()
        if agg_func not in {"mean", "median", "min", "max"}:
            agg_func = "mean"
        df = df.groupby(["TECHNOLOGY", "TIMESLICE"], as_index=False)["VALUE"].agg(agg_func)
        year_label = f"{agg_func.title()} over years"
    else:
        if isinstance(year_opt, (list, tuple, set)):
            df = df[df["YEAR"].isin([float(y) for y in year_opt])]
            year_label = f"Years {', '.join(str(y) for y in year_opt)}"
        else:
            year_tag = str(year_opt).lower() if year_opt is not None else "latest"
            if year_tag in {"latest", "max"}:
                year_val = df["YEAR"].max()
            elif year_tag in {"earliest", "min"}:
                year_val = df["YEAR"].min()
            else:
                year_val = float(year_opt)
            df = df[df["YEAR"] == year_val]
            year_label = f"Year {int(year_val)}" if pd.notna(year_val) else "Selected year"
        df = df.groupby(["TECHNOLOGY", "TIMESLICE"], as_index=False)["VALUE"].mean()

    if df.empty:
        return None

    if as_percent:
        df["VALUE"] = df["VALUE"] * 100.0

    df = apply_common_cleaning(df, "TECHNOLOGY", merged_opts)
    update_missing_colors(df["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())

    title = f"Capacity factor by timeslice ({year_label})"
    fig = px.line(
        df,
        x="TIMESLICE",
        y="VALUE",
        color="TECHNOLOGY",
        title=title,
        color_discrete_map=tech_colors,
        category_orders={"TIMESLICE": timeslice_order} if timeslice_order else None,
    )
    return fig


def build_capacity_factor_annual_from_availability_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build annual capacity factor from AvailabilityFactor and YearSplit.

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, powerplant mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (as_percent, technology_aggregation, technology_weight_source)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    as_percent = merged_opts.get("as_percent", True)

    try:
        af = pd.read_sql_query('SELECT t, l, y, val FROM "AvailabilityFactor"', conn)
        year_split = pd.read_sql_query('SELECT l, y, val FROM "YearSplit"', conn)
    except Exception as exc:
        print(f"Failed to load AvailabilityFactor/YearSplit: {exc}")
        return None

    if af.empty or year_split.empty:
        return None

    af = af.rename(columns={"t": "TECHNOLOGY", "l": "TIMESLICE", "y": "YEAR", "val": "VALUE"})
    year_split = year_split.rename(columns={"l": "TIMESLICE", "y": "YEAR", "val": "SHARE"})
    af["YEAR"] = pd.to_numeric(af["YEAR"], errors="coerce")
    af["VALUE"] = pd.to_numeric(af["VALUE"], errors="coerce")
    year_split["YEAR"] = pd.to_numeric(year_split["YEAR"], errors="coerce")
    year_split["SHARE"] = pd.to_numeric(year_split["SHARE"], errors="coerce")
    af = af.dropna(subset=["YEAR", "VALUE"])
    year_split = year_split.dropna(subset=["YEAR", "SHARE"])

    df = af.merge(year_split, on=["TIMESLICE", "YEAR"], how="inner")
    if df.empty:
        return None
    df["WEIGHTED"] = df["VALUE"] * df["SHARE"]
    df = df.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["WEIGHTED"].sum()
    aggregation_method = str(merged_opts.get("technology_aggregation", "mean")).lower()
    if aggregation_method in {"simple_mean", "mean"}:
        aggregation_method = "mean"
    if aggregation_method == "weighted_mean":
        weight_source = str(merged_opts.get("technology_weight_source", "activity")).lower()
        weight_query_map = {
            "activity": 'SELECT t, y, val FROM "vtotaltechnologyannualactivity"',
            "production": 'SELECT t, y, val FROM "vproductionbytechnologyannual"',
            "capacity": 'SELECT t, y, val FROM "vtotalcapacityannual"',
        }
        weight_query = weight_query_map.get(weight_source)
        if weight_query is None:
            print(f"Unknown technology_weight_source '{weight_source}', falling back to simple mean.")
            aggregation_method = "mean"
        try:
            if aggregation_method == "weighted_mean":
                activity = pd.read_sql_query(weight_query, conn)
        except Exception as exc:
            print(f"Failed to load weights for weighted availability aggregation: {exc}")
            activity = pd.DataFrame()
        if activity.empty:
            aggregation_method = "mean"
        else:
            activity = activity.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "ACTIVITY"})
            activity["YEAR"] = pd.to_numeric(activity["YEAR"], errors="coerce")
            activity["ACTIVITY"] = pd.to_numeric(activity["ACTIVITY"], errors="coerce")
            activity = activity.dropna(subset=["YEAR", "ACTIVITY"])
            activity = activity.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["ACTIVITY"].sum()
            df = df.merge(activity, on=["TECHNOLOGY", "YEAR"], how="left")
            df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, merged_opts)
            df["WEIGHTED_VALUE"] = df["WEIGHTED"] * df["ACTIVITY"].fillna(0.0)
            df = df.groupby(["TECHNOLOGY", "YEAR"], as_index=False).agg(
                WEIGHTED_SUM=("WEIGHTED_VALUE", "sum"),
                WEIGHT_SUM=("ACTIVITY", "sum"),
                MEAN_VALUE=("WEIGHTED", "mean"),
            )
            df["VALUE"] = df["WEIGHTED_SUM"] / df["WEIGHT_SUM"]
            df.loc[df["WEIGHT_SUM"] <= 0, "VALUE"] = df.loc[df["WEIGHT_SUM"] <= 0, "MEAN_VALUE"]
            df = df[["TECHNOLOGY", "YEAR", "VALUE"]]
    if aggregation_method != "weighted_mean":
        if aggregation_method not in {"mean", "median", "min", "max"}:
            aggregation_method = "mean"
        df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, merged_opts)
        df = df.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["WEIGHTED"].agg(aggregation_method)
        df = df.rename(columns={"WEIGHTED": "VALUE"})
    if as_percent:
        df["VALUE"] = df["VALUE"] * 100.0

    df = apply_common_cleaning(df, "TECHNOLOGY", merged_opts)
    update_missing_colors(df["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Capacity factor (from availability) by technology"
    return px.line(df, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)


def build_costs_by_technology_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build fixed vs variable costs by technology (annual).

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, powerplant mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (fixed_label, variable_label, variable_line_dash)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}

    try:
        fixed = pd.read_sql_query('SELECT t, y, val FROM "FixedCost"', conn)
        variable = pd.read_sql_query('SELECT t, y, val FROM "VariableCost"', conn)
    except Exception as exc:
        print(f"Failed to load cost tables: {exc}")
        return None

    if fixed.empty and variable.empty:
        return None

    if not fixed.empty:
        fixed = fixed.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
        fixed["TECHNOLOGY"] = map_name(fixed["TECHNOLOGY"], powerplant_mapping, merged_opts)
        fixed["YEAR"] = pd.to_numeric(fixed["YEAR"], errors="coerce")
        fixed["VALUE"] = pd.to_numeric(fixed["VALUE"], errors="coerce")
        fixed = fixed.dropna(subset=["YEAR", "VALUE"])
        fixed = fixed.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()

    if not variable.empty:
        variable = variable.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
        variable["TECHNOLOGY"] = map_name(variable["TECHNOLOGY"], powerplant_mapping, merged_opts)
        variable["YEAR"] = pd.to_numeric(variable["YEAR"], errors="coerce")
        variable["VALUE"] = pd.to_numeric(variable["VALUE"], errors="coerce")
        variable = variable.dropna(subset=["YEAR", "VALUE"])
        variable = variable.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()

    combined = pd.concat([fixed.assign(COST_TYPE="Fixed"), variable.assign(COST_TYPE="Variable")], ignore_index=True)
    if combined.empty:
        return None
    combined = apply_common_cleaning(combined, "TECHNOLOGY", merged_opts)
    update_missing_colors(combined["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())

    fixed_label = merged_opts.get("fixed_label", "Fixed cost")
    variable_label = merged_opts.get("variable_label", "Variable cost")
    variable_dash = merged_opts.get("variable_line_dash", "dot")

    fig = go.Figure()
    for tech in sorted(combined["TECHNOLOGY"].unique()):
        tech_color = tech_colors.get(tech)
        fixed_part = combined[(combined["TECHNOLOGY"] == tech) & (combined["COST_TYPE"] == "Fixed")]
        variable_part = combined[(combined["TECHNOLOGY"] == tech) & (combined["COST_TYPE"] == "Variable")]
        if not fixed_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=fixed_part["YEAR"],
                    y=fixed_part["VALUE"],
                    mode="lines",
                    name=f"{tech} ({fixed_label})",
                    line=dict(color=tech_color, dash="solid"),
                )
            )
        if not variable_part.empty:
            fig.add_trace(
                go.Scatter(
                    x=variable_part["YEAR"],
                    y=variable_part["VALUE"],
                    mode="lines",
                    name=f"{tech} ({variable_label})",
                    line=dict(color=tech_color, dash=variable_dash),
                )
            )
    fig.update_layout(title="Fixed and variable costs by technology", legend_title_text="Technology")
    return fig


def build_cost_per_production_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build total cost per unit of production by technology.

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, powerplant mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (production_unit_multiplier)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    tech_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    production_unit_multiplier = float(merged_opts.get("production_unit_multiplier", 1.0))

    try:
        fixed = pd.read_sql_query('SELECT t, y, val FROM "FixedCost"', conn)
        variable = pd.read_sql_query('SELECT t, y, val FROM "VariableCost"', conn)
        capacity = pd.read_sql_query('SELECT t, y, val FROM "vtotalcapacityannual"', conn)
        activity = pd.read_sql_query('SELECT t, y, val FROM "vtotaltechnologyannualactivity"', conn)
        production = pd.read_sql_query('SELECT t, y, val FROM "vproductionbytechnologyannual"', conn)
    except Exception as exc:
        print(f"Failed to load cost/production inputs: {exc}")
        return None

    if production.empty:
        return None

    fixed = fixed.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    variable = variable.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    capacity = capacity.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    activity = activity.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
    production = production.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "PRODUCTION"})

    for df in (fixed, variable, capacity, activity, production):
        df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, merged_opts)
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df[df.columns.difference(["TECHNOLOGY", "YEAR"])] = df[df.columns.difference(["TECHNOLOGY", "YEAR"])].apply(
            pd.to_numeric, errors="coerce"
        )
        df.dropna(subset=["YEAR"], inplace=True)

    fixed = fixed.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    variable = variable.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    capacity = capacity.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    activity = activity.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["VALUE"].sum()
    production = production.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["PRODUCTION"].sum()

    if fixed.empty and variable.empty:
        return None

    fixed_total = fixed.merge(capacity, on=["TECHNOLOGY", "YEAR"], how="left", suffixes=("_FIXED", "_CAP"))
    if "VALUE_CAP" in fixed_total.columns:
        fixed_total["FIXED_TOTAL"] = fixed_total["VALUE_FIXED"] * fixed_total["VALUE_CAP"]
    else:
        fixed_total["FIXED_TOTAL"] = fixed_total["VALUE_FIXED"]
    fixed_total = fixed_total[["TECHNOLOGY", "YEAR", "FIXED_TOTAL"]]

    variable_total = variable.merge(activity, on=["TECHNOLOGY", "YEAR"], how="left", suffixes=("_VAR", "_ACT"))
    if "VALUE_ACT" in variable_total.columns:
        variable_total["VARIABLE_TOTAL"] = variable_total["VALUE_VAR"] * variable_total["VALUE_ACT"]
    else:
        variable_total["VARIABLE_TOTAL"] = variable_total["VALUE_VAR"]
    variable_total = variable_total[["TECHNOLOGY", "YEAR", "VARIABLE_TOTAL"]]

    totals = fixed_total.merge(variable_total, on=["TECHNOLOGY", "YEAR"], how="outer")
    totals = totals.merge(production, on=["TECHNOLOGY", "YEAR"], how="inner")
    if totals.empty:
        return None

    totals["FIXED_TOTAL"] = totals["FIXED_TOTAL"].fillna(0.0)
    totals["VARIABLE_TOTAL"] = totals["VARIABLE_TOTAL"].fillna(0.0)
    totals["PRODUCTION"] = totals["PRODUCTION"] * production_unit_multiplier
    totals = totals[totals["PRODUCTION"] > 0]
    if totals.empty:
        return None
    totals["VALUE"] = (totals["FIXED_TOTAL"] + totals["VARIABLE_TOTAL"]) / totals["PRODUCTION"]

    totals = apply_common_cleaning(totals, "TECHNOLOGY", merged_opts)
    update_missing_colors(totals["TECHNOLOGY"].unique(), tech_colors, missing_colors or set())
    title = "Total cost per unit of production by technology"
    fig = px.line(totals, x="YEAR", y="VALUE", color="TECHNOLOGY", title=title, color_discrete_map=tech_colors)
    yaxis_type = str(merged_opts.get("yaxis_type", "log")).lower()
    if yaxis_type == "log":
        fig.update_yaxes(type="log")
        fig.update_yaxes(title_text="Value (log scale)")
    return fig


def build_generation_timeslice_fig(
    conn: sqlite3.Connection,
    mapping: dict,
    missing_colors: set[str] | None = None,
    opts: dict | None = None,
):
    """
    Build average generation by timeslice (GW) for a given year or year-aggregate.

    Inputs:
        - conn: sqlite connection to the results database
        - mapping: plotting mappings (colors, fuel mapping)
        - missing_colors: set to collect missing color labels
        - opts: optional overrides (year, year_agg, hours_per_year, energy_unit_multiplier, timeslice_order,
          include_capacity_timeslice, capacity_timeslice_label)

    Outputs:
        - plotly figure (or None when inputs are missing)

    Side effects:
        - prints debug info when data loads fail
        - updates missing_colors set if provided
    """
    fuel_colors = mapping.get("colors", {})
    powerplant_mapping = mapping.get("powerplant")
    merged_opts = opts or {}
    hours_per_year = float(merged_opts.get("hours_per_year", 8760))
    energy_unit_multiplier = float(merged_opts.get("energy_unit_multiplier", 0.0036))
    year_opt = merged_opts.get("year", "latest")
    year_agg = merged_opts.get("year_agg")
    timeslice_order = merged_opts.get("timeslice_order")
    include_capacity_timeslice = merged_opts.get("include_capacity_timeslice", False)
    capacity_timeslice_label = merged_opts.get("capacity_timeslice_label", "CAPACITY")
    cap_include_substrings = merged_opts.get("capacity_include_tech_substrings", list(DEFAULT_CAPACITY_TECH_INCLUDE_SUBSTRINGS))
    cap_drop_substrings = merged_opts.get("capacity_drop_tech_substrings") or []
    default_cap_drop_substrings = ["_INF", "INF_"]
    cap_drop_substrings = list(cap_drop_substrings) + default_cap_drop_substrings

    gen_include_substrings = merged_opts.get(
        "generation_include_tech_substrings",
        list(DEFAULT_CAPACITY_TECH_INCLUDE_SUBSTRINGS),
    )
    gen_drop_substrings = merged_opts.get("generation_drop_tech_substrings")
    if gen_drop_substrings is None:
        gen_drop_substrings = cap_drop_substrings

    try:
        production = pd.read_sql_query('SELECT t, y, val FROM "vproductionbytechnologyannual"', conn)
        availability = pd.read_sql_query('SELECT t, l, y, val FROM "AvailabilityFactor"', conn)
        year_split = pd.read_sql_query('SELECT l, y, val FROM "YearSplit"', conn)
    except Exception as exc:
        print(f"Failed to load timeslice generation inputs: {exc}")
        return None

    if production.empty or availability.empty or year_split.empty:
        return None

    production = production.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "ANNUAL_PJ"})
    availability = availability.rename(columns={"t": "TECHNOLOGY", "l": "TIMESLICE", "y": "YEAR", "val": "AVAIL"})
    year_split = year_split.rename(columns={"l": "TIMESLICE", "y": "YEAR", "val": "SHARE"})
    production["TECHNOLOGY_RAW"] = production["TECHNOLOGY"].astype(str)
    availability["TECHNOLOGY_RAW"] = availability["TECHNOLOGY"].astype(str)
    if gen_include_substrings:
        pattern = "|".join(re.escape(item) for item in gen_include_substrings if item)
        if pattern:
            production = production[production["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
            availability = availability[availability["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
    if gen_drop_substrings:
        pattern = "|".join(re.escape(item) for item in gen_drop_substrings if item)
        if pattern:
            production = production[~production["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
            availability = availability[~availability["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
    production["YEAR"] = pd.to_numeric(production["YEAR"], errors="coerce")
    production["ANNUAL_PJ"] = pd.to_numeric(production["ANNUAL_PJ"], errors="coerce")
    availability["YEAR"] = pd.to_numeric(availability["YEAR"], errors="coerce")
    availability["AVAIL"] = pd.to_numeric(availability["AVAIL"], errors="coerce")
    year_split["YEAR"] = pd.to_numeric(year_split["YEAR"], errors="coerce")
    year_split["SHARE"] = pd.to_numeric(year_split["SHARE"], errors="coerce")
    production = production.dropna(subset=["YEAR", "ANNUAL_PJ"])
    availability = availability.dropna(subset=["YEAR", "AVAIL"])
    year_split = year_split.dropna(subset=["YEAR", "SHARE"])

    year_split["HOURS"] = year_split["SHARE"] * hours_per_year
    weights = availability.merge(year_split[["TIMESLICE", "YEAR", "SHARE", "HOURS"]], on=["TIMESLICE", "YEAR"], how="inner")
    if weights.empty:
        return None
    weights["WEIGHT"] = weights["AVAIL"] * weights["SHARE"]
    weights["WEIGHT_SUM"] = weights.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["WEIGHT"].transform("sum")
    weights["SHARE_SUM"] = weights.groupby(["TECHNOLOGY", "YEAR"], as_index=False)["SHARE"].transform("sum")
    weights["WEIGHT_NORM"] = weights["WEIGHT"] / weights["WEIGHT_SUM"]
    weights.loc[weights["WEIGHT_SUM"] <= 0, "WEIGHT_NORM"] = (
        weights.loc[weights["WEIGHT_SUM"] <= 0, "SHARE"] / weights.loc[weights["WEIGHT_SUM"] <= 0, "SHARE_SUM"]
    )
    gen = weights.merge(production[["TECHNOLOGY", "YEAR", "ANNUAL_PJ"]], on=["TECHNOLOGY", "YEAR"], how="inner")
    if gen.empty:
        return None
    gen["TIMESLICE_PJ"] = gen["ANNUAL_PJ"] * gen["WEIGHT_NORM"]
    gen["TECHNOLOGY"] = map_name(gen["TECHNOLOGY"], powerplant_mapping, merged_opts)
    gen = gen.groupby(["TECHNOLOGY", "TIMESLICE", "YEAR"], as_index=False).agg(
        TIMESLICE_PJ=("TIMESLICE_PJ", "sum"),
        HOURS=("HOURS", "first"),
    )
    gen["VALUE"] = gen["TIMESLICE_PJ"] / (energy_unit_multiplier * gen["HOURS"])

    if year_agg:
        agg_func = str(year_agg).lower()
        if agg_func not in {"mean", "median", "min", "max"}:
            agg_func = "mean"
        gen = gen.groupby(["TECHNOLOGY", "TIMESLICE"], as_index=False)["VALUE"].agg(agg_func)
        year_label = f"{agg_func.title()} over years"
    else:
        if isinstance(year_opt, (list, tuple, set)):
            gen = gen[gen["YEAR"].isin([float(y) for y in year_opt])]
            year_label = f"Years {', '.join(str(y) for y in year_opt)}"
        else:
            year_tag = str(year_opt).lower() if year_opt is not None else "latest"
            if year_tag in {"latest", "max"}:
                year_val = gen["YEAR"].max()
            elif year_tag in {"earliest", "min"}:
                year_val = gen["YEAR"].min()
            else:
                year_val = float(year_opt)
            gen = gen[gen["YEAR"] == year_val]
            year_label = f"Year {int(year_val)}" if pd.notna(year_val) else "Selected year"
        gen = gen.groupby(["TECHNOLOGY", "TIMESLICE"], as_index=False)["VALUE"].mean()

    cap = pd.DataFrame()
    if include_capacity_timeslice:
        try:
            cap = pd.read_sql_query('SELECT t, y, val FROM "vtotalcapacityannual"', conn)
        except Exception as exc:
            print(f"Failed to load capacity for timeslice chart: {exc}")
            cap = pd.DataFrame()

        if not cap.empty:
            cap = cap.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "VALUE"})
            cap["TECHNOLOGY_RAW"] = cap["TECHNOLOGY"].astype(str)
            if cap_include_substrings:
                pattern = "|".join(re.escape(item) for item in cap_include_substrings if item)
                if pattern:
                    cap = cap[cap["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
            if cap_drop_substrings:
                pattern = "|".join(re.escape(item) for item in cap_drop_substrings if item)
                if pattern:
                    cap = cap[~cap["TECHNOLOGY_RAW"].str.contains(pattern, na=False)]
            cap["TECHNOLOGY"] = map_name(cap["TECHNOLOGY"], mapping.get("powerplant"), merged_opts)
            # Align capacity categories with generation fuel labels.
            allowed_categories = {
                "Coal",
                "Gas",
                "Oil",
                "Nuclear",
                "Hydro",
                "Geothermal",
                "Solar",
                "Wind",
                "Biomass",
                "Other",
                "Imports",
            }
            unmapped = sorted(set(cap["TECHNOLOGY"].unique()) - allowed_categories)
            if unmapped:
                raise ValueError(
                    "Unmapped capacity categories for timeslice chart: "
                    + ", ".join(str(item) for item in unmapped)
                )
            cap["TECHNOLOGY"] = cap["TECHNOLOGY"]
            cap["YEAR"] = pd.to_numeric(cap["YEAR"], errors="coerce")
            cap["VALUE"] = pd.to_numeric(cap["VALUE"], errors="coerce")
            cap = cap.dropna(subset=["YEAR", "VALUE"])
            if year_agg:
                agg_func = str(year_agg).lower()
                if agg_func not in {"mean", "median", "min", "max"}:
                    agg_func = "mean"
                cap = cap.groupby(["TECHNOLOGY"], as_index=False)["VALUE"].agg(agg_func)
            else:
                if isinstance(year_opt, (list, tuple, set)):
                    cap = cap[cap["YEAR"].isin([float(y) for y in year_opt])]
                else:
                    year_tag = str(year_opt).lower() if year_opt is not None else "latest"
                    if year_tag in {"latest", "max"}:
                        year_val = cap["YEAR"].max()
                    elif year_tag in {"earliest", "min"}:
                        year_val = cap["YEAR"].min()
                    else:
                        year_val = float(year_opt)
                    cap = cap[cap["YEAR"] == year_val]
                cap = cap.groupby(["TECHNOLOGY"], as_index=False)["VALUE"].sum()
            cap = cap.rename(columns={"TECHNOLOGY": "FUEL"})
            cap["TIMESLICE"] = capacity_timeslice_label
            cap = cap[["FUEL", "TIMESLICE", "VALUE"]]

    if gen.empty:
        return None

    gen = apply_common_cleaning(gen, "TECHNOLOGY", merged_opts)
    gen = gen.rename(columns={"TECHNOLOGY": "FUEL"})
    allowed_fuels = set(gen["FUEL"].unique())
    if include_capacity_timeslice and not cap.empty:
        overlap = set(cap["FUEL"].unique()) & allowed_fuels
        if overlap:
            cap = cap[cap["FUEL"].isin(allowed_fuels)]
        else:
            print("Warning: capacity categories do not overlap generation fuels; keeping all capacity categories.")
    update_missing_colors(gen["FUEL"].unique(), fuel_colors, missing_colors or set())
    title = merged_opts.get("title") or f"Average generation by timeslice ({year_label})"
    barmode = merged_opts.get("barmode", "relative")
    if timeslice_order and include_capacity_timeslice and capacity_timeslice_label not in timeslice_order:
        timeslice_order = list(timeslice_order) + [capacity_timeslice_label]

    if include_capacity_timeslice and not cap.empty:
        fig = go.Figure()
        for fuel in sorted(gen["FUEL"].unique()):
            subset = gen[gen["FUEL"] == fuel]
            fig.add_trace(
                go.Bar(
                    x=subset["TIMESLICE"],
                    y=subset["VALUE"],
                    name=str(fuel),
                    marker_color=fuel_colors.get(fuel),
                )
            )
        for fuel in sorted(cap["FUEL"].unique()):
            subset = cap[cap["FUEL"] == fuel]
            fig.add_trace(
                go.Bar(
                    x=subset["TIMESLICE"],
                    y=subset["VALUE"],
                    name=f"{fuel} (capacity)",
                    marker_color=fuel_colors.get(fuel),
                    marker_line=dict(color="#222", width=2),
                    opacity=0.7,
                )
            )
        fig.update_layout(title=title, barmode=barmode, legend_title_text="Fuel")
        if timeslice_order:
            fig.update_xaxes(categoryorder="array", categoryarray=timeslice_order)
        return fig

    fig = px.bar(
        gen,
        x="TIMESLICE",
        y="VALUE",
        color="FUEL",
        title=title,
        color_discrete_map=fuel_colors,
        category_orders={"TIMESLICE": timeslice_order} if timeslice_order else None,
        barmode=barmode,
    )
    return fig


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
    merged_opts = opts or {}
    use_emission_activity_ratio = merged_opts.get("use_emission_activity_ratio", True)
    use_emissions_factors_csv = merged_opts.get("use_emissions_factors_csv", False)
    emissions_factor_csv = merged_opts.get("emissions_factor_csv", "config/9th_edition_emission_factors.csv")
    df = None

    if use_emissions_factors_csv:
        factors_path = Path(emissions_factor_csv)
        if not factors_path.is_absolute():
            factors_path = Path(__file__).resolve().parents[2] / factors_path
        try:
            factors_df = pd.read_csv(factors_path)
            if "fuel_code" in factors_df.columns and "Emissions factor (MT/PJ)" in factors_df.columns:
                factors_df = factors_df[["fuel_code", "Emissions factor (MT/PJ)"]].copy()
                factors_df["fuel_code"] = factors_df["fuel_code"].astype(str).str.strip()
                factors_df["Emissions factor (MT/PJ)"] = pd.to_numeric(
                    factors_df["Emissions factor (MT/PJ)"], errors="coerce"
                )
                factors_df = factors_df.dropna(subset=["fuel_code", "Emissions factor (MT/PJ)"])
            else:
                factors_df = pd.DataFrame()
        except Exception as exc:
            print(f"Failed to load emissions factors CSV '{factors_path}': {exc}")
            factors_df = pd.DataFrame()

        try:
            iar = pd.read_sql_query(
                'SELECT t, f, m, y, val FROM "InputActivityRatio"',
                conn,
            )
        except Exception as exc:
            print(f"Failed to load InputActivityRatio for emissions factors: {exc}")
            iar = pd.DataFrame()

        try:
            activity = pd.read_sql_query(
                'SELECT t, y, val FROM "vtotaltechnologyannualactivity"',
                conn,
            )
        except Exception as exc:
            print(f"Failed to load activity for emissions factors: {exc}")
            activity = pd.DataFrame()

        if not factors_df.empty and not iar.empty and not activity.empty:
            factors_map = dict(
                zip(factors_df["fuel_code"], factors_df["Emissions factor (MT/PJ)"])
            )
            iar["f"] = iar["f"].astype(str).str.strip()
            iar["f"] = iar["f"].map(EMISSIONS_FACTOR_FUEL_MAP).fillna(iar["f"])
            iar["val"] = pd.to_numeric(iar["val"], errors="coerce")
            iar = iar.dropna(subset=["val"])
            iar["factor"] = iar["f"].map(factors_map)
            calc_df = iar.dropna(subset=["factor"]).copy()
            if not calc_df.empty:
                calc_df["val"] = calc_df["val"] * calc_df["factor"]
                ear = (
                    calc_df.groupby(["t", "y"], dropna=False)["val"]
                    .sum()
                    .reset_index()
                )
                activity = activity.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "ACTIVITY"})
                ear = ear.rename(columns={"t": "TECHNOLOGY", "y": "YEAR", "val": "EAR"})
                activity["YEAR"] = pd.to_numeric(activity["YEAR"], errors="coerce")
                activity["ACTIVITY"] = pd.to_numeric(activity["ACTIVITY"], errors="coerce")
                ear["YEAR"] = pd.to_numeric(ear["YEAR"], errors="coerce")
                ear["EAR"] = pd.to_numeric(ear["EAR"], errors="coerce")
                activity = activity.dropna(subset=["YEAR", "ACTIVITY"])
                ear = ear.dropna(subset=["YEAR", "EAR"])
                df = activity.merge(ear, on=["TECHNOLOGY", "YEAR"], how="inner")
                if not df.empty:
                    df["VALUE"] = df["ACTIVITY"] * df["EAR"]
                    df = df[["TECHNOLOGY", "YEAR", "VALUE"]]

    if df is None or df.empty:
        try:
            df = pd.read_sql_query('SELECT t, y, val FROM "AnnualTechnologyEmission"', conn)
        except Exception:
            df = None

    if df is None or df.empty:
        try:
            df = pd.read_sql_query('SELECT t, y, val FROM "vannualtechnologyemission"', conn)
        except Exception:
            df = None

    if df is None or df.empty:
        try:
            df = pd.read_sql_query('SELECT t, y, val FROM "vannualtechnologyemissionbymode"', conn)
            if not df.empty:
                df = df.groupby(["t", "y"], as_index=False)["val"].sum()
        except Exception:
            df = None

    if (df is None or df.empty) and use_emission_activity_ratio and not use_emissions_factors_csv:
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

    if df is None or df.empty:
        return None

    df["TECHNOLOGY"] = map_name(df["TECHNOLOGY"], powerplant_mapping, merged_opts)
    df = apply_common_cleaning(df, "TECHNOLOGY", merged_opts)
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
    "capacity_factor": "%",
    "capacity_factor_availability": "%",
    "capacity_factor_timeslice": "%",
    "costs_by_technology": "USD",
    "cost_per_production": "USD per unit",
    "generation_timeslice": "GW",
    "input_use": "PJ",
    "emissions": "MtCO2",
    "new_capacity": "GW",
    "total_cost": "USD",
}

FUNCTION_BUILDERS = {
    "generation": build_generation_fig,
    "capacity": build_capacity_fig,
    "capacity_factor": build_capacity_factor_fig,
    "capacity_factor_availability": build_capacity_factor_annual_from_availability_fig,
    "capacity_factor_timeslice": build_capacity_factor_timeslice_fig,
    "costs_by_technology": build_costs_by_technology_fig,
    "cost_per_production": build_cost_per_production_fig,
    "generation_timeslice": build_generation_timeslice_fig,
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
                try:
                    yaxis_type = getattr(fig.layout.yaxis, "type", None)
                    yaxis_title = getattr(fig.layout.yaxis.title, "text", "") or ""
                    if yaxis_type == "log" and "log" not in yaxis_title.lower():
                        fig.update_yaxes(title_text=f"{yaxis_title} (log scale)".strip())
                except Exception:
                    pass
                note = (opts or {}).get("note")
                if isinstance(note, str) and note.strip():
                    try:
                        note = note.format(**(opts or {}))
                    except Exception:
                        note = note.strip()
                    fig.update_layout(meta={"note": note.strip()})
                figs.append(fig)
        except Exception as exc:
            print(f"Failed to build plot '{key}': {exc}")
            continue

    conn.close()
    return figs
