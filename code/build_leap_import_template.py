# Build a LEAP import spreadsheet by mapping NEMO DB values to LEAP Branch/Variable rows.
from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import pandas as pd

# -------------------------------------------------------------------
# User-configurable defaults
# -------------------------------------------------------------------
DEFAULT_SCENARIO = "Reference"
DEFAULT_REGION = None  # If None, we mirror the scenario string.
OUTPUT_PATH = Path("data/leap_import_template.xlsx")
NEMO_DB_PATH = Path("data/nemo.sqlite")  # Set to None to skip autofill.
AUTO_FILL_FROM_DB = True
DEDUPLICATE_ROWS = True  # Drop duplicate Branch/Variable pairs before filling.

# Region/tech mapping used when querying the NEMO DB.
REGION_FILTER = "20_USA"  # Change if you want another region or set to None to pull all.
TECH_MAP = {
    "Coal": "POW_Coal_PP",  # Update to your coal tech code if different.
}

# -------------------------------------------------------------------
# Base rows (Branch Path, Variable). Extend this list to add more items.
# -------------------------------------------------------------------
ROWS: list[tuple[str, str]] = [
    ("Transformation\\Electricity Generation", "Planning Reserve Margin"),
    ("Transformation\\Electricity Generation", "Peak Load Ratio"),
    ("Transformation\\Electricity Generation", "Module Costs"),
    ("Transformation\\Electricity Generation", "Planning Reserve Margin"),
    ("Transformation\\Electricity Generation", "Peak Load Ratio"),
    ("Transformation\\Electricity Generation", "Renewable Target"),
    ("Transformation\\Electricity Generation", "Module Costs"),
    ("Transformation\\Electricity Generation", "Optimize"),
    ("Transformation\\Electricity Generation", "Use Addition Size"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Shortfall Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Surplus Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Usage Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Import Target"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Export Target"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Price"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Share"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Price"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Dispatch Rule"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Lifetime"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Interest Rate"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Exogenous Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Availability"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Utilization"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capacity Credit"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Dispatchable"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Endogenous Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Merit Order"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "First Simulation Year"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Charge"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Starting Charge"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Full Load Hours"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Annual Storage Carryover"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Seasonal Storage Carryover"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Hourly Storage Carryover"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Process Efficiency"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Variable OM Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capital Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Stranded Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Salvage Value"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Fixed OM Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Optimized New Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Renewable Qualified"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Lifetime"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Production"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Production"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Share of Production"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Interest Rate"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Capacity Addition"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Capacity Addition"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Exogenous Capacity"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Availability"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Utilization"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capacity Credit"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Charge"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Full Load Hours"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Process Efficiency"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Variable OM Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capital Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Stranded Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Salvage Value"),
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Fixed OM Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Feedstock Fuel Share"),
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Fuel Cost"),
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Feedstock Fuel Share"),
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Fuel Cost"),
]

# LEAP columns and level columns
LEVEL_COLS = [f"Level {i}" for i in range(1, 9)]
OUTPUT_COLUMNS = ["Scenario", "Region", "Scale", "Units", "Per", "Expression", "Variable"] + LEVEL_COLS + ["Notes"]

# Variable metadata for Scale/Units/Per defaults (extend as needed)
VARIABLE_META = {
    "Planning Reserve Margin": {"Scale": "", "Units": "", "Per": ""},
    "Interest Rate": {"Scale": "", "Units": "%", "Per": ""},
    "Variable OM Cost": {"Scale": "", "Units": "USD/MWh", "Per": ""},
    "Fixed OM Cost": {"Scale": "", "Units": "USD/MW", "Per": ""},
    "Capital Cost": {"Scale": "", "Units": "USD/MW", "Per": ""},
    "Maximum Capacity": {"Scale": "", "Units": "MW", "Per": ""},
    "Minimum Capacity": {"Scale": "", "Units": "MW", "Per": ""},
    "Maximum Capacity Addition": {"Scale": "", "Units": "MW", "Per": ""},
    "Minimum Capacity Addition": {"Scale": "", "Units": "MW", "Per": ""},
    "Exogenous Capacity": {"Scale": "", "Units": "MW", "Per": ""},
    "Maximum Availability": {"Scale": "", "Units": "fraction", "Per": ""},
    "Minimum Utilization": {"Scale": "", "Units": "fraction", "Per": ""},
    "Capacity Credit": {"Scale": "", "Units": "fraction", "Per": ""},
    "Full Load Hours": {"Scale": "", "Units": "hours", "Per": ""},
    "Maximum Production": {"Scale": "", "Units": "GWh", "Per": ""},
    "Minimum Production": {"Scale": "", "Units": "GWh", "Per": ""},
    "Minimum Share of Production": {"Scale": "", "Units": "fraction", "Per": ""},
    "Lifetime": {"Scale": "", "Units": "years", "Per": ""},
    "Renewable Qualified": {"Scale": "", "Units": "flag", "Per": ""},
}

# -------------------------------------------------------------------
# Mapping definitions
# Each entry maps (Branch Path, Variable) -> config:
#   table: NEMO table name
#   select: {OutCol: SQL_expr} with aliases (Year/Value/Units/Notes supported)
#   filters: dict of column->value
#   group_by: list of columns to group on (for aggregates)
#   transform: string key for special handling
# -------------------------------------------------------------------
MAPPINGS: dict[tuple[str, str], dict | None] = {
    # System-level
    ("Transformation\\Electricity Generation", "Planning Reserve Margin"): {
        "table": "ReserveMargin",
        "select": {
            "Year": "y",
            "Value": "val",
            "Units": "''",
            "Notes": "r || '|' || f",
        },
        "filters": {"r": REGION_FILTER},
    },
    # Coal process mappings
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Lifetime"): {
        "table": "OperationalLife",
        "select": {"Year": "''", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Interest Rate"): {
        "table": "InterestRateTechnology",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Exogenous Capacity"): {
        "table": "ResidualCapacity",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Availability"): {
        "table": "AvailabilityFactor",
        "select": {"Year": "y", "Value": "MAX(val)"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
        "group_by": ["y"],
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Utilization"): {
        "table": "MinimumUtilization",
        "select": {"Year": "y", "Value": "MAX(val)"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
        "group_by": ["y"],
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capacity Credit"): {
        "table": "ReserveMarginTagTechnology",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"], "f": "ALL"},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Full Load Hours"): {
        "transform": "full_load_hours",
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Variable OM Cost"): {
        "table": "VariableCost",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Capital Cost"): {
        "table": "CapitalCost",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Fixed OM Cost"): {
        "table": "FixedCost",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Capacity"): {
        "table": "TotalAnnualMaxCapacity",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Capacity"): {
        "table": "TotalAnnualMinCapacity",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Capacity Addition"): {
        "table": "TotalAnnualMaxCapacityInvestment",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Capacity Addition"): {
        "table": "TotalAnnualMinCapacityInvestment",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Maximum Production"): {
        "table": "TotalTechnologyAnnualActivityUpperLimit",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Production"): {
        "table": "TotalTechnologyAnnualActivityLowerLimit",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Share of Production"): {
        "table": "MinShareProduction",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"], "f": "ALL"},
    },
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Renewable Qualified"): {
        "table": "RETagTechnology",
        "select": {"Year": "y", "Value": "val"},
        "filters": {"r": REGION_FILTER, "t": TECH_MAP["Coal"]},
    },
    # Placeholders (no NEMO mapping yet -> leave blank)
    ("Transformation\\Electricity Generation", "Peak Load Ratio"): None,
    ("Transformation\\Electricity Generation", "Module Costs"): None,
    ("Transformation\\Electricity Generation", "Optimize"): None,
    ("Transformation\\Electricity Generation", "Use Addition Size"): None,
    ("Transformation\\Electricity Generation", "Renewable Target"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Shortfall Rule"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Surplus Rule"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Usage Rule"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Import Target"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Export Target"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Price"): None,
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Share"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Dispatch Rule"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Dispatchable"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Endogenous Capacity"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Merit Order"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "First Simulation Year"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Minimum Charge"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Starting Charge"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Annual Storage Carryover"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Seasonal Storage Carryover"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Hourly Storage Carryover"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Optimized New Capacity"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Stranded Cost"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal", "Salvage Value"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Feedstock Fuel Share"): None,
    ("Transformation\\Electricity Generation\\Processes\\Coal\\Feedstock Fuels\\Coal", "Fuel Cost"): None,
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def path_to_levels(branch: str, max_levels: int = 8) -> list[str]:
    parts = branch.split("\\")
    parts = parts[:max_levels] + [""] * max(0, max_levels - len(parts))
    return parts


def build_dataframe(rows: list[tuple[str, str]], deduplicate: bool) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Branch Path", "Variable"])
    if deduplicate:
        df = df.drop_duplicates(subset=["Branch Path", "Variable"])
    return df


def _decode_val(v):
    if isinstance(v, (bytes, bytearray)):
        try:
            return int.from_bytes(v, byteorder="little", signed=True)
        except Exception:
            return None
    return v


def _run_simple_mapping(conn: sqlite3.Connection, mapping: dict) -> pd.DataFrame:
    table = mapping.get("table")
    select_map: dict = mapping.get("select") or {}
    filters: dict = mapping.get("filters") or {}
    group_by = mapping.get("group_by") or []

    if not table or not select_map:
        return pd.DataFrame()

    select_parts = [f"{expr} AS {alias}" for alias, expr in select_map.items()]
    where = []
    params = []
    for col, val in filters.items():
        if val is None:
            continue
        where.append(f"{col} = ?")
        params.append(val)
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    group_sql = f" GROUP BY {', '.join(group_by)}" if group_by else ""
    sql = f'SELECT {", ".join(select_parts)} FROM "{table}"{where_sql}{group_sql}'

    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()
    for col in ("Value", "Year"):
        if col in df.columns:
            df[col] = df[col].apply(_decode_val)
    for col in ["Year", "Value", "Units", "Notes"]:
        if col not in df.columns:
            df[col] = ""
    return df[["Year", "Value", "Units", "Notes"]]


def _transform_full_load_hours(conn: sqlite3.Connection, filters: dict) -> pd.DataFrame:
    r = filters.get("r")
    t = filters.get("t")
    try:
        sql = """
        SELECT af.y AS Year,
               SUM(af.val * ys.val * 8760.0) AS Value
        FROM AvailabilityFactor af
        JOIN YearSplit ys ON af.l = ys.l AND af.y = ys.y
        WHERE (:r IS NULL OR af.r = :r)
          AND (:t IS NULL OR af.t = :t)
        GROUP BY af.y
        """
        df = pd.read_sql_query(sql, conn, params={"r": r, "t": t})
    except Exception:
        return pd.DataFrame()
    for col in ("Year", "Value"):
        if col in df.columns:
            df[col] = df[col].apply(_decode_val)
    df["Units"] = ""
    df["Notes"] = ""
    return df[["Year", "Value", "Units", "Notes"]]


TRANSFORMS = {
    "full_load_hours": _transform_full_load_hours,
}


def run_mapping(conn: sqlite3.Connection, mapping: dict | None) -> pd.DataFrame:
    if mapping is None:
        return pd.DataFrame()
    if "transform" in mapping:
        tf = TRANSFORMS.get(mapping["transform"])
        if tf is None:
            return pd.DataFrame()
        return tf(conn, mapping.get("filters") or {})
    return _run_simple_mapping(conn, mapping)


def apply_mappings(df: pd.DataFrame, conn: sqlite3.Connection | None) -> pd.DataFrame:
    """
    Returns a tall dataframe with columns: Branch Path, Variable, Year, Value, Units, Notes.
    """
    if conn is None:
        return df.assign(Year="", Value="", Units="", Notes="")

    out_rows: list[dict] = []
    for _, row in df.iterrows():
        key = (row["Branch Path"], row["Variable"])
        mapping = MAPPINGS.get(key)
        extracted = run_mapping(conn, mapping) if mapping is not None else pd.DataFrame()
        if extracted is None or extracted.empty:
            out_rows.append({**row, "Year": "", "Value": "", "Units": "", "Notes": ""})
            continue
        for _, er in extracted.iterrows():
            out_rows.append(
                {
                    "Branch Path": row["Branch Path"],
                    "Variable": row["Variable"],
                    "Year": er.get("Year", ""),
                    "Value": er.get("Value", ""),
                    "Units": er.get("Units", ""),
                    "Notes": er.get("Notes", ""),
                }
            )
    return pd.DataFrame(out_rows)


def build_expression(year_values: list[tuple]) -> str:
    """
    Build a LEAP DATA expression from (year, value) pairs.
    If no years, return a single value string (if present) or empty.
    """
    clean_pairs = [(y, v) for y, v in year_values if pd.notna(y) and pd.notna(v) and y != ""]
    if clean_pairs:
        # sort by year, coerce to int where possible
        def to_int(y):
            try:
                return int(y)
            except Exception:
                return y

        clean_pairs = sorted(clean_pairs, key=lambda p: to_int(p[0]))
        parts = []
        for y, v in clean_pairs:
            parts.append(str(to_int(y)))
            parts.append(str(v))
        return f"DATA({', '.join(parts)})"

    # fallback to single value (first non-null)
    for _, v in year_values:
        if pd.notna(v) and v != "":
            return str(v)
    return ""


def aggregate_rows(df: pd.DataFrame, scenario: str, region: str) -> pd.DataFrame:
    """
    Aggregate tall Year/Value rows into one row per Branch/Variable (per Units/Notes).
    Build expression strings and expand branch levels.
    """
    records: list[dict] = []
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    group_cols = ["Branch Path", "Variable", "Units", "Notes"]
    for (branch, var, units, notes), g in df.groupby(group_cols, dropna=False):
        pairs = list(zip(g["Year"].tolist(), g["Value"].tolist()))
        expr = build_expression(pairs)
        meta = VARIABLE_META.get(var, {})
        scale = meta.get("Scale", "")
        per = meta.get("Per", "")
        # prefer units from data; else meta
        units_out = units if units not in (None, "", pd.NA) else meta.get("Units", "")
        levels = path_to_levels(branch, max_levels=len(LEVEL_COLS))
        record = {
            "Scenario": scenario,
            "Region": region,
            "Scale": scale,
            "Units": units_out,
            "Per": per,
            "Expression": expr,
            "Variable": var,
            "Notes": notes or "",
        }
        record.update({LEVEL_COLS[i]: levels[i] for i in range(len(LEVEL_COLS))})
        records.append(record)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def parse_args(argv: list[str]) -> tuple[str, str]:
    """
    Accept optional CLI args: scenario [region]
    """
    scenario = argv[1] if len(argv) > 1 else DEFAULT_SCENARIO
    region = argv[2] if len(argv) > 2 else (DEFAULT_REGION or scenario)
    return scenario, region


def main():
    scenario, region = parse_args(sys.argv)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_base = build_dataframe(ROWS, deduplicate=DEDUPLICATE_ROWS)
    if AUTO_FILL_FROM_DB and NEMO_DB_PATH is not None and NEMO_DB_PATH.exists():
        with sqlite3.connect(NEMO_DB_PATH) as conn:
            df_tall = apply_mappings(df_base, conn)
    else:
        df_tall = df_base.assign(Year="", Value="", Units="", Notes="")

    df_out = aggregate_rows(df_tall, scenario=scenario, region=region)
    df_out.to_excel(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df_out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
