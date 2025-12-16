
#%%
# Build a LEAP import spreadsheet by mapping NEMO DB values to LEAP Branch/Variable rows.
from __future__ import annotations

from pathlib import Path
import copy
import sqlite3
import sys
import pandas as pd

# -------------------------------------------------------------------
# User-configurable defaults
# -------------------------------------------------------------------
LEAP_TEMPLATE_DEFAULTS = {
    "DEFAULT_SCENARIO": "Target",
    "CURRENT_ACCOUNTS_SCENARIO": "Current Accounts",
    "DEFAULT_REGION": "Region 1",  # If None, mirror REGION_FILTER (or scenario if that is also None).
    "OUTPUT_PATH": Path("../data/leap_import_template.xlsx"),
    # Path to an existing LEAP export to copy IDs from (set to None to skip).
    "IMPORT_ID_SOURCE": Path("../data/import_files/USA_power_leap_import_REF.xlsx"),
    "IMPORT_ID_SHEET": "Export",  # Sheet name in the import file.
    "IMPORT_ID_HEADER_ROW": 2,  # Header row index (0-based) in the import file.
    "ID_CHECK_STRICT": True,  # If True, raise when IDs are missing after merge.
    "ID_CHECK_BREAK": True,  # If True, breakpoint() when IDs are missing after merge.
    "NEMO_DB_PATH": Path("../data/nemo.sqlite"),  # Set to None to skip autofill.
    "AUTO_FILL_FROM_DB": True,
    "DEDUPLICATE_ROWS": True,  # Drop duplicate Branch/Variable pairs before filling.
    "LEAP_MODEL_NAME": "USA transport",  # Populates the header row; change to your LEAP Area/Model name.
    "LEAP_VERSION": "2",  # Populates the Version field in the header row.
    # Region/tech mapping used when querying the NEMO DB.
    "REGION_FILTER": "20_USA",  # Change if you want another region or set to None to pull all.
    "TECH_MAP": {
        "Coal": "POW_Coal_PP",  # Update to your coal tech code if different.
    },
}


def apply_leap_template_defaults(vars_cfg: dict, data_dir: Path) -> dict:
    """
    Fill in LEAP-template-specific defaults using LEAP_TEMPLATE_DEFAULTS, resolving
    data-relative paths to the repository data directory.
    """
    out = dict(vars_cfg)

    def resolve_data_path(default_path: Path) -> Path:
        if default_path.is_absolute():
            return default_path
        parts = list(default_path.parts)
        if "data" in parts:
            data_idx = parts.index("data")
            remainder = Path(*parts[data_idx + 1 :])
            return data_dir / remainder
        return data_dir / default_path

    out.setdefault("GENERATE_LEAP_TEMPLATE", False)
    out.setdefault("LEAP_TEMPLATE_OUTPUT", resolve_data_path(LEAP_TEMPLATE_DEFAULTS["OUTPUT_PATH"]))
    out.setdefault("LEAP_TEMPLATE_REGION", LEAP_TEMPLATE_DEFAULTS["DEFAULT_REGION"])
    out.setdefault("LEAP_IMPORT_ID_SOURCE", resolve_data_path(LEAP_TEMPLATE_DEFAULTS["IMPORT_ID_SOURCE"]))
    return out


DEFAULT_SCENARIO = LEAP_TEMPLATE_DEFAULTS["DEFAULT_SCENARIO"]
CURRENT_ACCOUNTS_SCENARIO = LEAP_TEMPLATE_DEFAULTS["CURRENT_ACCOUNTS_SCENARIO"]
DEFAULT_REGION = LEAP_TEMPLATE_DEFAULTS["DEFAULT_REGION"]
OUTPUT_PATH = LEAP_TEMPLATE_DEFAULTS["OUTPUT_PATH"]
IMPORT_ID_SOURCE = LEAP_TEMPLATE_DEFAULTS["IMPORT_ID_SOURCE"]
IMPORT_ID_SHEET = LEAP_TEMPLATE_DEFAULTS["IMPORT_ID_SHEET"]
IMPORT_ID_HEADER_ROW = LEAP_TEMPLATE_DEFAULTS["IMPORT_ID_HEADER_ROW"]
ID_CHECK_STRICT = LEAP_TEMPLATE_DEFAULTS["ID_CHECK_STRICT"]
ID_CHECK_BREAK = LEAP_TEMPLATE_DEFAULTS["ID_CHECK_BREAK"]
NEMO_DB_PATH = LEAP_TEMPLATE_DEFAULTS["NEMO_DB_PATH"]
AUTO_FILL_FROM_DB = LEAP_TEMPLATE_DEFAULTS["AUTO_FILL_FROM_DB"]
DEDUPLICATE_ROWS = LEAP_TEMPLATE_DEFAULTS["DEDUPLICATE_ROWS"]
LEAP_MODEL_NAME = LEAP_TEMPLATE_DEFAULTS["LEAP_MODEL_NAME"]
LEAP_VERSION = LEAP_TEMPLATE_DEFAULTS["LEAP_VERSION"]
REGION_FILTER = LEAP_TEMPLATE_DEFAULTS["REGION_FILTER"]
TECH_MAP = dict(LEAP_TEMPLATE_DEFAULTS["TECH_MAP"])

# Processes to include in the template. Add new entries here and the rows will be generated
# automatically for both the main scenario and Current Accounts (unless disabled).
PROCESS_CONFIGS: list[dict] = [
    {"name": "Coal", "feedstock": "Coal"},
    {"name": "Natural Gas", "feedstock": "Natural Gas"},
    {"name": "Wind", "feedstock": "Wind"},
    {"name": "Hydro", "feedstock": "Hydro"},
]

# -------------------------------------------------------------------
# Base rows (Branch Path, Variable). Extend this list to add more items.
# -------------------------------------------------------------------
ROWS: list[tuple[str, str]] = [
    ("Transformation\\Electricity Generation", "Planning Reserve Margin"),
    ("Transformation\\Electricity Generation", "Peak Load Ratio"),
    ("Transformation\\Electricity Generation", "Module Costs"),
    ("Transformation\\Electricity Generation", "Renewable Target"),
    ("Transformation\\Electricity Generation", "Optimize"),
    ("Transformation\\Electricity Generation", "Use Addition Size"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Price"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Share"),
]

# Rows specific to the Current Accounts scenario.
CURRENT_ACCOUNTS_ROWS: list[tuple[str, str]] = [
    ("Transformation\\Electricity Generation", "Planning Reserve Margin"),
    ("Transformation\\Electricity Generation", "Peak Load Ratio"),
    ("Transformation\\Electricity Generation", "Module Costs"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Shortfall Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Surplus Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Usage Rule"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Import Target"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Export Target"),
    ("Transformation\\Electricity Generation\\Output Fuels\\Electricity", "Output Price"),
]

# LEAP columns and level columns
LEVEL_COLS = [f"Level {i}" for i in range(1, 9)]
OUTPUT_COLUMNS = [
    "BranchID",
    "VariableID",
    "ScenarioID",
    "RegionID",
    "Branch Path",
    "Variable",
    "Scenario",
    "Region",
    "Scale",
    "Units",
    "Per...",
    "Expression",
] + LEVEL_COLS

# Variable metadata for Scale/Units/Per defaults (extend as needed)
VARIABLE_META = {
    "Planning Reserve Margin": {"Scale": "", "Units": "", "Per...": ""},
    "Interest Rate": {"Scale": "", "Units": "%", "Per...": ""},
    "Variable OM Cost": {"Scale": "", "Units": "USD/MWh", "Per...": ""},
    "Fixed OM Cost": {"Scale": "", "Units": "USD/MW", "Per...": ""},
    "Capital Cost": {"Scale": "", "Units": "USD/MW", "Per...": ""},
    "Maximum Capacity": {"Scale": "", "Units": "MW", "Per...": ""},
    "Minimum Capacity": {"Scale": "", "Units": "MW", "Per...": ""},
    "Maximum Capacity Addition": {"Scale": "", "Units": "MW", "Per...": ""},
    "Minimum Capacity Addition": {"Scale": "", "Units": "MW", "Per...": ""},
    "Exogenous Capacity": {"Scale": "", "Units": "MW", "Per...": ""},
    "Maximum Availability": {"Scale": "", "Units": "fraction", "Per...": ""},
    "Minimum Utilization": {"Scale": "", "Units": "fraction", "Per...": ""},
    "Capacity Credit": {"Scale": "", "Units": "fraction", "Per...": ""},
    "Full Load Hours": {"Scale": "", "Units": "hours", "Per...": ""},
    "Maximum Production": {"Scale": "", "Units": "GWh", "Per...": ""},
    "Minimum Production": {"Scale": "", "Units": "GWh", "Per...": ""},
    "Minimum Share of Production": {"Scale": "", "Units": "fraction", "Per...": ""},
    "Lifetime": {"Scale": "", "Units": "years", "Per...": ""},
    "Renewable Qualified": {"Scale": "", "Units": "flag", "Per...": ""},
}

# Process row templates used to generate rows for each process in PROCESS_CONFIGS.
CURRENT_ACCOUNTS_PROCESS_VARS = [
    "Dispatch Rule",
    "Lifetime",
    "Interest Rate",
    "Exogenous Capacity",
    "Maximum Availability",
    "Minimum Utilization",
    "Capacity Credit",
    "Dispatchable",
    "Endogenous Capacity",
    "Merit Order",
    "First Simulation Year",
    "Minimum Charge",
    "Starting Charge",
    "Full Load Hours",
    "Annual Storage Carryover",
    "Seasonal Storage Carryover",
    "Hourly Storage Carryover",
    "Process Efficiency",
    "Variable OM Cost",
    "Capital Cost",
    "Stranded Cost",
    "Salvage Value",
    "Fixed OM Cost",
]

TARGET_PROCESS_VARS = [
    # "Dispatch Rule",
    "Optimized New Capacity",
    "Renewable Qualified",
    "Lifetime",
    "Maximum Production",
    "Minimum Production",
    "Minimum Share of Production",
    "Interest Rate",
    "Minimum Capacity",
    "Maximum Capacity",
    "Maximum Capacity Addition",
    "Minimum Capacity Addition",
    "Exogenous Capacity",
    "Maximum Availability",
    "Minimum Utilization",
    "Capacity Credit",
    "Minimum Charge",
    "Full Load Hours",
    "Process Efficiency",
    "Variable OM Cost",
    "Capital Cost",
    "Stranded Cost",
    "Salvage Value",
    "Fixed OM Cost",
]

CURRENT_ACCOUNTS_FEEDSTOCK_VARS = [
    "Feedstock Fuel Share",
    "Fuel Cost",
]

TARGET_FEEDSTOCK_VARS = [
    "Feedstock Fuel Share",
    "Fuel Cost",
]

# -------------------------------------------------------------------
# Mapping definitions
# Each entry maps (Branch Path, Variable) -> config:
#   table: NEMO table name
#   select: {OutCol: SQL_expr} with aliases (Year/Value/Units supported)
#   filters: dict of column->value
#   group_by: list of columns to group on (for aggregates)
#   transform: string key for special handling
# -------------------------------------------------------------------
BASE_MAPPINGS: dict[tuple[str, str], dict | None] = {
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

UNIMPORTANT_VARIABLES_TO_SKIP_FROM_LEAP_IMPORT = {
    "Module Costs",
}

# Process mapping templates (applied per entry in PROCESS_CONFIGS using TECH_MAP).
PROCESS_MAPPING_TEMPLATES = [
    ("Lifetime", {"table": "OperationalLife", "select": {"Year": "''", "Value": "val"}}),
    ("Interest Rate", {"table": "InterestRateTechnology", "select": {"Year": "y", "Value": "val"}}),
    ("Exogenous Capacity", {"table": "ResidualCapacity", "select": {"Year": "y", "Value": "val"}}),
    ("Maximum Availability", {"table": "AvailabilityFactor", "select": {"Year": "y", "Value": "MAX(val)"}, "group_by": ["y"]}),
    ("Minimum Utilization", {"table": "MinimumUtilization", "select": {"Year": "y", "Value": "MAX(val)"}, "group_by": ["y"]}),
    ("Capacity Credit", {"table": "ReserveMarginTagTechnology", "select": {"Year": "y", "Value": "val"}, "filters_extra": {"f": "ALL"}}),
    ("Full Load Hours", {"transform": "full_load_hours"}),
    ("Variable OM Cost", {"table": "VariableCost", "select": {"Year": "y", "Value": "val"}}),
    ("Capital Cost", {"table": "CapitalCost", "select": {"Year": "y", "Value": "val"}}),
    ("Fixed OM Cost", {"table": "FixedCost", "select": {"Year": "y", "Value": "val"}}),
    ("Maximum Capacity", {"table": "TotalAnnualMaxCapacity", "select": {"Year": "y", "Value": "val"}}),
    ("Minimum Capacity", {"table": "TotalAnnualMinCapacity", "select": {"Year": "y", "Value": "val"}}),
    ("Maximum Capacity Addition", {"table": "TotalAnnualMaxCapacityInvestment", "select": {"Year": "y", "Value": "val"}}),
    ("Minimum Capacity Addition", {"table": "TotalAnnualMinCapacityInvestment", "select": {"Year": "y", "Value": "val"}}),
    ("Maximum Production", {"table": "TotalTechnologyAnnualActivityUpperLimit", "select": {"Year": "y", "Value": "val"}}),
    ("Minimum Production", {"table": "TotalTechnologyAnnualActivityLowerLimit", "select": {"Year": "y", "Value": "val"}}),
    ("Minimum Share of Production", {"table": "MinShareProduction", "select": {"Year": "y", "Value": "val"}, "filters_extra": {"f": "ALL"}}),
    ("Renewable Qualified", {"table": "RETagTechnology", "select": {"Year": "y", "Value": "val"}}),
]


def generate_process_mappings_for_process(process_name: str, tech_code: str | None) -> dict[tuple[str, str], dict]:
    """
    Build mapping dict entries for a single process using PROCESS_MAPPING_TEMPLATES and the given tech code.
    """
    out: dict[tuple[str, str], dict] = {}
    base_branch = f"Transformation\\Electricity Generation\\Processes\\{process_name}"
    for var, template in PROCESS_MAPPING_TEMPLATES:
        mapping = copy.deepcopy(template)
        filters = mapping.get("filters", {})
        filters.update({"r": REGION_FILTER})
        if tech_code is not None:
            filters.update({"t": tech_code})
        if "filters_extra" in mapping:
            filters.update(mapping.pop("filters_extra"))
        mapping["filters"] = filters
        out[(base_branch, var)] = mapping
    return out


def build_mappings(process_configs: list[dict]) -> dict[tuple[str, str], dict | None]:
    """
    Combine base mappings with per-process mappings derived from PROCESS_CONFIGS and TECH_MAP.
    """
    mappings = dict(BASE_MAPPINGS)
    for cfg in process_configs:
        proc_name = cfg.get("name")
        if not proc_name:
            continue
        tech_code = cfg.get("tech") or TECH_MAP.get(proc_name) or proc_name
        mappings.update(generate_process_mappings_for_process(proc_name, tech_code))
    return mappings


MAPPINGS: dict[tuple[str, str], dict | None] = build_mappings(PROCESS_CONFIGS)


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


def _run_simple_mapping_sql(conn: sqlite3.Connection, mapping: dict) -> pd.DataFrame:
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
    for col in ["Year", "Value", "Units"]:
        if col not in df.columns:
            df[col] = ""
    return df[["Year", "Value", "Units"]]


def _run_simple_mapping_excel(table_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    select_map: dict = mapping.get("select") or {}
    filters: dict = mapping.get("filters") or {}
    group_by = mapping.get("group_by") or []

    if table_df is None or table_df.empty or not select_map:
        return pd.DataFrame()

    df = table_df.copy()
    for col, val in filters.items():
        if val is None:
            continue
        if col not in df.columns:
            return pd.DataFrame()
        df = df[df[col] == val]

    # Helper to resolve expressions
    def resolve_expr(expr: str, frame: pd.DataFrame):
        if expr == "''":
            return ""
        if expr in frame.columns:
            return frame[expr]
        if expr == "MAX(val)":
            return frame["val"].max()
        return None

    if group_by:
        if any(g not in df.columns for g in group_by):
            return pd.DataFrame()
        grouped = df.groupby(group_by)
        rows = []
        for keys, g in grouped:
            row = {}
            for alias, expr in select_map.items():
                val = resolve_expr(expr, g)
                row[alias] = val
            rows.append(row)
        out_df = pd.DataFrame(rows)
    else:
        out = {}
        for alias, expr in select_map.items():
            val = resolve_expr(expr, df)
            out[alias] = val
        out_df = pd.DataFrame(out)

    for col in ("Value", "Year"):
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(_decode_val)
    for col in ["Year", "Value", "Units"]:
        if col not in out_df.columns:
            out_df[col] = ""
    return out_df[["Year", "Value", "Units"]]


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
    return df[["Year", "Value", "Units"]]


def _transform_full_load_hours_excel(tables: dict[str, pd.DataFrame], filters: dict) -> pd.DataFrame:
    af = tables.get("AvailabilityFactor")
    ys = tables.get("YearSplit")
    if af is None or ys is None or af.empty or ys.empty:
        return pd.DataFrame()
    r = filters.get("r")
    t = filters.get("t")
    af_df = af.copy()
    if r is not None:
        af_df = af_df[af_df["r"] == r]
    if t is not None:
        af_df = af_df[af_df["t"] == t]
    merged = af_df.merge(ys, on=["l", "y"], how="inner", suffixes=("_af", "_ys"))
    if merged.empty:
        return pd.DataFrame()
    merged["Value"] = merged["val_af"] * merged["val_ys"] * 8760.0
    out = merged.groupby("y")["Value"].sum().reset_index()
    out.rename(columns={"y": "Year"}, inplace=True)
    out["Units"] = ""
    return out[["Year", "Value", "Units"]]


TRANSFORMS = {
    "full_load_hours": _transform_full_load_hours,
}


def run_mapping(
    conn: sqlite3.Connection | None, mapping: dict | None, excel_tables: dict | None = None
) -> pd.DataFrame:
    if mapping is None:
        return pd.DataFrame()
    if "transform" in mapping:
        tf = TRANSFORMS.get(mapping["transform"])
        if tf is None:
            return pd.DataFrame()
        if conn is not None:
            return tf(conn, mapping.get("filters") or {})
        if excel_tables is not None and mapping["transform"] == "full_load_hours":
            return _transform_full_load_hours_excel(excel_tables, mapping.get("filters") or {})
        return pd.DataFrame()
    if conn is not None:
        return _run_simple_mapping_sql(conn, mapping)
    if excel_tables is not None:
        table_name = mapping.get("table")
        return _run_simple_mapping_excel(excel_tables.get(table_name), mapping)
    return pd.DataFrame()


def apply_mappings(df: pd.DataFrame, conn: sqlite3.Connection | None, excel_tables: dict | None = None) -> pd.DataFrame:
    """
    Returns a tall dataframe with columns: Branch Path, Variable, Year, Value, Units.
    """
    if conn is None:
        return df.assign(Year="", Value="", Units="")

    out_rows: list[dict] = []
    for _, row in df.iterrows():
        key = (row["Branch Path"], row["Variable"])
        mapping = MAPPINGS.get(key)
        extracted = run_mapping(conn, mapping, excel_tables=excel_tables) if mapping is not None else pd.DataFrame()
        if extracted is None or extracted.empty:
            out_rows.append({**row, "Year": "", "Value": "", "Units": ""})
            continue
        for _, er in extracted.iterrows():
            out_rows.append(
                {
                    "Branch Path": row["Branch Path"],
                    "Variable": row["Variable"],
                    "Year": er.get("Year", ""),
                    "Value": er.get("Value", ""),
                    "Units": er.get("Units", ""),
                }
            )
    return pd.DataFrame(out_rows)


def generate_process_rows(process_name: str, feedstock_name: str | None = None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Build lists of (Branch Path, Variable) tuples for a single process for both Current
    Accounts and the main scenario. Feedstock branch defaults to the process name.
    """
    base_branch = f"Transformation\\Electricity Generation\\Processes\\{process_name}"
    feedstock = feedstock_name or process_name

    ca_rows = [(base_branch, v) for v in CURRENT_ACCOUNTS_PROCESS_VARS]
    target_rows = [(base_branch, v) for v in TARGET_PROCESS_VARS]

    if feedstock:
        feed_branch = f"{base_branch}\\Feedstock Fuels\\{feedstock}"
        ca_rows.extend((feed_branch, v) for v in CURRENT_ACCOUNTS_FEEDSTOCK_VARS)
        target_rows.extend((feed_branch, v) for v in TARGET_FEEDSTOCK_VARS)

    return ca_rows, target_rows


def assemble_all_rows(process_configs: list[dict]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Combine static rows with dynamically generated process rows from PROCESS_CONFIGS.
    Returns (current_accounts_rows, scenario_rows).
    """
    ca_rows = list(CURRENT_ACCOUNTS_ROWS)
    scenario_rows = list(ROWS)

    for cfg in process_configs:
        proc_name = cfg.get("name")
        if not proc_name:
            continue
        feedstock = cfg.get("feedstock")
        include_ca = cfg.get("include_current_accounts", True)
        include_scenario = cfg.get("include_scenario", True)

        ca_add, scen_add = generate_process_rows(proc_name, feedstock_name=feedstock)
        if include_ca:
            ca_rows.extend(ca_add)
        if include_scenario:
            scenario_rows.extend(scen_add)

    return ca_rows, scenario_rows


def load_import_ids(path: Path, sheet_name: str, header_row: int) -> pd.DataFrame | None:
    """
    Read an existing LEAP export/import file to pull IDs for Branch/Variable/Scenario/Region.
    Returns None if the file is missing or cannot be read.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    except Exception:
        return None
    needed_cols = {"Branch Path", "Variable", "Scenario", "Region", "BranchID", "VariableID", "ScenarioID", "RegionID"}
    missing = needed_cols - set(df.columns)
    if missing:
        # If structure doesn't match, skip using IDs.
        return None
    df = df[list(needed_cols)]
    return df.drop_duplicates(subset=["Branch Path", "Variable", "Scenario", "Region"], keep="last")


def attach_ids(df: pd.DataFrame, import_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge ID columns from an imported LEAP sheet if available.
    """
    if import_df is None or import_df.empty:
        merged = df.copy()
        merged["_merge"] = "left_only"
    else:
        merged = df.merge(
            import_df,
            how="left",
            on=["Branch Path", "Variable", "Scenario", "Region"],
            suffixes=("", "_import"),
            indicator=True,
        )
    for col in ["BranchID", "VariableID", "ScenarioID", "RegionID"]:
        import_col = f"{col}_import"
        fallback = (
            merged[import_col]
            if import_col in merged.columns
            else pd.Series([pd.NA] * len(merged), index=merged.index)
        )
        merged[col] = merged[col].combine_first(fallback)
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_import")])
    return merged

def validate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Warn/error when ID columns are missing after merge; controlled by ID_CHECK_* flags.
    Handles merge diagnostics via the pandas merge indicator column when present.
    """
    id_cols = [c for c in ["BranchID", "VariableID", "ScenarioID", "RegionID"] if c in df.columns]
    if not id_cols:
        return df

    merge_col = "_merge" if "_merge" in df.columns else None
    subset_cols = [c for c in ["Branch Path", "Variable", "Scenario", "Region"] + id_cols if c in df.columns]

    def _report(mask: pd.Series, msg: str) -> None:
        rows = df.loc[mask, subset_cols]
        if not rows.empty:
            print(f"[WARN] {msg}")
            print(rows.head(10).to_string(index=False))
            if ID_CHECK_BREAK:
                breakpoint()
            if ID_CHECK_STRICT:
                #save teh df to a csv for easier debugging.. data\errors/leap_id_errors.csv
                error_path = Path("../data/errors/leap_id_errors.csv")
                error_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(error_path, index=False)
                print(f"[INFO] Saved error rows to {error_path}")
                
                raise ValueError(msg)

    if merge_col:
        left_only = df[merge_col] == "left_only"
        right_only = df[merge_col] == "right_only"
        if "Variable" in df.columns and merge_col in df.columns:
            skip_mask = left_only & df["Variable"].isin(UNIMPORTANT_VARIABLES_TO_SKIP_FROM_LEAP_IMPORT)
            left_only = left_only & ~skip_mask
        _report(
            left_only,
            f"{left_only.sum()} rows in the generated template could not find matching IDs in the reference file. "
            "Please confirm the Branch/Variable/Scenario/Region names match the import sheet."
        )
        if right_only.any():
            _report(
                right_only,
                f"{right_only.sum()} rows exist in the reference import file but were not recreated here. "
                "Please confirm the process list in this script covers every required Branch/Variable combination."
            )

    missing_mask = df[id_cols].isna().any(axis=1)
    if merge_col:
        missing_mask &= df[merge_col] == "both"
    if missing_mask.any():
        _report(missing_mask, f"Missing IDs for {missing_mask.sum()} matched rows (_merge=both)")

    if merge_col and "_merge" in df.columns:
        df = df.drop(columns=["_merge"])
    return df

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


def prepend_leap_headers(df: pd.DataFrame, model_name: str, version: str) -> pd.DataFrame:
    """
    Insert two header rows plus a column-name row to match LEAP's expected import format.
    """
    df2 = df.copy()
    header0 = {col: "" for col in df2.columns}
    # Place Area/Model/Ver in columns E-H (Branch Path, Variable, Scenario, Region) after the ID columns.
    if "Branch Path" in df2.columns:
        header0["Branch Path"] = "Area:"
    if "Variable" in df2.columns:
        header0["Variable"] = model_name
    if "Scenario" in df2.columns:
        header0["Scenario"] = "Ver:"
    if "Region" in df2.columns:
        header0["Region"] = version
    row0 = pd.DataFrame([header0])
    row1 = pd.DataFrame([{col: pd.NA for col in df2.columns}])
    row2 = pd.DataFrame([df2.columns], columns=df2.columns)
    return pd.concat([row0, row1, row2, df2], ignore_index=True)


def aggregate_rows(df: pd.DataFrame, scenario: str, region: str) -> pd.DataFrame:
    """
    Aggregate tall Year/Value rows into one row per Branch/Variable (per Units).
    Build expression strings and expand branch levels.
    """
    records: list[dict] = []
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    group_cols = ["Branch Path", "Variable", "Units"]
    for (branch, var, units), g in df.groupby(group_cols, dropna=False):
        pairs = list(zip(g["Year"].tolist(), g["Value"].tolist()))
        expr = build_expression(pairs)
        meta = VARIABLE_META.get(var, {})
        scale = meta.get("Scale", "")
        per = meta.get("Per...", "")
        # prefer units from data; else meta
        units_out = units if units not in (None, "", pd.NA) else meta.get("Units", "")
        levels = path_to_levels(branch, max_levels=len(LEVEL_COLS))
        record = {
            "BranchID": pd.NA,
            "VariableID": pd.NA,
            "ScenarioID": pd.NA,
            "RegionID": pd.NA,
            "Branch Path": branch,
            "Variable": var,
            "Scenario": scenario,
            "Region": region,
            "Scale": scale,
            "Units": units_out,
            "Per...": per,
            "Expression": expr,
        }
        record.update({LEVEL_COLS[i]: levels[i] for i in range(len(LEVEL_COLS))})
        records.append(record)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def build_scenario_output(
    rows: list[tuple[str, str]],
    scenario: str,
    region: str,
    conn: sqlite3.Connection | None,
    excel_tables: dict | None = None,
) -> pd.DataFrame:
    df_base = build_dataframe(rows, deduplicate=DEDUPLICATE_ROWS)
    df_tall = apply_mappings(df_base, conn, excel_tables=excel_tables)
    return aggregate_rows(df_tall, scenario=scenario, region=region)


def parse_args(argv: list[str]) -> tuple[str, str]:
    """
    Accept optional CLI args: scenario [region]. If region is omitted, prefer DEFAULT_REGION,
    then REGION_FILTER, and finally fall back to the scenario string.
    """
    # Filter out IPython/Jupyter injected args like "-f"/"--f" that carry a JSON path.
    cleaned = [a for a in argv[1:] if not (a.startswith("-f") or a.startswith("--f"))]
    scenario = cleaned[0] if cleaned else DEFAULT_SCENARIO
    if len(cleaned) > 1:
        region = cleaned[1]
    elif DEFAULT_REGION is not None:
        region = DEFAULT_REGION
    elif REGION_FILTER is not None:
        region = REGION_FILTER
    else:
        region = scenario
    return scenario, region


def main():
    scenario, region = parse_args(sys.argv)
    generate_leap_template(
        scenario=scenario,
        region=region,
        nemo_db_path=NEMO_DB_PATH,
        output_path=OUTPUT_PATH,
        import_id_source=IMPORT_ID_SOURCE,
    )
    return OUTPUT_PATH


def generate_leap_template(
    scenario: str | None = None,
    region: str | None = None,
    nemo_db_path: Path | str | None = None,
    nemo_entry_excel_path: Path | str | None = None,
    output_path: Path | str | None = None,
    import_id_source: Path | str | None = None,
) -> Path:
    """
    Build the LEAP import template, optionally overriding DB/output paths.
    """
    scenario = scenario or DEFAULT_SCENARIO
    region = region or DEFAULT_REGION or REGION_FILTER or scenario
    output_path = Path(output_path or OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = Path(nemo_db_path) if nemo_db_path is not None else NEMO_DB_PATH
    excel_tables = None
    if nemo_entry_excel_path is not None and Path(nemo_entry_excel_path).exists():
        excel_tables = load_excel_tables(Path(nemo_entry_excel_path))
    ca_rows, scenario_rows = assemble_all_rows(PROCESS_CONFIGS)

    scenario_sets: list[tuple[str, list[tuple[str, str]]]] = [
        (CURRENT_ACCOUNTS_SCENARIO, ca_rows),
        (scenario, scenario_rows),
    ]
    seen_scenarios: set[str] = set()
    frames: list[pd.DataFrame] = []

    conn: sqlite3.Connection | None = None
    if AUTO_FILL_FROM_DB and db_path is not None and Path(db_path).exists():
        conn = sqlite3.connect(db_path)

    try:
        for scen_name, rows in scenario_sets:
            if scen_name in seen_scenarios:
                continue
            seen_scenarios.add(scen_name)
            frames.append(
                build_scenario_output(
                    rows,
                    scenario=scen_name,
                    region=region,
                    conn=conn,
                    excel_tables=excel_tables,
                )
            )
    finally:
        if conn is not None:
            conn.close()

    df_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    import_source = Path(import_id_source) if import_id_source is not None else IMPORT_ID_SOURCE
    import_ids_df = (
        load_import_ids(import_source, sheet_name=IMPORT_ID_SHEET, header_row=IMPORT_ID_HEADER_ROW)
        if import_source is not None
        else None
    )
    if import_ids_df is None and import_source is not None:
        raise ValueError(f"Could not load import IDs from the specified file: {import_source}")
    df_out = attach_ids(df_out, import_ids_df)
    df_out = validate_ids(df_out)
    df_out_with_headers = prepend_leap_headers(df_out, model_name=LEAP_MODEL_NAME, version=LEAP_VERSION)
    df_out_with_headers.to_excel(output_path, index=False, header=False)
    print(f"Wrote {len(df_out)} rows (plus headers) to {output_path}")
    return output_path


__all__ = [
    "LEAP_TEMPLATE_DEFAULTS",
    "apply_leap_template_defaults",
    "generate_leap_template",
    "main",
]

#%%
if __name__ == "__main__":
    main()

#%%
