#%%
import pandas as pd
import sqlite3
from pathlib import Path
from shutil import copyfile
from typing import Any, Mapping, DefaultDict
from collections import defaultdict

#note that the .sqlite DB template must be created once with NEMO's createnemodb() in Julia or via LEAP before running this script e.g. in Julia:
# # using Pkg
# # Pkg.add(url="https://github.com/sei-international/NemoMod.jl")
# # then:
# # using NemoMod
# # cd("C:/Users/YOU/PROJECTNAME/data")   # change to your data folder path VERY IMPORTANT otherwise createnemodb will create the DB template in the wrong folder!
# # NemoMod.createnemodb("nemo_template.sqlite")

# For each Excel sheet: define NEMO table + index columns (Excel names)
PARAM_SPECS = {
    # Demand
    "SpecifiedAnnualDemand": {
        "nemo_table": "SpecifiedAnnualDemand",
        "indices": ["REGION", "FUEL"],  # -> r, f, plus y, val
        "filter_scenario": True,
        "unit_type": "energy",
    },
    "SpecifiedDemandProfile": {
        "nemo_table": "SpecifiedDemandProfile",
        "indices": ["REGION", "FUEL", "TIMESLICE"],  # -> r, f, l, y, val
        "filter_scenario": True,
        "unit_type": "energy",
    },
    # Tech parameters
    "CapacityFactor": {
        # NEMO uses AvailabilityFactor for per-timeslice capacity factors.
        "nemo_table": "AvailabilityFactor",
        "indices": ["REGION", "TECHNOLOGY", "TIMESLICE"],
        "filter_scenario": False,
    },
    "CapacityOfOneTechnologyUnit": {
        "nemo_table": "CapacityOfOneTechnologyUnit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "has_years": False,
    },
    "CapacityToActivityUnit": {
        "nemo_table": "CapacityToActivityUnit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "has_years": False,
    },
    "CapitalCost": {
        "nemo_table": "CapitalCost",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": None,  # currency per capacity; leave as-is
    },
    "FixedCost": {
        "nemo_table": "FixedCost",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
    },
    "VariableCost": {
        "nemo_table": "VariableCost",
        "indices": ["REGION", "TECHNOLOGY", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},
    },
    "ResidualCapacity": {
        "nemo_table": "ResidualCapacity",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "power",
    },
    "InputActivityRatio": {
        "nemo_table": "InputActivityRatio",
        "indices": ["REGION", "TECHNOLOGY", "FUEL", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},
    },
    "OutputActivityRatio": {
        "nemo_table": "OutputActivityRatio",
        "indices": ["REGION", "TECHNOLOGY", "FUEL", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},
    },
    "EmissionActivityRatio": {
        "nemo_table": "EmissionActivityRatio",
        "indices": ["REGION", "TECHNOLOGY", "EMISSION", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},  # fallback when MO column missing/blank
    },
    # Advanced: Reserve margin (system-level)
    "ReserveMargin": {
        "nemo_table": "ReserveMargin",
        "indices": ["REGION", "FUEL"],  # -> r, f, y, val
        "defaults": {"FUEL": "ALL"},
        "filter_scenario": False,
        "advanced_flag": "ReserveMargin",
    },
    "ReserveMarginTagTechnology": {
        "nemo_table": "ReserveMarginTagTechnology",
        "indices": ["REGION", "TECHNOLOGY", "FUEL"],
        "filter_scenario": False,
        "advanced_flag": "ReserveMargin",
        # Many OSeMOSYS workbooks omit the FUEL column; default to ALL so we
        # still import the sheet instead of skipping it.
        "defaults": {"FUEL": "ALL"},
    },
    # Advanced: Annual emission limit
    "AnnualEmissionLimit": {
        "nemo_table": "AnnualEmissionLimit",
        "indices": ["REGION", "EMISSION"],  # -> r, e, y, val
        "filter_scenario": True,
        "advanced_flag": "AnnualEmissionLimit",
    },
    "TotalAnnualMaxCapacity": {
        "nemo_table": "TotalAnnualMaxCapacity",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "power",
    },
    "TotalAnnualMinCapacity": {
        "nemo_table": "TotalAnnualMinCapacity",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "power",
    },
    "TotalTechnologyAnnualActivityUp": {
        "nemo_table": "TotalTechnologyAnnualActivityUpperLimit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "energy",
    },
    "TotalTechnologyAnnualActivityLo": {
        "nemo_table": "TotalTechnologyAnnualActivityLowerLimit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "energy",
    },
    "TotalTechnologyModelPeriodActLo": {
        "nemo_table": "TotalTechnologyModelPeriodActivityLowerLimit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "energy",
    },
    "TotalTechnologyModelPeriodActUp": {
        "nemo_table": "TotalTechnologyModelPeriodActivityUpperLimit",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "unit_type": "energy",
    },
    # Storage and related tables
    "OperationalLife": {
        "nemo_table": "OperationalLife",
        "indices": ["REGION", "TECHNOLOGY"],
        "filter_scenario": False,
        "has_years": False,
    },
    "OperationalLifeStorage": {
        "nemo_table": "OperationalLifeStorage",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
        "has_years": False,
    },
    "CapitalCostStorage": {
        "nemo_table": "CapitalCostStorage",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
    },
    "ResidualStorageCapacity": {
        "nemo_table": "ResidualStorageCapacity",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
    },
    "StorageFullLoadHours": {
        "nemo_table": "StorageFullLoadHours",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
    },
    "StorageLevelStart": {
        "nemo_table": "StorageLevelStart",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
        "has_years": False,
    },
    "StorageMaxChargeRate": {
        "nemo_table": "StorageMaxChargeRate",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
        "has_years": False,
    },
    "StorageMaxDischargeRate": {
        "nemo_table": "StorageMaxDischargeRate",
        "indices": ["REGION", "STORAGE"],
        "filter_scenario": False,
        "has_years": False,
    },
    "TechnologyToStorage": {
        "nemo_table": "TechnologyToStorage",
        "indices": ["REGION", "TECHNOLOGY", "STORAGE", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},
        "has_years": False,
    },
    "TechnologyFromStorage": {
        "nemo_table": "TechnologyFromStorage",
        "indices": ["REGION", "TECHNOLOGY", "STORAGE", "MODE_OF_OPERATION"],
        "filter_scenario": False,
        "defaults": {"MODE_OF_OPERATION": "1"},
        "has_years": False,
    },
    "DefaultParams": {
        "nemo_table": "DefaultParams",
        "indices": ["TABLENAME"],
        "filter_scenario": False,
        "has_years": False,
        "raw": True,
    },
    "STORAGE": {
        "nemo_table": "STORAGE",
        "indices": ["STORAGE", "DESC", "NETZEROYEAR", "NETZEROTG1", "NETZEROTG2"],
        "filter_scenario": False,
        "has_years": False,
        "raw": True,
    },
    "TSGROUP1": {
        "nemo_table": "TSGROUP1",
        "indices": ["NAME", "DESC", "ORDER", "MULTIPLIER"],
        "filter_scenario": False,
        "has_years": False,
        "raw": True,
    },
    "TSGROUP2": {
        "nemo_table": "TSGROUP2",
        "indices": ["NAME", "DESC", "ORDER", "MULTIPLIER"],
        "filter_scenario": False,
        "has_years": False,
        "raw": True,
    },
    "LTsGroup": {
        "nemo_table": "LTsGroup",
        "indices": ["TIMESLICE", "TSGROUP2", "TSGROUP1", "LORDER"],
        "filter_scenario": False,
        "has_years": False,
        "raw": True,
    },
    # Time-slice weights
    "YearSplit": {
        "nemo_table": "YearSplit",
        "indices": ["TIMESLICE"],  # -> l, y, val
        "filter_scenario": False,
    },
    # You can add more advanced ones here:
    # "EmissionsPenalty": {...},
    # "ModelPeriodEmissionLimit": {...},
}

# Map Excel index names -> NEMO index names
INDEX_NAME_MAP = {
    "REGION": "r",
    "TECHNOLOGY": "t",
    "FUEL": "f",
    "TIMESLICE": "l",
    "EMISSION": "e",
    "MODE_OF_OPERATION": "m",
    "STORAGE": "s",
    "TSGROUP1": "tg1",
    "TSGROUP2": "tg2",
    "LORDER": "lorder",
    "TABLENAME": "tablename",
    "DESC": "desc",
    "ORDER": "order",
    "MULTIPLIER": "multiplier",
    "NETZEROYEAR": "netzeroyear",
    "NETZEROTG1": "netzerotg1",
    "NETZEROTG2": "netzerotg2",
}
INDEX_NAME_MAP_REV = {v: k for k, v in INDEX_NAME_MAP.items()}

# Sheets that populate set tables directly (simple VALUE column).
SET_SHEETS = {
    "REGION": "REGION",
    "TECHNOLOGY": "TECHNOLOGY",
    "FUEL": "FUEL",
    "TIMESLICE": "TIMESLICE",
    "EMISSION": "EMISSION",
    "MODE_OF_OPERATION": "MODE_OF_OPERATION",
    "YEAR": "YEAR",
}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def get_year_columns(cols):
    """Extract columns that look like years (e.g., 2017, 2018, 2030...)."""
    out = []
    for c in cols:
        c_str = str(c).strip()
        try:
            int(c_str)
            # Keep column names as strings for lookup; convert to int later if needed.
            out.append(c_str)
        except ValueError:
            continue
    return out


def copy_template_db(template: Path, output: Path):
    if not template.exists():
        raise FileNotFoundError(
            f"Template DB '{template}' not found. "
            "Create it once with NEMO's createnemodb() in Julia or via LEAP."
        )
    copyfile(template, output)
    print(f"Copied template DB to '{output}'.")


def load_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except ValueError:
        print(f"  Sheet '{sheet_name}' not found. Skipping.")
        return pd.DataFrame()

    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def filter_scenario(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if "SCENARIO" not in df.columns:
        return df
    return df[df["SCENARIO"].astype(str) == target]


def _normalize_mode_value(val):
    """
    Normalize MODE_OF_OPERATION values so that numeric inputs like 1.0 become '1'.
    Keeps other strings as-is after stripping.
    """
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s == "":
        return pd.NA
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def build_rows_from_wide(df: pd.DataFrame, indices, year_cols, defaults: dict[str, Any] | None = None):
    """
    Convert wide-year dataframe to list of dicts with
    keys: indices + ['YEAR', 'VALUE'].
    """
    defaults = defaults or {}
    rows = []
    for _, r in df.iterrows():
        base = {}
        skip = False
        for idx in indices:
            val = r[idx] if idx in r else None
            if pd.isna(val):
                if idx in defaults:
                    val = defaults[idx]
                else:
                    skip = True
                    break
            base[idx] = str(val)
        if skip:
            continue
        for y_col in year_cols:
            val = r[y_col]
            if pd.isna(val):
                continue
            rows.append({**base, "YEAR": int(y_col), "VALUE": float(val)})
    return rows


def insert_rows(conn, nemo_table: str, rows: list[dict], indices: list[str]):
    if not rows:
        print(f"  No rows to insert into {nemo_table}.")
        return

    # NEMO column names
    nemo_idx_cols = [INDEX_NAME_MAP[i] for i in indices]
    cols = nemo_idx_cols + ["y", "val"]

    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {nemo_table} ({','.join(cols)}) VALUES ({placeholders})"

    data = []
    for r in rows:
        tup = [r[i] for i in indices] + [r["YEAR"], r["VALUE"]]
        data.append(tup)

    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()
    print(f"  Inserted {len(rows)} rows into {nemo_table}.")


def insert_rows_no_year(conn, nemo_table: str, rows: list[dict], indices: list[str]):
    if not rows:
        print(f"  No rows to insert into {nemo_table}.")
        return
    nemo_idx_cols = [INDEX_NAME_MAP[i] for i in indices]
    cols = nemo_idx_cols + ["val"]
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {nemo_table} ({','.join(cols)}) VALUES ({placeholders})"
    data = []
    for r in rows:
        tup = [r[i] for i in indices] + [r["VALUE"]]
        data.append(tup)
    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()
    print(f"  Inserted {len(rows)} rows into {nemo_table}.")

def insert_raw_table(conn, nemo_table: str, df: pd.DataFrame):
    """
    Raw copy helper for tables that don't fit the YEAR/VALUE pattern.
    Assumes df already has DB column names.
    """
    if df.empty:
        print(f"  No rows to insert into {nemo_table}.")
        return
    # Map verbose column names back to DB short names when possible.
    rename_map = {col: INDEX_NAME_MAP.get(col, col) for col in df.columns}
    rename_map["VALUE"] = "val"
    df = df.rename(columns=rename_map)
    cur = conn.cursor()
    cur.execute(f'DELETE FROM "{nemo_table}"')
    conn.commit()
    df.to_sql(nemo_table, conn, if_exists="append", index=False)
    print(f"  Inserted {len(df)} rows into {nemo_table} (raw copy).")

def _detect_year_table(conn) -> tuple[str | None, str | None]:
    """
    Locate the YEAR table and its year column.
    Returns (table_name, column_name) or (None, None) if not found.
    """
    cur = conn.cursor()
    for tbl in ("YEAR", "year"):
        try:
            info = cur.execute(f'PRAGMA table_info("{tbl}")').fetchall()
        except sqlite3.OperationalError:
            continue
        if not info:
            continue
        cols = [r[1] for r in info]
        col = None
        if "y" in cols:
            col = "y"
        elif "val" in cols:
            col = "val"
        elif cols:
            col = cols[0]
        return tbl, col
    return None, None


# -------------------------------------------------------------------
# Units handling
# -------------------------------------------------------------------
# Scaling factors to target units
UNIT_SCALES = {
    "energy": {
        "MWh": 1.0,
        "GWh": 1e3,
        "TWh": 1e6,
        "kWh": 1e-3,
        "PJ": 277_777.77777777775,  # 1 PJ = 277_777.78 MWh
        "TJ": 277.77777777777777,
        "MJ": 0.0002777777777777778,
        "J": 2.777777777777778e-10,
    },
    "power": {
        "MW": 1.0,
        "GW": 1e3,
        "kW": 1e-3,
        "W": 1e-6,
    },
}

DEFAULT_TARGET_UNITS = {
    "energy": "MWh",
    "power": "MW"
}
UNITS_TO_IGNORE = ["PJ/MWh", 'PJ/PJ', "fraction of year", 'years', 'million $/GW', 'million $/PJ', 'points', 'million tonnes CO2/PJ', 'Years']

MISSING_UNITS_MAPPING = {
    'SpecifiedAnnualDemand': 'PJ'
}

def convert_units(df: pd.DataFrame, unit_type: str | None, target_unit: str, warnings: list[str], sheet_name:str|None=None) -> pd.DataFrame:
    """
    Convert numeric value columns to target_unit based on UNITS column (if present).
    Expects wide-year tables with numeric year columns, or VALUE column for non-year tables.
    """
    if unit_type is None:
        return df
    if "UNITS" not in df.columns:
        if sheet_name in MISSING_UNITS_MAPPING:
            df = df.copy()
            df["UNITS"] = MISSING_UNITS_MAPPING[sheet_name]
        else:
            return df
    scale_map = UNIT_SCALES.get(unit_type, {})
    target_scale = scale_map.get(target_unit)
    if target_scale is None:
        warnings.append(f"Unknown target unit '{target_unit}' for unit_type '{unit_type}'")
        return df

    df = df.copy()
    year_cols = [c for c in df.columns if str(c).strip().isdigit()]
    value_cols = year_cols if year_cols else (["VALUE"] if "VALUE" in df.columns else [])
    if not value_cols:
        return df

    for idx, row in df.iterrows():
        unit = str(row.get("UNITS", "")).strip()
        if not unit and sheet_name in MISSING_UNITS_MAPPING:
            unit = MISSING_UNITS_MAPPING[sheet_name]
        if not unit:
            continue
        if unit in UNITS_TO_IGNORE:
            continue
        scale = scale_map.get(unit)
        if scale is None:
            warnings.append(f"Unknown unit '{unit}' for unit_type '{unit_type}' in row {idx}")
            continue
        factor = scale / target_scale
        for col in value_cols:
            try:
                val = row[col]
                if pd.isna(val):
                    continue
                df.at[idx, col] = float(val) * factor
            except Exception:
                continue
    return df


def _fetch_years(conn) -> list[int]:
    """Return all years from the YEAR table, if present."""
    tbl, col = _detect_year_table(conn)
    if not tbl or not col:
        return []
    try:
        df_years = pd.read_sql_query(f'SELECT "{col}" as y FROM "{tbl}"', conn)
        years = sorted(int(y) for y in df_years["y"].dropna().unique())
        return years
    except Exception:
        return []


def _table_to_wide(
    conn,
    table_name: str,
    sheet_name: str,
    indices_verbose: list[str],
    all_years: list[int],
    max_rows: int | None,
):
    """Convert a narrow (YEAR, VALUE) table to wide with years as columns."""
    query = f'SELECT * FROM "{table_name}"'
    if max_rows is not None:
        query += f" LIMIT {int(max_rows)}"

    try:
        df = pd.read_sql_query(query, conn)
    except Exception as exc:
        print(f"  Skipping '{sheet_name}' (table '{table_name}'): {exc}")
        return pd.DataFrame(columns=indices_verbose + [str(y) for y in all_years])

    if df.empty:
        # Empty table -> return headers only.
        return pd.DataFrame(columns=indices_verbose + [str(y) for y in all_years])

    rename_map = {"y": "YEAR", "val": "VALUE"}
    for idx in indices_verbose:
        short = INDEX_NAME_MAP[idx]
        rename_map[short] = idx
    df = df.rename(columns=rename_map)

    missing_idx = [idx for idx in indices_verbose if idx not in df.columns]
    if missing_idx:
        print(f"  Skipping '{sheet_name}': missing columns {missing_idx} in DB.")
        return pd.DataFrame(columns=indices_verbose + [str(y) for y in all_years])

    if "YEAR" not in df.columns or "VALUE" not in df.columns:
        print(f"  Skipping '{sheet_name}': missing YEAR/VALUE columns.")
        return pd.DataFrame(columns=indices_verbose + [str(y) for y in all_years])

    # Coerce YEAR to numeric to avoid string/int comparison errors
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    years_from_df = {int(y) for y in df["YEAR"].dropna().unique()}
    years = sorted(set(all_years) | years_from_df)
    if not years:
        return pd.DataFrame(columns=indices_verbose)

    wide = (
        df.pivot_table(
            index=indices_verbose, columns="YEAR", values="VALUE", aggfunc="first"
        )
        .reset_index()
    )

    # Ensure all years are present as columns and ordered.
    for y in years:
        if y not in wide.columns:
            wide[y] = pd.NA
    ordered_cols = indices_verbose + [y for y in years]
    wide = wide[ordered_cols]

    # Convert year columns to string labels for Excel friendliness.
    wide.columns = [str(c) if isinstance(c, (int, float)) else c for c in wide.columns]
    return wide

def dump_db_to_entry_excel(
    db_path: Path,
    excel_path: Path,
    specs: dict,
    tables: list[str] | None = None,
    max_rows: int | None = None,
    use_advanced: Mapping[str, bool] | None = None,
):
    """
    Export DB to an Excel workbook for data entry:
    - One sheet per spec in PARAM_SPECS (respecting USE_ADVANCED and filter).
    - Verbose column names (REGION, TECHNOLOGY, ...).
    - Years pivoted into columns.
    """
    use_advanced = use_advanced or {}
    conn = sqlite3.connect(db_path)
    all_years = _fetch_years(conn)

    allowed = set(tables) if tables else None

    used_sheet_names = set()

    def unique_sheet_name(base: str) -> str:
        candidate = base[:31]  # Excel sheet name limit
        counter = 1
        while candidate in used_sheet_names:
            suffix = f"_{counter}"
            candidate = (base[: 31 - len(suffix)] + suffix)[:31]
            counter += 1
        used_sheet_names.add(candidate)
        return candidate

    with pd.ExcelWriter(excel_path) as writer:
        for sheet_name, spec in specs.items():
            adv_flag = spec.get("advanced_flag", None)
            if adv_flag is not None and not use_advanced.get(adv_flag, False):
                continue

            table_name = spec["nemo_table"]
            if allowed and table_name not in allowed and sheet_name not in allowed:
                continue

            indices_verbose = spec["indices"]
            if spec.get("raw"):
                try:
                    wide_df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                    rename_map = {k: INDEX_NAME_MAP_REV.get(k, k) for k in wide_df.columns}
                    wide_df = wide_df.rename(columns=rename_map)
                except Exception:
                    wide_df = pd.DataFrame()
            elif not spec.get("has_years", True):
                try:
                    wide_df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                    rename_map = {k: INDEX_NAME_MAP_REV.get(k, k) for k in wide_df.columns}
                    wide_df = wide_df.rename(columns=rename_map)
                    if "VALUE" not in wide_df.columns and "val" in wide_df.columns:
                        wide_df = wide_df.rename(columns={"val": "VALUE"})
                except Exception:
                    wide_df = pd.DataFrame()
            else:
                wide_df = _table_to_wide(
                    conn,
                    table_name=table_name,
                    sheet_name=sheet_name,
                    indices_verbose=indices_verbose,
                    all_years=all_years,
                    max_rows=max_rows,
                )

            out_sheet = unique_sheet_name(sheet_name)
            wide_df.to_excel(writer, sheet_name=out_sheet, index=False)

    conn.close()
    print(f"Wrote {len(used_sheet_names)} sheets to '{excel_path}'.")

def populate_year_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    tbl_year, col_year = _detect_year_table(conn)
    if not tbl_year or not col_year:
        print("No YEAR/year table found; skipping.")
        return

    # Get all distinct years from all param tables you use.
    # Simple approach: union of YEARs from SpecifiedAnnualDemand and AvailabilityFactor etc.
    year_sources = [
        ("SpecifiedAnnualDemand", "y"),
        ("AvailabilityFactor", "y"),
        ("CapitalCost", "y"),
        ("FixedCost", "y"),
        ("ResidualCapacity", "y"),
        ("EmissionActivityRatio", "y"),
        ("ReserveMargin", "y"),
        ("AnnualEmissionLimit", "y"),
    ]

    years = set()
    for table, col in year_sources:
        try:
            rows = cur.execute(f'SELECT DISTINCT {col} FROM "{table}"').fetchall()
            for (y,) in rows:
                if y is not None:
                    years.add(int(y))
        except sqlite3.OperationalError:
            # Table may not exist or may be empty – skip
            continue

    if not years:
        print("No years found in parameter tables; YEAR table will remain empty.")
        return

    # Clear existing rows in YEAR table and insert unique years
    cur.execute(f'DELETE FROM "{tbl_year}"')

    for y in sorted(years):
        cur.execute(f'INSERT INTO "{tbl_year}" ("{col_year}") VALUES (?)', (y,))

    conn.commit()
    print(f"Inserted {len(years)} distinct years into YEAR table.")


def _insert_set_values(conn: sqlite3.Connection, table: str, values: set[str]):
    """
    Insert distinct set values into a set table (REGION, TECHNOLOGY, etc.).
    """
    if not values:
        return
    cur = conn.cursor()
    data = [(v,) for v in sorted(values)]
    cur.executemany(f'INSERT OR IGNORE INTO "{table}" (val) VALUES (?)', data)
    conn.commit()
    print(f"Inserted {len(values)} values into {table}.")


def _backfill_capacity_unit_years(conn: sqlite3.Connection):
    """
    Some templates define CapacityOfOneTechnologyUnit with a year column (y).
    When we load year-less data, y stays NULL and can break NEMO. Duplicate
    rows across all YEARS to ensure y is populated.
    """
    cur = conn.cursor()
    try:
        cols = [r[1] for r in cur.execute('PRAGMA table_info("CapacityOfOneTechnologyUnit")')]
        if "y" not in cols:
            return
    except Exception:
        return

    years = [r[0] for r in cur.execute('SELECT val FROM "YEAR"')]
    if not years:
        return

    rows = cur.execute(
        'SELECT id, r, t, val FROM "CapacityOfOneTechnologyUnit" WHERE y IS NULL'
    ).fetchall()
    if not rows:
        return

    cur.execute('DELETE FROM "CapacityOfOneTechnologyUnit" WHERE y IS NULL')
    new_rows = []
    for _, r, t, val in rows:
        for y in years:
            try:
                v = float(val)
            except Exception:
                v = val
            new_rows.append((r, t, y, v))
    cur.executemany(
        'INSERT INTO "CapacityOfOneTechnologyUnit" (r, t, y, val) VALUES (?, ?, ?, ?)',
        new_rows,
    )
    conn.commit()
    print(f"Backfilled CapacityOfOneTechnologyUnit for years; added {len(new_rows)} rows.")

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def convert_osemosys_input_to_nemo(config: Mapping[str, Any]):
    """
    Populate a NEMO scenario database from an OSeMOSYS-style Excel workbook.

    The config mapping is expected to provide:
      - INPUT_MODE: "osemosys" or "nemo_entry"
      - OSEMOSYS_EXCEL_PATH / NEMO_ENTRY_EXCEL_PATH: source workbooks
      - TEMPLATE_DB: empty NEMO template database
      - OUTPUT_DB: destination database that will be overwritten
      - SCENARIO: scenario label used when filtering sheets
      - USE_ADVANCED: dict of feature flags (e.g., {"ReserveMargin": True})
      - EXPORT_DB_TO_EXCEL: whether to dump the populated DB back to Excel
      - EXPORT_EXCEL_PATH: target path for the dump if enabled
      - EXPORT_TABLE_FILTER / EXPORT_MAX_ROWS: optional export limits
    """
    required_keys = [
        "INPUT_MODE",
        "OSEMOSYS_EXCEL_PATH",
        "NEMO_ENTRY_EXCEL_PATH",
        "TEMPLATE_DB",
        "OUTPUT_DB",
        "SCENARIO",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    input_mode = str(config["INPUT_MODE"]).lower()
    if input_mode not in {"osemosys", "nemo_entry"}:
        raise ValueError("INPUT_MODE must be either 'osemosys' or 'nemo_entry'.")

    osemosys_excel_path = Path(config["OSEMOSYS_EXCEL_PATH"])
    nemo_entry_excel_path = Path(config["NEMO_ENTRY_EXCEL_PATH"])
    template_db = Path(config["TEMPLATE_DB"])
    output_db = Path(config["OUTPUT_DB"])
    SCENARIO = str(config["SCENARIO"])
    use_advanced = dict(config.get("USE_ADVANCED") or {})
    target_units_cfg = dict(config.get("TARGET_UNITS") or {})
    target_energy_unit = target_units_cfg.get("energy", DEFAULT_TARGET_UNITS["energy"])
    target_power_unit = target_units_cfg.get("power", DEFAULT_TARGET_UNITS["power"])
    strict_errors = bool(config.get("STRICT_ERRORS", False))
    export_db_to_excel = bool(config.get("EXPORT_DB_TO_EXCEL", False))
    export_excel_path = Path(
        config.get("EXPORT_EXCEL_PATH", output_db.with_suffix(".xlsx"))
    )
    export_table_filter = config.get("EXPORT_TABLE_FILTER", None)
    export_max_rows = config.get("EXPORT_MAX_ROWS", None)

    excel_path = osemosys_excel_path if input_mode == "osemosys" else nemo_entry_excel_path

    print(f"Reading input workbook ({input_mode}) from '{excel_path}'.")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    copy_template_db(template_db, output_db)
    conn = sqlite3.connect(output_db)

    # Collect set values while processing sheets
    collected_sets: DefaultDict[str, set[str]] = defaultdict(set)
    collected_years: set[int] = set()
    warnings: list[str] = []

    # Populate simple set tables directly from VALUE-based sheets (if present)
    for set_sheet, target_table in SET_SHEETS.items():
        df_set = load_sheet(excel_path, set_sheet)
        if df_set.empty:
            continue
        if "VALUE" not in df_set.columns:
            print(f"Skipping set sheet '{set_sheet}' (no VALUE column).")
            continue
        vals = {str(v) for v in df_set["VALUE"].dropna().unique().tolist()}
        if not vals:
            continue
        _insert_set_values(conn, target_table, vals)
        collected_sets[set_sheet].update(vals)
        if set_sheet == "YEAR":
            collected_years.update(
                int(v) for v in vals if str(v).strip().isdigit()
            )

    for sheet_name, spec in PARAM_SPECS.items():
        nemo_table = spec["nemo_table"]
        indices = spec["indices"]
        filter_scen = spec.get("filter_scenario", False)
        adv_flag = spec.get("advanced_flag", None)

        # If this is an advanced param and its switch is off, skip
        if adv_flag is not None and not use_advanced.get(adv_flag, False):
            print(
                f"\nSkipping advanced parameter '{sheet_name}' (flag {adv_flag}=False)."
            )
            continue

        print(f"\nProcessing sheet '{sheet_name}' -> table '{nemo_table}'")

        df = load_sheet(excel_path, sheet_name)
        if df.empty:
            print("  Sheet empty or missing. Skipping.")
            continue

        defaults_map = dict(spec.get("defaults", {}) or {})
        has_years = spec.get("has_years", True)
        is_raw = bool(spec.get("raw", False))

        # Filter scenario if needed
        if filter_scen:
            df = filter_scenario(df, SCENARIO)
            if df.empty:
                print(f"  No rows for scenario '{SCENARIO}'. Skipping.")
                continue

        # Simple renames for val->VALUE when has_years=False (non-raw tables)
        if (not spec.get("raw", False)) and (not has_years) and "VALUE" not in df.columns and "val" in df.columns:
            df = df.rename(columns={"val": "VALUE"})

        # Check indices exist (skip for raw tables)
        if not is_raw:
            missing = [i for i in indices if i not in df.columns and i not in defaults_map]
            if missing:
                print(f"  Missing required columns {missing} in '{sheet_name}'. Skipping.")
                continue

        # Fill optional/defaulted columns that are absent in the sheet
        for idx, default_val in defaults_map.items():
            if idx not in df.columns:
                df[idx] = default_val

        # Normalize MODE_OF_OPERATION values where present
        if "MODE_OF_OPERATION" in df.columns:
            df["MODE_OF_OPERATION"] = df["MODE_OF_OPERATION"].apply(_normalize_mode_value)

        if is_raw:
            # For raw tables, just copy rows directly.
            insert_raw_table(conn, nemo_table, df)
            continue

        # Warn about missing index values that will be skipped
        for idx in indices:
            if df[idx].isna().any():
                warnings.append(
                    f"{sheet_name}: {df[idx].isna().sum()} rows missing {idx} will be skipped."
                )

        # Optional unit conversion
        unit_type = spec.get("unit_type")
        if unit_type == "energy":
            df = convert_units(df, "energy", target_energy_unit, warnings, sheet_name=sheet_name)
        elif unit_type == "power":
            df = convert_units(df, "power", target_power_unit, warnings, sheet_name=sheet_name)

        if has_years:
            # Identify year columns
            year_cols = get_year_columns(df.columns)
            if not year_cols:
                print(f"  No year columns found in '{sheet_name}'. Skipping.")
                continue

            # Build rows & insert
            rows = build_rows_from_wide(df, indices, year_cols, defaults=spec.get("defaults"))
            insert_rows(conn, nemo_table, rows, indices)
            collected_years.update(int(y) for y in year_cols)
        else:
            # Expect a VALUE column only
            if "VALUE" not in df.columns:
                print(f"  No VALUE column found in '{sheet_name}'. Skipping.")
                continue
            # Normalize index values to string
            df_indices = df[indices].copy()
            for idx in indices:
                df_indices[idx] = df_indices[idx].astype(str)
            rows = []
            for _, row in df_indices.iterrows():
                rows.append({idx: row[idx] for idx in indices} | {"VALUE": df.loc[_, "VALUE"]})
            insert_rows_no_year(conn, nemo_table, rows, indices)

        # Track set members and years
        for idx in indices:
            vals = df[idx].dropna().unique().tolist()
            collected_sets[idx].update(str(v) for v in vals)
        if has_years:
            collected_years.update(int(y) for y in year_cols)
        
    # populate set tables (REGION/TECHNOLOGY/FUEL/TIMESLICE/EMISSION) and YEAR
    _insert_set_values(conn, "REGION", collected_sets.get("REGION", set()))
    _insert_set_values(conn, "TECHNOLOGY", collected_sets.get("TECHNOLOGY", set()))
    fuels = collected_sets.get("FUEL", set())
    if "ALL" in fuels or any(spec.get("defaults", {}).get("FUEL") == "ALL" for spec in PARAM_SPECS.values()):
        fuels.add("ALL")
    _insert_set_values(conn, "FUEL", fuels)
    _insert_set_values(conn, "TIMESLICE", collected_sets.get("TIMESLICE", set()))
    _insert_set_values(conn, "EMISSION", collected_sets.get("EMISSION", set()))
    _insert_set_values(conn, "MODE_OF_OPERATION", collected_sets.get("MODE_OF_OPERATION", set()))

    # populate YEAR table based on years found in parameter tables
    if collected_years:
        cur = conn.cursor()
        cur.execute('DELETE FROM "YEAR"')
        cur.executemany('INSERT INTO "YEAR" (val) VALUES (?)', [(y,) for y in sorted(collected_years)])
        conn.commit()
        print(f"Inserted {len(collected_years)} distinct years into YEAR table.")
    else:
        populate_year_table(conn)

    _backfill_capacity_unit_years(conn)

    # Cross-check modes and YearSplit consistency
    cur = conn.cursor()
    iar_pairs = {(t, m) for t, m in cur.execute('select distinct t, m from InputActivityRatio')}
    oar_pairs = {(t, m) for t, m in cur.execute('select distinct t, m from OutputActivityRatio')}
    iar_techs = {t for t, _ in iar_pairs}
    oar_techs = {t for t, _ in oar_pairs}

    missing_oar = iar_techs - oar_techs
    # Supply-only technologies (fuel producers/imports) are allowed, so OAR without IAR is OK.
    missing_iar = oar_techs - iar_techs

    if missing_oar and strict_errors:
        breakpoint()#
        raise Exception(f"Technologies present in InputActivityRatio but missing in OutputActivityRatio: {sorted(missing_oar)}")

    if missing_oar:
        warnings.append(
            f"{len(missing_oar)} technologies present in InputActivityRatio but missing in OutputActivityRatio (sample {sorted(missing_oar)[:5]})"
        )
    if missing_iar:
        warnings.append(
            f"{len(missing_iar)} technologies present in OutputActivityRatio but missing in InputActivityRatio (supply-only techs? sample {sorted(missing_iar)[:5]})"
        )
    used_modes = {m for _, m in (iar_pairs | oar_pairs) if m is not None}
    mode_set = {r[0] for r in cur.execute('select val from MODE_OF_OPERATION')}
    extra_modes = sorted(used_modes - mode_set)
    if extra_modes:
        warnings.append(f"MODE_OF_OPERATION set missing values: {extra_modes}")
    bad_yearsplits = cur.execute(
        "select y, sum(val) as s from YearSplit group by y having abs(s-1.0) > 1e-6"
    ).fetchall()
    if bad_yearsplits:
        warnings.append(f"YearSplit rows do not sum to 1 for years: {bad_yearsplits[:5]}")

    if warnings:
        print("\nWarnings detected during conversion:")
        for w in warnings:
            print("  -", w)

    conn.close()
    print(f"\nDone. NEMO scenario DB written to '{output_db}'.")

    if export_db_to_excel:
        dump_db_to_entry_excel(
            output_db,
            export_excel_path,
            specs=PARAM_SPECS,
            tables=export_table_filter,
            max_rows=export_max_rows,
            use_advanced=use_advanced,
        )


#%%
# if __name__ == "__main__":
#     convert_osemosys_input_to_nemo()
#%%
