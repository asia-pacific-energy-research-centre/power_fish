"""
Lightweight diagnostics for NEMO DB inputs.

Usage examples:
  # Show quick health report
  python diagnostics.py --db data/usa_power_nemo.sqlite

  # Check only a subset of years
  python diagnostics.py --db data/usa_power_nemo.sqlite --years 2017,2018

  # Create a trimmed DB with only selected years (for faster test runs)
  python diagnostics.py --db data/usa_power_nemo.sqlite --years 2017,2018 --write-trimmed data/usa_power_nemo_trim.sqlite
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def parse_years(arg: str | None) -> list[int] | None:
    if not arg:
        return None
    out = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def load_df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    except Exception:
        return pd.DataFrame()


def trim_years(conn: sqlite3.Connection, years: Sequence[int], tables: Iterable[str]):
    years_set = set(int(y) for y in years)
    cur = conn.cursor()
    for table in tables:
        try:
            info = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
        except Exception:
            continue
        cols = [c[1] for c in info]
        if "y" not in cols:
            continue
        cur.execute(f'DELETE FROM "{table}" WHERE y NOT IN ({",".join("?"*len(years_set))})', tuple(years_set))
    # YEAR set table
    cur.execute('DELETE FROM "YEAR" WHERE val NOT IN ({})'.format(",".join("?"*len(years_set))), tuple(years_set))
    conn.commit()


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def run_diagnostics(db_path: str | Path, years: list[int] | None = None, write_trimmed: str | Path | None = None):
    db_path = Path(db_path)
    work_db = db_path
    if write_trimmed and years:
        work_db = Path(write_trimmed)
        shutil.copyfile(db_path, work_db)
        with sqlite3.connect(work_db) as conn:
            trim_years(conn, years, tables=[])
        print(f"Wrote trimmed DB to {work_db}")

    with sqlite3.connect(work_db) as conn:
        # Sets
        sets = {}
        for tbl in ["REGION", "TECHNOLOGY", "FUEL", "TIMESLICE", "EMISSION", "YEAR", "MODE_OF_OPERATION"]:
            df_set = load_df(conn, tbl)
            sets[tbl] = set(df_set["val"].astype(str)) if not df_set.empty else set()

        print_section("Years and demand coverage")
        sad = load_df(conn, "SpecifiedAnnualDemand")
        if sad.empty:
            print("SpecifiedAnnualDemand: empty")
        else:
            if years:
                sad = sad[sad["y"].astype(int).isin(years)]
            print(f"Demand rows: {len(sad)}; regions: {sorted(sad['r'].unique())}; fuels: {sorted(sad['f'].unique())}; years: {sorted(sad['y'].unique())[:10]}")

        print_section("Supply tech presence")
        af = load_df(conn, "AvailabilityFactor")
        if af.empty:
            print("AvailabilityFactor: empty")
        else:
            if years:
                af = af[af["y"].astype(int).isin(years)]
            print(f"Techs with availability factors: {len(sorted(af['t'].unique()))}; sample: {sorted(af['t'].unique())[:5]}")

        print_section("Input/Output activity ratios")
        iar = load_df(conn, "InputActivityRatio")
        oar = load_df(conn, "OutputActivityRatio")
        if years:
            if not iar.empty and "y" in iar.columns:
                iar = iar[iar["y"].astype(int).isin(years)]
            if not oar.empty and "y" in oar.columns:
                oar = oar[oar["y"].astype(int).isin(years)]
        if iar.empty:
            print("InputActivityRatio: empty (no fuel inputs defined!)")
        else:
            modes = sorted(set(str(m) for m in iar["m"].dropna().unique()))
            fuels = sorted(set(str(f) for f in iar["f"].dropna().unique()))
            print(f"IAR rows: {len(iar)}; techs: {len(set(iar['t']))}; fuels: {fuels[:5]}; modes: {modes}")
        if oar.empty:
            print("OutputActivityRatio: empty (no fuel outputs defined!)")
        else:
            modes = sorted(set(str(m) for m in oar["m"].dropna().unique()))
            fuels = sorted(set(str(f) for f in oar["f"].dropna().unique()))
            print(f"OAR rows: {len(oar)}; techs: {len(set(oar['t']))}; fuels: {fuels[:5]}; modes: {modes}")

        print_section("Demand fuels coverage by output ratios")
        if not sad.empty and not oar.empty:
            demand_fuels = set(sad["f"])
            oar_fuels = set(oar["f"])
            missing_fuels = sorted(demand_fuels - oar_fuels)
            if missing_fuels:
                print(f"Demand fuels with no OutputActivityRatio: {missing_fuels[:10]}")
            else:
                print("All demand fuels appear in OutputActivityRatio.")

        print_section("Reserve margin status")
        rm = load_df(conn, "ReserveMargin")
        if rm.empty:
            print("ReserveMargin: empty (or disabled).")
        else:
            if years:
                rm = rm[rm["y"].astype(int).isin(years)]
            fuels = sorted(set(str(f) for f in rm["f"].unique()))
            print(f"ReserveMargin rows: {len(rm)}; fuels tags: {fuels[:10]}")

        print_section("Emission limits and ratios")
        ael = load_df(conn, "AnnualEmissionLimit")
        if ael.empty:
            print("AnnualEmissionLimit: empty (or disabled).")
        else:
            if years:
                ael = ael[ael["y"].astype(int).isin(years)]
            print(f"AnnualEmissionLimit rows: {len(ael)}; emissions: {sorted(ael['e'].unique())[:5]}")
        ear = load_df(conn, "EmissionActivityRatio")
        if ear.empty:
            print("EmissionActivityRatio: empty!")
        else:
            modes = sorted(set(str(m) for m in ear["m"].dropna().unique()))
            print(f"EmissionActivityRatio rows: {len(ear)}; modes present: {modes}")

        print_section("Basic conflict checks")
        # Demand years vs availability years
        if not sad.empty and not af.empty:
            demand_years = set(int(y) for y in sad["y"].unique())
            avail_years = set(int(y) for y in af["y"].unique())
            missing_years = sorted(demand_years - avail_years)
            if missing_years:
                print(f"No availability factors for demand years: {missing_years[:10]}")
            else:
                print("Availability factors cover all demand years.")

        # Demand fuels vs output fuels
        if not sad.empty and not oar.empty:
            demand_fuels = set(sad["f"])
            oar_fuels = set(oar["f"])
            missing_fuels = sorted(demand_fuels - oar_fuels)
            if missing_fuels:
                print(f"WARNING: Demand fuels missing in OutputActivityRatio: {missing_fuels[:10]}")

        # Emission coverage: if limits exist but no emission ratios
        if not ael.empty and ear.empty:
            print("WARNING: Emission limits present but no EmissionActivityRatio rows.")

        # Modes coverage
        if not ear.empty:
            modes_set = sets.get("MODE_OF_OPERATION", set())
            if modes_set and not set(modes_set) >= set(str(m) for m in ear["m"].dropna().unique()):
                print("WARNING: EmissionActivityRatio modes not all in MODE_OF_OPERATION set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to NEMO scenario sqlite")
    parser.add_argument("--years", help="Comma list of years to keep/check (e.g., 2017,2018)")
    parser.add_argument(
        "--write-trimmed",
        help="Optional path to write a copy of the DB trimmed to the selected years",
    )
    args = parser.parse_args()
    years = parse_years(args.years)
    run_diagnostics(db_path=args.db, years=years, write_trimmed=args.write_trimmed)


if __name__ == "__main__":
    main()
