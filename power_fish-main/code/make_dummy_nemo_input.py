
"""
Create a minimal OSeMOSYS-style Excel workbook that can be converted to a
NEMO scenario DB for quick smoke tests.

Usage:
    python code/make_dummy_nemo_input.py [output_path]

By default it writes to data/dummy_osemosys.xlsx.
"""
#%%
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def make_dummy_workbook(
    out_path: Path,
    *,
    scenario: str = "Reference",
    years: list[int] | None = None,
    region: str = "R1",
    timeslice: str = "ANNUAL",
    gen_tech: str = "GEN_GAS",
    supply_tech: str = "GAS_SUPPLY",  # unused in simplified dummy
    input_fuel: str = "GAS",  # unused in simplified dummy
    output_fuel: str = "ELEC",
    mode: str = "1",
):
    years = years or [2017, 2018]
    techs = [gen_tech]
    fuels = [output_fuel]

    # Set sheets (VALUE column)
    set_frames = {
        "REGION": pd.DataFrame({"VALUE": [region]}),
        "TECHNOLOGY": pd.DataFrame({"VALUE": techs}),
        "FUEL": pd.DataFrame({"VALUE": fuels}),
        "TIMESLICE": pd.DataFrame({"VALUE": [timeslice]}),
        "EMISSION": pd.DataFrame({"VALUE": []}),  # empty but present
        "MODE_OF_OPERATION": pd.DataFrame({"VALUE": [mode]}),
        "YEAR": pd.DataFrame({"VALUE": years}),
    }

    def wide(df_dict: dict[str, list], value_per_year):
        """
        Attach per-year columns. Accepts either a scalar or a list/tuple with one
        value per year in the `years` list.
        """
        df = pd.DataFrame(df_dict)
        for idx, y in enumerate(years):
            if isinstance(value_per_year, (list, tuple)) and len(value_per_year) == len(years):
                val = value_per_year[idx]
            else:
                val = value_per_year
            df[str(y)] = val
        return df

    # Demand for output fuel
    specified_demand = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "FUEL": [output_fuel],
            "UNITS": ["GWh"],
        },
        [0, 0],  # zero demand to keep dummy trivially feasible
    )

    # InputActivityRatio omitted (fuel-free generator)
    iar = pd.DataFrame(columns=["SCENARIO", "REGION", "TECHNOLOGY", "FUEL", "MODE_OF_OPERATION"])
    oar = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "FUEL": [output_fuel],
            "MODE_OF_OPERATION": [mode],
            "UNITS": ["PJ/PJ"],
        },
        [1.0, 1.0],
    )

    # Simple capacity/availability and costs
    capacity_factor = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "TIMESLICE": [timeslice],
            "UNITS": ["fraction"],
        },
        [0.9, 0.9],
    )
    supply_capacity_factor = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [supply_tech],
            "TIMESLICE": [timeslice],
            "UNITS": ["fraction"],
        },
        [1.0, 1.0],
    )
    capital_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["$/kW"],
        },
        [1200, 1100],
    )
    supply_capital_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [supply_tech],
            "UNITS": ["$/kW"],
        },
        [0, 0],
    )
    fixed_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["$/kW-yr"],
        },
        [30, 30],
    )
    variable_cost = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "MODE_OF_OPERATION": [mode],
            "UNITS": ["$/MWh"],
        },
        [10, 10],
    )

    # Tables without year columns (use VALUE)
    capacity_per_unit = pd.DataFrame(
        {
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW/unit"],
            "VALUE": [200],
        }
    )
    capacity_to_activity = pd.DataFrame(
        {
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["GWh/MWyr"],
            "VALUE": [8.76],
        }
    )

    # YearSplit: single timeslice sums to 1
    yearsplit = wide(
        {
            "TIMESLICE": [timeslice],
            "UNITS": ["fraction"],
        },
        [1.0, 1.0],
    )

    total_annual_max_capacity = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW"],
        },
        [1_000_000_000, 1_000_000_000],  # effectively unconstrained
    )
    residual_capacity = wide(
        {
            "SCENARIO": [scenario],
            "REGION": [region],
            "TECHNOLOGY": [gen_tech],
            "UNITS": ["MW"],
        },
        [1000, 1000],  # seed enough capacity to satisfy demand
    )

    sheets: dict[str, pd.DataFrame] = {
        **set_frames,
        "SpecifiedAnnualDemand": specified_demand,
        "InputActivityRatio": iar,
        "OutputActivityRatio": oar,
        # Converter expects this sheet name to populate AvailabilityFactor table
        "CapacityFactor": capacity_factor,
        "VariableCost": variable_cost,
        "CapitalCost": capital_cost,
        "FixedCost": fixed_cost,
        # Seed residual capacity and give a finite max cap so capacity binds correctly.
        "ResidualCapacity": residual_capacity,
        "TotalAnnualMaxCapacity": total_annual_max_capacity,
        "CapacityOfOneTechnologyUnit": capacity_per_unit,
        "CapacityToActivityUnit": capacity_to_activity,
        "YearSplit": yearsplit,
    }
    # Include both names to avoid sheet-name mismatches
    sheets["AvailabilityFactor"] = capacity_factor.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Wrote dummy workbook to {out_path}")


def main(out_path: str | Path | None = None):
    """
    Entrypoint usable both from CLI and notebooks.
    - If a path argument is provided, use it.
    - Otherwise, default to ../data/dummy_osemosys.xlsx relative to cwd (works in notebooks).
    """
    default_path = Path("../data/dummy_osemosys.xlsx")

    # If caller passed a notebook/kernel flag like "--f=...", ignore it.
    if out_path is not None and str(out_path).startswith("-"):
        out_path = None

    if out_path is None:
        cli_paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
        out_path = cli_paths[0] if cli_paths else default_path

    make_dummy_workbook(Path(out_path))

#%%
if __name__ == "__main__":
    main()
#%%
