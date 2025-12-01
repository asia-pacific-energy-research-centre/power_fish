#%%
from pathlib import Path

from convert_osemosys_input_to_nemo import convert_osemosys_input_to_nemo
from run_nemo_via_julia import run_nemo_on_db
from diagnostics import run_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------------------------
# USER CONFIGURATION
# -------------------------------------------------------------------
VARS = {
    "INPUT_MODE": "osemosys",  # "osemosys" or "nemo_entry"
    "OSEMOSYS_EXCEL_PATH": DATA_DIR / "POWER 20_USA_data_REF9_S3_test.xlsx",
    "NEMO_ENTRY_EXCEL_PATH": DATA_DIR / "usa_power_nemo_dump.xlsx",
    "TEMPLATE_DB": DATA_DIR / "nemo_template.sqlite",
    "OUTPUT_DB": DATA_DIR / "usa_power_nemo.sqlite",
    "SCENARIO": "Reference",
    "USE_ADVANCED": {
        "ReserveMargin": False,
        "AnnualEmissionLimit": False,
    },
    # Export the populated NEMO DB to an Excel workbook in the NEMO format (one sheet per table).
    "EXPORT_DB_TO_EXCEL": True,
    "EXPORT_EXCEL_PATH": DATA_DIR / "usa_power_nemo_dump.xlsx",
    # Restrict export to a subset of tables/sheets (None -> all in PARAM_SPECS).
    "EXPORT_TABLE_FILTER": None,
    # Limit rows per table when exporting (None -> all rows).
    "EXPORT_MAX_ROWS": None,
    # Optional: set to the Julia executable path; if None, will look at env JULIA_EXE/NEMO_JULIA_EXE or PATH.
    "JULIA_EXE": r"C:\ProgramData\Julia\Julia-1.9.3\bin\julia.exe",  # None,
    # Diagnostics
    "RUN_DIAGNOSTICS": True,
    "DIAGNOSTICS_YEARS": [2017, 2018],  # e.g., [2017, 2018] to narrow horizon
    "DIAGNOSTICS_WRITE_TRIMMED": DATA_DIR / "usa_power_nemo_trim.sqlite",#None,  # optional path to write trimmed DB for testing
    "AUTO_FILL_MISSING_MODES": True,
    
}
CONVERT_OSEMOSYS_INPUT_TO_NEMO = True
RUN_NEMO = True

def main():
    if CONVERT_OSEMOSYS_INPUT_TO_NEMO:
        convert_osemosys_input_to_nemo(VARS)
        if VARS.get("RUN_DIAGNOSTICS"):
            run_diagnostics(
                VARS["OUTPUT_DB"],
                years=VARS.get("DIAGNOSTICS_YEARS"),
                write_trimmed=VARS.get("DIAGNOSTICS_WRITE_TRIMMED"),
            )
    if RUN_NEMO:
        run_nemo_on_db(
            VARS["OUTPUT_DB"],
            julia_exe=VARS.get("JULIA_EXE"),
            log_path=DATA_DIR / "nemo_run.log",
        )
        

if __name__ == "__main__":
    main()
#%%


####################################################
#scratch code to inspect the output db:
# import sqlite3, pandas as pd
# db = VARS["OUTPUT_DB"]
# conn = sqlite3.connect(db)
# print(pd.read_sql("SELECT * FROM RunLog", conn).head())
# print(pd.read_sql("SELECT * FROM vtotalcapacityannual LIMIT 5", conn))
# conn.close()

# import sqlite3, pandas as pd
# db = r"../../data/usa_power_nemo.sqlite"
# with sqlite3.connect(db) as conn:
#     tables = pd.read_sql(
#         "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
#     )
#     # Option A: show all rows
#     with pd.option_context('display.max_rows', None):
#         print(tables)
#     # Option B: simple list
#     print(tables['name'].to_list())

#to get views:
# SELECT name FROM sqlite_master WHERE name LIKE 'v%';

####################################################

#all tables in the nemo db:

# tables = pd.read_sql(
#         "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
#     )
# ['AccumulatedAnnualDemand', 'AnnualEmissionLimit', 'AnnualExogenousEmission', 'AvailabilityFactor', 'CapacityOfOneTechnologyUnit', 'CapacityToActivityUnit', 'CapitalCost', 'CapitalCostStorage', 'DefaultParams', 'DepreciationMethod', 'DiscountRate', 'EMISSION', 'EmissionActivityRatio', 'EmissionsPenalty', 'FUEL', 'FixedCost', 'InputActivityRatio', 'InterestRateStorage', 'InterestRateTechnology', 'LTsGroup', 'MODE_OF_OPERATION', 'MaxAnnualTransmissionNodes', 'MinAnnualTransmissionNodes', 'MinShareProduction', 'MinStorageCharge', 'MinimumUtilization', 'ModelPeriodEmissionLimit', 'ModelPeriodExogenousEmission', 'NODE', 'NodalDistributionDemand', 'NodalDistributionStorageCapacity', 'NodalDistributionTechnologyCapacity', 'OperationalLife', 'OperationalLifeStorage', 'OutputActivityRatio', 'REGION', 'REGIONGROUP', 'REMinProductionTarget', 'REMinProductionTargetRG', 'RETagTechnology', 'RRGroup', 'RampRate', 'RampingReset', 'ReserveMargin', 'ReserveMarginTagTechnology', 'ResidualCapacity', 'ResidualStorageCapacity', 'STORAGE', 'SpecifiedAnnualDemand', 'SpecifiedDemandProfile', 'StorageFullLoadHours', 'StorageLevelStart', 'StorageMaxChargeRate', 'StorageMaxDischargeRate', 'TECHNOLOGY', 'TIMESLICE', 'TSGROUP1', 'TSGROUP2', 'TechnologyFromStorage', 'TechnologyToStorage', 'TotalAnnualMaxCapacity', 'TotalAnnualMaxCapacityInvestment', 'TotalAnnualMaxCapacityInvestmentStorage', 'TotalAnnualMaxCapacityStorage', 'TotalAnnualMinCapacity', 'TotalAnnualMinCapacityInvestment', 'TotalAnnualMinCapacityInvestmentStorage', 'TotalAnnualMinCapacityStorage', 'TotalTechnologyAnnualActivityLowerLimit', 'TotalTechnologyAnnualActivityUpperLimit', 'TotalTechnologyModelPeriodActivityLowerLimit', 'TotalTechnologyModelPeriodActivityUpperLimit', 'TradeRoute', 'TransmissionAvailabilityFactor', 'TransmissionCapacityToActivityUnit', 'TransmissionLine', 'TransmissionModelingEnabled', 'VariableCost', 'Version', 'YEAR', 'YearSplit', 'nodalstorage', 'sqlite_sequence', 'yearintervals']


####################################################