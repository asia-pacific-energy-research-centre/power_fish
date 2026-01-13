from __future__ import annotations

from pathlib import Path

# Core pipeline defaults (formerly in nemo_core.py)
# Adjust these to change the standard behavior without editing code elsewhere.
DEFAULTS = {
    # --- Main outputs / exports ---
    "EXPORT_DB_TO_EXCEL": True,  # dump populated DB back to Excel
    "EXPORT_RESULTS_TO_EXCEL": True,  # narrow results export
    "EXPORT_RESULTS_TO_EXCEL_PATH": None,
    "EXPORT_RESULTS_WIDE_TO_EXCEL": True,  # pivoted-by-year results export
    "EXPORT_RESULTS_WIDE_TO_EXCEL_PATH": None,

    # --- Plotly dashboard ---
    "PLOTLY_DASHBOARD": True,
    # Plotly dashboard driven by YAML only; path, layout, dict/function figs read from file
    "PLOTLY_CONFIG_YAML": Path("../config/plotly_charts.yml"),
    "PLOTLY_DASHBOARD_PATH": None,
    "PLOTLY_PNG_DIR": None,
    # Optional: auto-generate EmissionActivityRatio from fuel factors
    "EMISSIONS_FACTOR_CSV": Path("../config/9th_edition_emission_factors.csv"),
    "EMISSIONS_LABEL": "CO2",
    "EMISSIONS_FACTOR_OVERRIDE": True,

    # --- Paths (override in main if needed) ---
    "TEMPLATE_DB": "nemo_template.sqlite",
    "OUTPUT_DB": "nemo.sqlite",
    "NEMO_CONFIG_PATH": "nemo.cfg",
    "NEMO_TEST_DB_DIR": "nemo_tests",
    "NEMO_TEST_NAME": "storage_test",
    "TEST_OUTPUT_DB": "nemo_test.sqlite",
    "TEST_INPUT_MODE": "nemo_entry",

    # --- Units ---
    "TARGET_UNITS": {"energy": "PJ", "power": "GW"},
    "USE_UNIT_CONVERSION": False,

    # --- Transmission / demand handling ---
    "ENABLE_NEMO_TRANSMISSION_METHODS": False,
    "REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS": False,
    "TEST_DB_PATH": None,
    "TEST_INPUT_PATH": None,
    "TEST_EXPORT_EXCEL_PATH": "nemo_entry_dump.xlsx",
    "TEST_EXPORT_DB_TO_EXCEL_PATH": "nemo_entry_dump.xlsx",

    # --- Template creation ---
    "AUTO_CREATE_TEMPLATE_DB": True,

    # --- Diagnostics ---
    "RUN_DIAGNOSTICS": True,
    "AUTO_FILL_MISSING_MODES": True,
    "STRICT_ERRORS": True,

    # --- NEMO / Julia ---
    "JULIA_EXE": r"C:\\ProgramData\\Julia\\Julia-1.9.3\\bin\\julia.exe",
    "NEMO_WRITE_LP": "../intermediate_data/nemo_model_dump.lp",

    # --- Post-processing only mode ---
    "RUN_POSTPROCESS_ONLY": False,  # when True, skip conversion/NEMO and only run exports/plots on OUTPUT_DB
    "SAVE_TEST_SNAPSHOT": True,  # save latest run inputs/settings to tests/last_run

    # --- NEMO results selection ---
    # List of NemoMod result variables to save (comma-separated in Julia).
    # Set to []/None to use NemoMod defaults.
    # See VARS_TO_SAVE_AVAILABLE for common options.
    "VARS_TO_SAVE": [
        "vdemandnn", "vnewcapacity", "vtotalcapacityannual", "vproductionbytechnologyannual",
        "vproductionnn", "vusebytechnologyannual", "vusenn", "vtotaldiscountedcost",
        "vtotaltechnologyannualactivity", "vannualtechnologyemissionbymode",
        "vannualtechnologyemission", "vannualemissions", "vmodelperiodemissions",
    ],
}

# Common NemoMod result variables; use to populate VARS_TO_SAVE.
VARS_TO_SAVE_AVAILABLE = [
    # Core activity/capacity outputs
    "vdemandnn",  # demand by fuel and year
    "vnewcapacity",  # new capacity by technology and year
    "vtotalcapacityannual",  # total capacity by technology and year
    "vproductionbytechnologyannual",  # production by technology and year
    "vproductionnn",  # production by fuel and year
    "vusebytechnologyannual",  # use by technology and year
    "vusenn",  # use by fuel and year
    "vtotaldiscountedcost",  # total discounted cost by year
    "vtotaltechnologyannualactivity",  # total annual activity by technology and year
    # Emissions outputs (see NemoMod.jl docs)
    "vannualtechnologyemissionbymode",  # emissions by technology and mode (annual)
    "vannualtechnologyemission",  # emissions by technology (annual)
    "vannualemissions",  # total emissions by year
    "vmodelperiodemissions",  # total emissions over model period
]
# DefaultParams defaults for NEMO DB when sheet is missing/empty.
DEFAULT_PARAMS = [
    (1, "AccumulatedAnnualDemand", 0),
    (3, "AnnualExogenousEmission", 0),
    (5, "AvailabilityFactor", 1),
    (6, "CapacityOfOneTechnologyUnit", 0),
    (7, "CapacityToActivityUnit", 31.536),
    (8, "CapitalCost", 0),
    (9, "CapitalCostStorage", 0),
    (10, "DepreciationMethod", 1),
    (11, "DiscountRate", 0.05),
    (12, "EmissionActivityRatio", 0),
    (13, "EmissionsPenalty", 0),
    (14, "FixedCost", 0),
    (15, "InputActivityRatio", 0),
    (16, "MinStorageCharge", 0),
    (18, "ModelPeriodExogenousEmission", 0),
    (19, "OperationalLife", 1),
    (20, "OperationalLifeStorage", 99),
    (21, "OutputActivityRatio", 0),
    (24, "RETagTechnology", 0),
    (28, "ResidualCapacity", 0),
    (29, "ResidualStorageCapacity", 999),
    (30, "SpecifiedAnnualDemand", 0),
    (31, "SpecifiedDemandProfile", 0),
    (32, "StorageLevelStart", 999),
    (33, "StorageMaxChargeRate", 99),
    (34, "StorageMaxDischargeRate", 99),
    (35, "TechnologyFromStorage", 0),
    (36, "TechnologyToStorage", 0),
    (39, "TotalAnnualMinCapacity", 0),
    (40, "TotalAnnualMinCapacityInvestment", 0),
    (42, "TotalTechnologyModelPeriodActivityLowerLimit", 0),
    (44, "TradeRoute", 0),
    (45, "VariableCost", 0),
]

# LEAP template defaults (formerly in build_leap_import_template.py)
# Configure how the LEAP import workbook is generated.
LEAP_TEMPLATE_DEFAULTS = {
    "DEFAULT_SCENARIO": "Target",  # Name for the main scenario rows
    "CURRENT_ACCOUNTS_SCENARIO": "Current Accounts",  # Name for Current Accounts rows
    "DEFAULT_REGION": "Region 1",  # If None, mirror REGION_FILTER (or scenario if that is also None).
    "OUTPUT_PATH": Path("../results/leap_import_template.xlsx"),  # Where to write the LEAP import workbook

    # Existing LEAP export to copy IDs from (set to None to skip)
    "IMPORT_ID_SOURCE": Path("../data/import_files/USA_power_leap_import_REF.xlsx"),
    "IMPORT_ID_SHEET": "Export",  # Sheet name in the import file.
    "IMPORT_ID_HEADER_ROW": 2,  # Header row index (0-based) in the import file.
    "ID_CHECK_STRICT": True,  # Raise when IDs are missing after merge.
    "ID_CHECK_BREAK": True,  # breakpoint() when IDs are missing after merge.

    "NEMO_DB_PATH": Path("../data/nemo.sqlite"),  # Set to None to skip autofill from NEMO
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

def apply_defaults(user_vars: dict, data_dir: Path) -> dict:
    """Fill in less-frequently changed defaults into user_vars, resolving paths via data_dir where needed."""
    out = dict(user_vars)
    out.setdefault("EXPORT_DB_TO_EXCEL", DEFAULTS["EXPORT_DB_TO_EXCEL"])
    out.setdefault("TEMPLATE_DB", data_dir / DEFAULTS["TEMPLATE_DB"])
    out.setdefault("OUTPUT_DB", data_dir / DEFAULTS["OUTPUT_DB"])
    out.setdefault("NEMO_CONFIG_PATH", data_dir / DEFAULTS["NEMO_CONFIG_PATH"])
    out.setdefault("NEMO_TEST_DB_DIR", data_dir / DEFAULTS["NEMO_TEST_DB_DIR"])
    out.setdefault("NEMO_TEST_DB_PATH", None)
    out.setdefault("NEMO_TEST_NAME", DEFAULTS["NEMO_TEST_NAME"])
    out.setdefault("TEST_OUTPUT_DB", data_dir / DEFAULTS["TEST_OUTPUT_DB"])
    out.setdefault("TEST_INPUT_MODE", DEFAULTS["TEST_INPUT_MODE"])
    out.setdefault("TARGET_UNITS", DEFAULTS["TARGET_UNITS"])
    out.setdefault("USE_UNIT_CONVERSION", DEFAULTS["USE_UNIT_CONVERSION"])
    out.setdefault("ENABLE_NEMO_TRANSMISSION_METHODS", DEFAULTS["ENABLE_NEMO_TRANSMISSION_METHODS"])
    out.setdefault("REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS", DEFAULTS["REMAP_DEMAND_FUELS_AND_STRIP_TRANSMISSION_TECHS"])
    out.setdefault("TEST_INPUT_PATH", DEFAULTS["TEST_INPUT_PATH"])
    out.setdefault("TEST_DB_PATH", DEFAULTS["TEST_DB_PATH"] if DEFAULTS["TEST_DB_PATH"] is None else data_dir / DEFAULTS["TEST_DB_PATH"])
    out.setdefault("TEST_EXPORT_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_EXCEL_PATH"])
    out.setdefault("TEST_EXPORT_DB_TO_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_DB_TO_EXCEL_PATH"])
    out.setdefault("EXPORT_RESULTS_TO_EXCEL", DEFAULTS["EXPORT_RESULTS_TO_EXCEL"])
    out.setdefault("EXPORT_RESULTS_TO_EXCEL_PATH", DEFAULTS["EXPORT_RESULTS_TO_EXCEL_PATH"])
    out.setdefault("EXPORT_RESULTS_WIDE_TO_EXCEL", DEFAULTS["EXPORT_RESULTS_WIDE_TO_EXCEL"])
    out.setdefault("EXPORT_RESULTS_WIDE_TO_EXCEL_PATH", DEFAULTS["EXPORT_RESULTS_WIDE_TO_EXCEL_PATH"])
    out.setdefault("PLOTLY_DASHBOARD", DEFAULTS["PLOTLY_DASHBOARD"])
    if "PLOTLY_PNG_DIR" not in out:
        png_dir = DEFAULTS["PLOTLY_PNG_DIR"]
        out["PLOTLY_PNG_DIR"] = Path(png_dir) if png_dir is not None else None
    if "PLOTLY_DASHBOARD_PATH" not in out:
        out["PLOTLY_DASHBOARD_PATH"] = data_dir.parent / "plotting_output" / "dashboard.html"
    out.setdefault("EMISSIONS_FACTOR_CSV", DEFAULTS["EMISSIONS_FACTOR_CSV"])
    out.setdefault("EMISSIONS_LABEL", DEFAULTS["EMISSIONS_LABEL"])
    out.setdefault("EMISSIONS_FACTOR_OVERRIDE", DEFAULTS["EMISSIONS_FACTOR_OVERRIDE"])
    # Plotly YAML lives under the project (not under data_dir). Keep user-provided Path/str if set.
    if "PLOTLY_CONFIG_YAML" not in out:
        cfg_path = DEFAULTS["PLOTLY_CONFIG_YAML"]
        out["PLOTLY_CONFIG_YAML"] = Path(cfg_path)
    out.setdefault("NEMO_TEST_EXCEL_PATH", data_dir / DEFAULTS["TEST_EXPORT_EXCEL_PATH"])  # backward compat name
    out.setdefault("AUTO_CREATE_TEMPLATE_DB", DEFAULTS["AUTO_CREATE_TEMPLATE_DB"])
    out.setdefault("JULIA_EXE", DEFAULTS["JULIA_EXE"])
    out.setdefault("RUN_DIAGNOSTICS", DEFAULTS["RUN_DIAGNOSTICS"])
    out.setdefault("AUTO_FILL_MISSING_MODES", DEFAULTS["AUTO_FILL_MISSING_MODES"])
    out.setdefault("STRICT_ERRORS", DEFAULTS["STRICT_ERRORS"])
    out.setdefault("NEMO_WRITE_LP", DEFAULTS["NEMO_WRITE_LP"])
    out.setdefault("RUN_POSTPROCESS_ONLY", DEFAULTS.get("RUN_POSTPROCESS_ONLY", False))
    out.setdefault("SAVE_TEST_SNAPSHOT", DEFAULTS.get("SAVE_TEST_SNAPSHOT", False))
    out.setdefault("VARS_TO_SAVE", DEFAULTS["VARS_TO_SAVE"])
    vars_to_save = out.get("VARS_TO_SAVE") or []
    if vars_to_save:
        known = set(VARS_TO_SAVE_AVAILABLE)
        unknown = [v for v in vars_to_save if str(v) not in known]
        if unknown:
            print(f"WARNING: VARS_TO_SAVE has unknown entries: {unknown}")
    return out

__all__ = [
    "DEFAULTS",
    "DEFAULT_PARAMS",
    "VARS_TO_SAVE_AVAILABLE",
    "LEAP_TEMPLATE_DEFAULTS",
    "apply_defaults",
]
