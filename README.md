# power_fish: NEMO + LEAP starter kit

This repository converts OSeMOSYS-style Excel inputs into a NEMO scenario database, runs NEMO via Julia, and provides quick diagnostics. It also seeds future LEAP interactions while keeping the vibe light with a few Finding NEMO jokes.

## Prerequisites
- Python 3.10+ (conda-friendly); recommended env file: `config/env_leap.yml`, use: conda env create --prefix ./env_leap --file ./config/env_leap.yml
- Julia 1.9+ with the NEMO model (NemoMod.jl) installed.
- Microsoft Excel/`openpyxl` for reading the input workbook.
- (Planned) LEAP for model exchange and API automation.

## Install LEAP and Julia/NEMO  
- Download LEAP and NEMO/Julia from https://leap.sei.org/default.asp?action=download (requires a free account). 
- LEAP docs & help center live at https://leap.sei.org/learn/ (API and automation guidance are in the LEAP help menu).
- The COM API type library included here (`config/TypeLib_LEAP_API_full.txt`) is for reference when wiring Python automation later.

## Set up NEMO
1. Create a template database once (placed in your `data/` folder). The pipeline can also auto-create it at runtime if `AUTO_CREATE_TEMPLATE_DB=True` and Julia/NemoMod are available:
   ```julia
   using NemoMod
   cd("C:/Users/YOU/path/to/power_fish/data")
   NemoMod.createnemodb("nemo_template.sqlite")
   ```
2. NEMO docs: https://sei-international.github.io/NemoMod.jl/stable/ 

## Python environment
```bash
conda env create --prefix ./env_leap --file config/env_leap.yml
conda activate ./env_leap
```
The environment includes pandas/openpyxl/matplotlib/plotly for data prep and inspection.

## Running the pipeline
1. Configure paths and options in `code/main.py` (input Excel, template DB, output DB, scenario name, Julia path, diagnostics flags, `AUTO_CREATE_TEMPLATE_DB` toggle).
2. If `AUTO_CREATE_TEMPLATE_DB` is True, the script will build `data/nemo_template.sqlite` for you on first run (requires Julia with NemoMod installed). Otherwise, create it manually (see NEMO setup above).
3. Run everything in Jupyter interactive via visual studio code (which is how finn codes), or in python e.g.:
   ```bash
   python code/main.py
   ```
   - Converts the original OSeMOSYS Excel workbook to a NEMO DB (`OUTPUT_DB`).
   - Optionally runs diagnostics (`RUN_DIAGNOSTICS`).
   - Solves the scenario in Julia/NEMO (`run_nemo_via_julia.py`) and writes an optional log (e.g., `data/nemo_run.log`).
   - To skip conversion and just run a test DB, run with `mode="test"` (e.g., `python code/main.py` with `mode` set in the call or cell). You can point `TEST_INPUT_PATH` to a local `.sqlite` or `.xlsx` (nemo_entry/osemosys; Excel will be converted), or pick an upstream `NEMO_TEST_NAME` (`storage_test`, `storage_transmission_test`, `ramp_test`) to auto-download into `data/nemo_tests/` (auto-download also kicks in when `TEST_INPUT_PATH` is missing but `NEMO_TEST_NAME` is set). Solver-specific test names like `cbc_tests`/`glpk_tests` will download the upstream Julia test script and run it against the bundled NEMO test DBs (logs in `results/logs/<solver>_tests.log`). During a test flow the DB-to-Excel dump uses `TEST_EXPORT_DB_TO_EXCEL_PATH` (falls back to `TEST_EXPORT_EXCEL_PATH`) so you can keep test exports separate from main runs.
   - To run an existing NEMO database without reconverting Excel, use `mode="db_only"`; set `OUTPUT_DB` to the `.sqlite` you want to run/diagnose.

## Coding style at a glance
- Scripts are split into `#%%` cells for quick interactive runs (VS Code with Jupyter interactive).
- Functions first, orchestration later: utilities and data loaders sit near the top; `MAIN_*` blocks call them in order.
- Prefer small, single-purpose functions over classes; keep inputs and outputs explicit.
- Breakpoints are sprinkled in so you can step through cells when debugging.

## LEAP integration roadmap
- export the populated NEMO DB back to Excel (`EXPORT_DB_TO_EXCEL`) for LEAP imports.
- script LEAP COM calls (using the provided type library and what has already been done in leap_utils_with_transport_toolkit) to pull scenarios, push results, and align metadata.
- automate end-to-end runs from LEAP through NEMO and back.
- simplify and robustify the workflow for non-Python users.

## Where things live
- Workflow scripts: `code/main.py`, `code/convert_osemosys_input_to_nemo.py`, `code/run_nemo_via_julia.py`, `code/diagnostics.py`
- Julia runner: `code/nemo_process.jl`
- Sample data and outputs: `data/`, `intermediate_data/`, `results/`, `plotting_output/`

🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟

## Finding NEMO (jokes, obligatory)
- Why did the optimizer go to the reef? It heard the solution space was convex down there.
- If you can’t find your fish, just keep iterating. The solver will eventually converge to Nemo.
- Dory’s pro tip: when in doubt, ask finn and swim on.

🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟
