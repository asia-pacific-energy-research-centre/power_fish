# power_fish: NEMO + LEAP starter kit

Convert OSeMOSYS-style Excel inputs into a NEMO scenario database, run NEMO via Julia, and generate diagnostics plus Plotly dashboards. 

## Quick start
1. Create the Python environment:
   ```bash
   conda env create --prefix ./env_leap --file config/env_leap.yml
   conda activate ./env_leap
   ```
2. Create a NEMO template database (one-time):
   ```julia
   using NemoMod
   cd("C:/Users/YOU/path/to/power_fish/data")
   NemoMod.createnemodb("nemo_template.sqlite")
   ```
3. Run the pipeline (from a notebook cell or CLI):
   ```bash
   python code/main.py
   ```

## What you get
- NEMO scenario database (`.sqlite`) built from the input workbook.
- Optional diagnostics and exports (Excel outputs).
- Plotly dashboard HTML with charts defined in YAML.

## Prerequisites
- Python 3.10+ (conda-friendly); environment file: `config/env_leap.yml`.
- Julia 1.9+ with NemoMod.jl installed.
- Excel / `openpyxl` for reading input workbooks.
- (Planned) LEAP integration.

## Pipeline usage (short version)
- Configure paths and options in `code/main.py` (input Excel, template DB, output DB, scenario name, Julia path, diagnostics flags, `AUTO_CREATE_TEMPLATE_DB`).
- Run with `python code/main.py`.
- For postprocess only (no Julia run), set `RUN_POSTPROCESS_ONLY=true` or use `mode="results"`.
- To run an existing NEMO DB without reconverting Excel, use `mode="db_only"` with `OUTPUT_DB` set to the `.sqlite` to process.

## Visualization features (Plotly dashboards)
Charts are controlled by `config/plotly_charts.yml` and rendered after each run.

### How it works
- Two chart types:
  - `function_figs`: named builders in `code/plotting/plotly_dashboard_functions.py`.
  - `dict_figs`: direct table/X/Y/color configs defined in YAML.
- Mappings + colors come from `config/plotting_config_and_timeslices.xlsx`.
- Any unmapped labels are written to `config/missing_plot_colors.csv` for cleanup.
- Dashboard HTML is written to `results/plots/` (or `PLOTLY_DASHBOARD_PATH` when set).

### Common edits in `config/plotly_charts.yml`
- Add/remove charts by editing `function_figs`.
- Tweak each chart with options under its name.
- Use `drop_zero_categories`, `drop_categories`, or `drop_category_substrings` to clean legends.

### Example: enable/disable a chart
```yaml
function_figs:
  generation:
    drop_zero_categories: true
  costs_by_technology:
    drop_zero_categories: true
```

### Example: adjust the generation chart demand line
```yaml
function_figs:
  generation:
    add_demand_line: true
    demand_line_label: Demand
    demand_line_color: "#000000"
    demand_line_width: 2
```

## LEAP integration roadmap
- Export the populated NEMO DB back to Excel (`EXPORT_DB_TO_EXCEL`) for LEAP imports.
- Script LEAP COM calls (using `config/TypeLib_LEAP_API_full.txt`) for scenario exchange.
- Automate end-to-end runs between LEAP and NEMO.

## Where things live
- Workflow scripts: `code/main.py`, `code/convert_osemosys_input_to_nemo.py`, `code/run_nemo_via_julia.py`
- Plotting: `code/plotting/plotly_dashboard.py`, `code/plotting/plotly_dashboard_functions.py`
- Data + outputs: `data/`, `intermediate_data/`, `results/`, `plotting_output/`

## Install LEAP and Julia/NEMO
- Download LEAP and NEMO/Julia: https://leap.sei.org/default.asp?action=download
- LEAP docs: https://leap.sei.org/learn/
- NEMO docs: https://sei-international.github.io/NemoMod.jl/stable/

🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟

## Finding NEMO (jokes, obligatory)
- Why did the optimizer go to the reef? It heard the solution space was convex down there.
- If you can’t find your fish, just keep iterating. The solver will eventually converge to Nemo.
- Dory’s pro tip: when in doubt, ask finn and swim on.

🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟🐟
