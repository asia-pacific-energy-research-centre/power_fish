## Testing Guidelines
- always test changes after making them:
- after making changes, run `python tests/main_agent_test.py` (tune AGENT_TEST_* in that file as needed) to verify nothing is broken. Make sure to change AGENT_TEST_MODE and AGENT_TEST_RUN_NEMO based on what changes were made. For example if a change was made that would adjust the input to NEMO, set AGENT_TEST_RUN_NEMO to True and AGENT_TEST_MODE to "nemo". If a change was made that would adjust the output from NEMO, set AGENT_TEST_RUN_NEMO to False and AGENT_TEST_MODE to one of db_only, osemosys_input_xlsx and nemo_input_xlsx depending on how far down the pipeline the change would affect. For full conversion+NEMO runs, set AGENT_TEST_OVERRIDES["RUN_POSTPROCESS_ONLY"]=False and AGENT_TEST_USE_LAST_RUN=False. When running the NEMO model, set a timeout to ~5x the last observed run time (latest full run was ~522s, so use ~2600s).
- if AGENT_TEST_MODE is one of db_only, osemosys_input_xlsx and nemo_input_xlsx then AGENT_TEST_RUN_NEMO should be true
- keep the tests up to date as you add features or fix bugs
- use `tests/last_run/runtime.json` to set per-mode timeouts; apply ~5x the recorded seconds for NEMO runs in that mode
## Visualization Validation
- for new/updated visualization code, validate a PNG export to ensure the figure is tidy:
- generate a PNG with Plotly (use `fig.write_image(...)` or `plotly.io.write_image`; requires `kaleido`)
- check that titles, axis labels, tick labels, and legends are not clipped or overlapping
- verify margins/padding, font sizes, and line/marker sizes are readable at the target resolution
- confirm long category labels are wrapped/rotated and do not collide with tick labels
- ensure colors and background contrast are clear in PNG (no transparency surprises)
- if any issues are found, adjust layout (`margin`, `legend`, `title`, `autosize`, `height/width`) and re-export until tidy
