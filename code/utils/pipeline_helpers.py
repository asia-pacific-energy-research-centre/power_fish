from __future__ import annotations

import os
from pathlib import Path

from nemo_core import handle_test_run as _core_handle_test_run, resolve_solver_from_test_name
from run_nemo_via_julia import resolve_lp_dump_path, DEFAULT_LP_PATH


def handle_test_run(vars_cfg: dict, data_dir: Path, run_nemo: bool) -> bool:
    """
    Wrapper that delegates to nemo_core.handle_test_run.
    Supports local sqlite/Excel or upstream NEMO test DBs (with optional Excel export).
    """
    log_dir = Path(vars_cfg.get("LOG_DIR", data_dir))
    return _core_handle_test_run(vars_cfg, data_dir, log_dir, run_nemo)


# Backward compatibility for older imports
def handle_storage_test(vars_cfg: dict, data_dir: Path, run_nemo: bool) -> bool:
    return handle_test_run(vars_cfg, data_dir, run_nemo)


def print_run_summary(cfg: dict, log_dir: Path, postprocess_only: bool = False):
    """Print the key files used in this run; include only relevant artifacts, in creation order."""
    log_path = log_dir / "nemo_run.log"
    lp_path = resolve_lp_dump_path()
    mode = cfg.get("RUN_MODE") or cfg.get("INPUT_MODE")
    is_test = mode == "test"

    print("\nRun summary")
    print("--------------------")
    print(f"  Mode:                 {mode}")

    # Inputs
    solver_test = None
    if not postprocess_only:
        if is_test:
            test_src = cfg.get("TEST_INPUT_PATH") or cfg.get("NEMO_TEST_NAME") or cfg.get("TEST_DB_PATH")
            if test_src:
                print(f"  Test input:           {test_src}")
            solver_test = resolve_solver_from_test_name(cfg.get("NEMO_TEST_NAME"))
        else:
            if cfg.get("OSEMOSYS_EXCEL_PATH"):
                print(f"  OSeMOSYS Excel:       {cfg['OSEMOSYS_EXCEL_PATH']}")
            if cfg.get("NEMO_ENTRY_EXCEL_PATH"):
                print(f"  NEMO entry Excel:     {cfg['NEMO_ENTRY_EXCEL_PATH']}")
        if cfg.get("NEMO_CONFIG_PATH"):
            print(f"  NEMO config (cwd):    {cfg['NEMO_CONFIG_PATH']}")

    # Intermediate / outputs
    if cfg.get("OUTPUT_DB"):
        print(f"  Output DB (sqlite):   {cfg['OUTPUT_DB']}")
    if not postprocess_only and cfg.get("EXPORT_DB_TO_EXCEL_PATH"):
        print(f"  DB export Excel:      {cfg['EXPORT_DB_TO_EXCEL_PATH']}")
    if cfg.get("EXPORT_RESULTS_TO_EXCEL_PATH") and cfg.get("EXPORT_RESULTS_TO_EXCEL"):
        print(f"  Results export:       {cfg['EXPORT_RESULTS_TO_EXCEL_PATH']}")
    if cfg.get("EXPORT_RESULTS_WIDE_TO_EXCEL_PATH") and cfg.get("EXPORT_RESULTS_WIDE_TO_EXCEL"):
        print(f"  Results export (wide):{cfg['EXPORT_RESULTS_WIDE_TO_EXCEL_PATH']}")
    if cfg.get("PLOTLY_DASHBOARD") and (cfg.get("PLOTLY_DASHBOARD_PATH")):
        print(f"  Plotly dashboard:     {cfg['PLOTLY_DASHBOARD_PATH']}")
    if cfg.get("GENERATE_LEAP_TEMPLATE") and cfg.get("LEAP_TEMPLATE_OUTPUT"):
        print(f"  LEAP template output: {cfg['LEAP_TEMPLATE_OUTPUT']}")

    # LP and log last; hide solver details in postprocess-only mode.
    if not postprocess_only:
        cfg_lp = cfg.get("NEMO_WRITE_LP")
        env_lp = os.environ.get("NEMO_WRITE_LP")
        if solver_test:
            if cfg_lp or env_lp:
                hint = "(env NEMO_WRITE_LP)" if env_lp else "(from cfg)"
                path = cfg_lp or env_lp
                print(f"  LP dump path:         {path} {hint}")
            else:
                print("  LP dump path:         <unknown> (solver test script; set NEMO_WRITE_LP to capture)")
        else:
            if cfg_lp:
                print(f"  LP dump path:         {cfg_lp} (from cfg)")
            elif env_lp:
                print(f"  LP dump path:         {env_lp} (env NEMO_WRITE_LP)")
            else:
                print(f"  LP dump path:         {lp_path} (default if write triggered)")
    print(f"  Log file:             {log_path}")
    print("--------------------\n")
