from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from nemo_core import handle_test_run as _core_handle_test_run, resolve_solver_from_test_name
from run_nemo_via_julia import resolve_lp_dump_path, DEFAULT_LP_PATH
from convert_osemosys_input_to_nemo import export_results_to_excel, export_results_to_excel_wide
from plotting.plotly_dashboard import generate_plotly_dashboard
from build_leap_import_template import generate_leap_template


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


def _as_bool(val) -> bool:
    """Lightweight truthy coercion for user/CLI/env inputs."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(val)


def run_postprocess_steps(cfg: dict, db_path: Path, project_root: Path):
    """Run exports/plots/dashboard (shared by normal and postprocess-only flows)."""
    tokens = _build_output_tokens(cfg, db_path, project_root)
    if cfg.get("EXPORT_RESULTS_TO_EXCEL"):
        export_results_to_excel(
            db_path=db_path,
            excel_path=_format_path_template(
                cfg.get("EXPORT_RESULTS_TO_EXCEL_PATH")
                or project_root / "results" / "results.xlsx",
                tokens,
            ),
        )
    if cfg.get("EXPORT_RESULTS_WIDE_TO_EXCEL"):
        export_results_to_excel_wide(
            db_path=db_path,
            excel_path=_format_path_template(
                cfg.get("EXPORT_RESULTS_WIDE_TO_EXCEL_PATH")
                or project_root / "results" / "results_wide.xlsx",
                tokens,
            ),
        )
    if cfg.get("PLOTLY_DASHBOARD"):
        generate_plotly_dashboard(
            db_path=db_path,
            output_path=_format_path_template(
                cfg.get("PLOTLY_DASHBOARD_PATH")
                or project_root / "plotting_output" / "dashboard.html",
                tokens,
            ),
            plots_config_dict=None,
            layout="scroll",
            function_figs=None,
            config_yaml=cfg.get("PLOTLY_CONFIG_YAML"),
            no_columns=None,
            export_png_dir=cfg.get("PLOTLY_PNG_DIR"),
            name_tokens=tokens,
        )
    if cfg.get("GENERATE_LEAP_TEMPLATE"):
        generate_leap_template(
            scenario=cfg.get("SCENARIO"),
            region=cfg.get("LEAP_TEMPLATE_REGION"),
            nemo_db_path=db_path,
            output_path=_format_path_template(
                cfg.get("LEAP_TEMPLATE_OUTPUT"),
                tokens,
            ),
            import_id_source=cfg.get("LEAP_IMPORT_ID_SOURCE"),
        )


def save_latest_run_snapshot(cfg: dict, project_root: Path) -> Path:
    """
    Save the latest run settings and key input files into tests/last_run.
    Returns the path to the snapshot config JSON.
    """
    dest_root = project_root / "tests" / "last_run"
    dest_root.mkdir(parents=True, exist_ok=True)

    files_to_copy = {
        "OSEMOSYS_EXCEL_PATH": cfg.get("OSEMOSYS_EXCEL_PATH"),
        "NEMO_ENTRY_EXCEL_PATH": cfg.get("NEMO_ENTRY_EXCEL_PATH"),
        "NEMO_CONFIG_PATH": cfg.get("NEMO_CONFIG_PATH"),
        "LEAP_IMPORT_ID_SOURCE": cfg.get("LEAP_IMPORT_ID_SOURCE"),
        "OUTPUT_DB": cfg.get("OUTPUT_DB"),
    }

    files_snapshot = {}
    for key, src in files_to_copy.items():
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        dest_name = "output_db.sqlite" if key == "OUTPUT_DB" else f"{key.lower()}_{src_path.name}"
        dest_path = dest_root / dest_name
        shutil.copy2(src_path, dest_path)
        files_snapshot[key] = {
            "source": str(src_path),
            "snapshot": str(dest_path),
        }

    cfg_keys = [
        "SCENARIO",
        "YEARS_TO_USE",
        "VARS_TO_SAVE",
        "INPUT_MODE",
        "RUN_MODE",
        "RUN_POSTPROCESS_ONLY",
        "EXPORT_RESULTS_TO_EXCEL",
        "EXPORT_RESULTS_WIDE_TO_EXCEL",
        "PLOTLY_DASHBOARD",
        "PLOTLY_CONFIG_YAML",
    ]
    cfg_snapshot = {}
    for key in cfg_keys:
        val = cfg.get(key)
        if isinstance(val, Path):
            cfg_snapshot[key] = str(val)
        else:
            cfg_snapshot[key] = val

    snapshot = {
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": cfg_snapshot,
        "files": files_snapshot,
    }

    snapshot_path = dest_root / "config.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot_path


def _build_output_tokens(cfg: dict, db_path: Path, project_root: Path) -> dict:
    def _stem(value: str | Path | None) -> str:
        if not value:
            return ""
        return Path(value).stem

    no_dateid = bool(cfg.get("NO_DATEID"))
    dateid = "" if no_dateid else datetime.utcnow().strftime("%Y%m%d")
    return {
        "SCENARIO": str(cfg.get("SCENARIO") or ""),
        "ECONOMY": str(cfg.get("ECONOMY") or ""),
        "INPUTNAME": _stem(
            cfg.get("OSEMOSYS_EXCEL_PATH")
            or cfg.get("NEMO_ENTRY_EXCEL_PATH")
            or cfg.get("TEST_INPUT_PATH")
            or db_path
        ),
        "DATEID": dateid,
        "NO_DATEID": no_dateid,
        "RUN_MODE": str(cfg.get("RUN_MODE") or cfg.get("INPUT_MODE") or ""),
        "PROJECT": _stem(project_root),
    }


def _format_path_template(path: str | Path | None, tokens: dict) -> Path:
    if path is None:
        raise ValueError("Missing output path template.")
    path_str = str(path)
    for key, value in tokens.items():
        token = f"{{{key}}}"
        if token in path_str and value is not None:
            path_str = path_str.replace(token, str(value))
    return Path(_cleanup_tokenized_name(path_str))


def _cleanup_tokenized_name(value: str) -> str:
    # Collapse duplicate separators and clean up before extensions.
    out = re.sub(r"[_-]{2,}", "_", value)
    out = re.sub(r"_+\.", ".", out)
    out = re.sub(r"-+\.", ".", out)
    out = out.strip("_-")
    return out
