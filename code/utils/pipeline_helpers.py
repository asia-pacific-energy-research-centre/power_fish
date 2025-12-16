from __future__ import annotations

from pathlib import Path

from nemo_core import handle_test_run as _core_handle_test_run


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
