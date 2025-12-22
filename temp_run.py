import sys
from pathlib import Path
sys.path.append('code')
from config_defaults import apply_defaults
from build_leap_import_template import apply_leap_template_defaults
from convert_osemosys_input_to_nemo import convert_osemosys_input_to_nemo

log_path = Path('temp_run.log')
log_path.write_text('start\n')
DATA_DIR = Path('data')
USER_VARS = {
    'OSEMOSYS_EXCEL_PATH': DATA_DIR / 'POWER 20_USA_data_REF9_S3_test - new file.xlsx',
    'NEMO_ENTRY_EXCEL_PATH': DATA_DIR / 'nemo_entry_dump.xlsx',
    'SCENARIO': 'Reference',
    'EXPORT_DB_TO_EXCEL_PATH': DATA_DIR / 'nemo_entry_dump.xlsx',
    'YEARS_TO_USE': [2017,2018,2019],
    'INPUT_MODE': 'osemosys',
}
log_path.write_text(log_path.read_text() + 'apply_defaults\n')
cfg = apply_defaults(USER_VARS, DATA_DIR)
log_path.write_text(log_path.read_text() + 'apply_leap\n')
cfg = apply_leap_template_defaults(cfg, DATA_DIR)
log_path.write_text(log_path.read_text() + f"OUTPUT_DB={cfg['OUTPUT_DB']}\n")
try:
    convert_osemosys_input_to_nemo(cfg, VERBOSE_ERRORS=True)
    log_path.write_text(log_path.read_text() + 'done\n')
except Exception as e:
    log_path.write_text(log_path.read_text() + 'error:' + repr(e) + '\n')
    raise
