"""Central settings. Everything here is a public-data stand-in for the Navy's own inputs."""
from pathlib import Path

IGRA_BASE = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por"

# San Diego (Miramar) is the main station; Abu Dhabi (Persian Gulf coast) is a stress test
# for "conditions differ from what the model was built on".
DEFAULT_STATION = "USM00072293"
STRESS_STATION = "AEM00041217"
STATION_NAMES = {
    "USM00072293": "San Diego/Miramar, CA",
    "AEM00041217": "Abu Dhabi Intl, UAE (Persian Gulf)",
}

START_YEAR = 2000          # earlier soundings are coarser and less consistent
TEST_START = "2022-01-01"  # everything before is training data, everything after is held-out test

# Ducting detection
MAX_DUCT_HEIGHT_M = 3000.0   # look for trapping layers in the lowest 3 km
DUCT_MIN_STRENGTH_M = 5.0    # total M-unit drop needed to call it a duct (filters sampling noise)
SURFACE_DUCT_BASE_M = 50.0   # a trapping layer starting below this counts as surface-based

# Quality control. A sounding must reach 3 km with enough levels and no huge vertical gaps.
QC_STANDARD = {"min_levels": 12, "max_gap_m": 1000.0}
# Older / lower-resolution reports (e.g. the Persian Gulf station) only pass a relaxed check.
QC_RELAXED = {"min_levels": 8, "max_gap_m": 1500.0}
MAX_AGE_HOURS = 36.0         # warn if the newest sounding is older than this

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SHARED_DIR = ROOT / "shared"
RESULTS_DIR = ROOT / "results"
