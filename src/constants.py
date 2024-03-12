import os
import pathlib

IS_KAGGLE = os.getenv("KAGGLE")
COMPE_NAME = "hms-harmful-brain-activity-classification"
ROOT = pathlib.Path("/kaggle") if IS_KAGGLE else pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = INPUT_DIR / COMPE_NAME

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
