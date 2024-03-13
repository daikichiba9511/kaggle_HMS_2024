import os
import pathlib

IS_KAGGLE = os.getenv("KAGGLE")
COMPE_NAME = "hms-harmful-brain-activity-classification"
ROOT = pathlib.Path("/kaggle") if IS_KAGGLE else pathlib.Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = INPUT_DIR / COMPE_NAME

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

feature_cols = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
    "EKG",
]
