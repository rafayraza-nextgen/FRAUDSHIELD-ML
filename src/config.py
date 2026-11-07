"""
Configuration module for F.R.A.U.D.S.H.I.E.L.D.
Defines project directory paths, runtime settings, and constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Reference Datasets
PAYSIM_DATASET_NAME = "paysim"
PAYSIM_RAW_FILENAME = "PS_20174392719_1491204439457_log.csv"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)
