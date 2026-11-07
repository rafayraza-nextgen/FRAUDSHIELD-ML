"""
Data loading and ingestion utilities for F.R.A.U.D.S.H.I.E.L.D.
Provides memory-optimized loading, chunking, and stratified sampling.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, PAYSIM_RAW_FILENAME, RANDOM_STATE

# Memory-optimized dtype mapping for PaySim dataset (6.36M rows)
PAYSIM_DTYPES = {
    "step": "uint16",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "uint8",
    "isFlaggedFraud": "uint8",
}


def find_paysim_file(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Search for the PaySim CSV dataset in the raw data directory or a custom path.
    Looks for standard Kaggle filenames and variations.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p

    # Search in default raw directory
    candidates = [
        RAW_DATA_DIR / PAYSIM_RAW_FILENAME,
        RAW_DATA_DIR / "paysim.csv",
        RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search for any CSV file inside data/raw
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    return None


def load_dataset(
    filepath: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    use_optimized_dtypes: bool = True,
) -> pd.DataFrame:
    """
    Load financial transaction dataset with optional row limiting and memory optimization.

    Args:
        filepath: Direct path to dataset. If None, auto-discovers in data/raw.
        nrows: Number of rows to read. Useful for quick inspection or memory limits.
        use_optimized_dtypes: If True, uses 16/32-bit types to reduce RAM footprint by ~60%.

    Returns:
        pd.DataFrame containing transaction records.
    """
    target_path = find_paysim_file(filepath)
    if not target_path or not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath or RAW_DATA_DIR}'.\n"
            f"Please download the PaySim dataset from Kaggle and place the CSV in:\n"
            f"  {RAW_DATA_DIR.resolve()}\n"
            f"Expected filename: {PAYSIM_RAW_FILENAME} or any *.csv in data/raw/"
        )

    dtypes = PAYSIM_DTYPES if use_optimized_dtypes else None

    # Load data
    df = pd.read_csv(target_path, nrows=nrows, dtype=dtypes)
    return df


def inspect_dataset_metadata(df: pd.DataFrame, target_col: str = "isFraud") -> Dict[str, Any]:
    """
    Inspects key properties, shapes, nulls, duplicates, and class distribution of a dataset.

    Returns:
        Dictionary containing summary metadata.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    class_distribution = {}
    if target_col in df.columns:
        counts = df[target_col].value_counts().to_dict()
        percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()
        class_distribution = {
            "counts": counts,
            "percentages": {k: round(v, 4) for k, v in percentages.items()},
        }

    return {
        "rows": total_rows,
        "columns": total_cols,
        "column_names": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "total_missing_values": total_missing,
        "missing_per_column": missing_counts,
        "target_column": target_col,
        "class_distribution": class_distribution,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
