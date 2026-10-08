"""File loading and temporary directory utilities."""

import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import variables
from .logger import setup_logger

logger = setup_logger(__name__)


def load_json(file: str, encoding: str = "utf-8") -> dict:
    """Read a JSON file as a dictionary."""
    file = os.path.abspath(file)
    if not os.path.exists(file):
        raise FileNotFoundError(f'File "{file}" not found')

    try:
        with open(file, encoding=encoding) as f:
            data = json.load(f)
    except Exception as ex:
        raise Exception(f"Error reading the JSON file {file}\nException: {ex}")

    return data


def load_suvr_data(
    file: str, standarize: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load SUVRs and groups, optionally using sample standard deviations."""
    data = pd.read_parquet(variables.SUVR_FOLDER / file)
    data = data.reset_index().drop(columns=["file"])

    suvr_cols = [c for c in data.columns if c.endswith("suvr")]
    suvr = data[suvr_cols].values
    groups = data["group"].values

    if standarize:
        suvr = (suvr - np.mean(suvr, axis=0)) / np.std(suvr, axis=0, ddof=1)

    return data, suvr, groups


def create_temp_dir() -> Path:
    """Create a temporary directory in the working directory, removed at exit."""

    def cleanup():
        logger.info(f'Removing temporary directory "{temp_dir}"')
        shutil.rmtree(temp_dir)

    temp_dir = tempfile.mkdtemp(dir=".")
    logger.info(f'Creating temporary directory "{temp_dir}"')
    atexit.register(cleanup)

    return Path(temp_dir)
