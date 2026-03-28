# config.py
import os
import scipy.io
import numpy as np
import logging
from logging import Logger
from datetime import datetime
from typing import Optional, Dict, List, Any

# Paths
PATH_TO_DATA = 'project/ds006761'
DERIV_DIR = os.path.join(PATH_TO_DATA, 'derivatives')
ROOT_DIR = 'EEG-PROJECT'
PLOT_DIR = os.path.join(ROOT_DIR, 'results', 'plots')
LOG_DIR = os.path.join(ROOT_DIR, 'results', 'logs')

# Ensure directories exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Participant/Session Info
PAIR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
NUM_TRIALS = 480
NUM_TESTS = 4
NUM_TIME_BINS = 20
SFREQ = 256.0
NUM_CHAN = 64

# Channel Mapping (standard 10‑20 labels)
MATLAB_LAYOUT_LABELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
    'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
    'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
    'P10', 'PO8', 'PO4', 'O2'
]

# Classifier settings
SEARCHLIGHT_CLASSIFIERS = {'svm', 'lda'}          # only these run searchlight
N_JOBS_SEARCHLIGHT = -1                           # use all CPU cores for searchlight
SKIP_SEARCHLIGHT = False                           # set to True for faster testing
DEFAULT_CLASSIFIERS = ['svm', 'lda', 'logistic', 'ridge']  # classifiers to run by default

# ----------------------------------------------------------------------
# Electrode coordinates (from biosemi64.mat)
# ----------------------------------------------------------------------
def get_pos_dict() -> Optional[Dict[str, np.ndarray]]:
    """
    Load electrode coordinates from biosemi64.mat.

    Returns:
        dict: Mapping from channel label to (x,y,z) coordinate array.
        Returns None if the file cannot be loaded.
    """
    try:
        biosemi_mat = scipy.io.loadmat('biosemi64.mat')
        orig_coords = biosemi_mat['biosemi64']
        return {label: orig_coords[i] for i, label in enumerate(MATLAB_LAYOUT_LABELS)}
    except Exception as e:
        # Cannot use logger here because config is loaded before logging is set up.
        print(f"Warning: Could not load biosemi64.mat: {e}")
        return None

# ----------------------------------------------------------------------
# Logging setup – single master log file per run
# ----------------------------------------------------------------------
def setup_root_logger(level: int = logging.INFO, log_to_file: bool = True) -> None:
    """
    Configure the root logger with a console handler and an optional file handler.

    The file handler writes to a timestamped .md file inside LOG_DIR.
    This function should be called exactly once at the beginning of the main script.

    Args:
        level: Minimum logging level (default INFO).
        log_to_file: If True, add a file handler.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers (in case of re‑run in interactive session)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (single file per run)
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(LOG_DIR, f'run_{timestamp}.md')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    """
    Get a child logger with the given name. It propagates to the root logger,
    which is responsible for all output handlers (console + master file).

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)