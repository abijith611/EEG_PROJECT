# config.py
import os
import scipy.io
import numpy as np

# Paths
PATH_TO_DATA = 'project/ds006761'
DERIV_DIR = os.path.join(PATH_TO_DATA, 'derivatives')
ROOT_DIR = 'EEG-PROJECT'
PLOT_DIR = os.path.join(ROOT_DIR, 'results', 'plots')

# Participant/Session Info
PAIR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
NUM_TRIALS = 480
NUM_TESTS = 4
NUM_TIME_BINS = 20
SFREQ = 256.0
NUM_CHAN = 64

# Channel Mapping
MATLAB_LAYOUT_LABELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 
    'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 
    'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 
    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 
    'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 
    'P10', 'PO8', 'PO4', 'O2'
]

SEARCHLIGHT_CLASSIFIERS = {'svm', 'lda'}   # only these will run searchlight
N_JOBS_SEARCHLIGHT = -1   # Use all CPU cores for searchlight. Set to 1 to disable parallelism.
SKIP_SEARCHLIGHT = False  # Set to True to skip searchlight (faster for testing).
DEFAULT_CLASSIFIERS = ['svm', 'lda', 'logistic', 'ridge']  # can be overridden by command line

# Shared Loading Logic for Coordinates
def get_pos_dict():
    try:
        biosemi_mat = scipy.io.loadmat('biosemi64.mat')
        orig_coords = biosemi_mat['biosemi64']
        return {label: orig_coords[i] for i, label in enumerate(MATLAB_LAYOUT_LABELS)}
    except Exception:
        return None