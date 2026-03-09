"""
Decoding script (modified):
  - Decode own & opponent's response for current & previous trial
  - Uses linear SVM instead of LDA
  - Uses stratified 10‑fold cross‑validation with shuffling,
    preserving the group (chunk) structure of pseudo‑trials
  - Groups are the fold indices from pseudo‑trial generation

Toolboxes needed: mne, numpy, scipy, sklearn, pandas
"""

import os
import mne
import numpy as np
import pandas as pd
import scipy.io
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import pickle

path_to_data = 'project/ds006761'
# Matches the pair_ids used in the author's MATLAB script 
pair_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
num_trials = 480
num_chan = 64

# Exact channel list as per the author's FieldTrip layout mapping 
matlab_layout_labels = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 
    'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 
    'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 
    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 
    'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 
    'P10', 'PO8', 'PO4', 'O2'
]

# Load exact 3D coordinates from the provided biosemi64.mat [cite: 8, 10]
try:
    biosemi_mat = scipy.io.loadmat('biosemi64.mat')
    orig_coords = biosemi_mat['biosemi64']
    pos_dict = {label: orig_coords[i] for i, label in enumerate(matlab_layout_labels)}
except Exception:
    pos_dict = None

def get_time_bins(epochs):
    times = epochs.times
    data = epochs.get_data() * 1e6  # Convert to microvolts 
    
    # Slice into phases matching MATLAB latencies 
    mask_A = (times >= -0.2) & (times <= 2.0)
    mask_B = (times >= 1.8) & (times <= 4.0)
    mask_C = (times >= 3.8) & (times <= 5.0)
    
    data_A, data_B, data_C = data[:, :, mask_A], data[:, :, mask_B], data[:, :, mask_C]
    times_A = times[mask_A]
    
    # Baseline correction on [-0.2, 0] [cite: 10, 11]
    mask_base = (times_A >= -0.2) & (times_A <= 0)
    baseline_A = np.mean(data_A[:, :, mask_base], axis=2, keepdims=True)
    baseline_B = np.mean(data_B[:, :, mask_base], axis=2, keepdims=True)
    baseline_C = np.mean(data_C[:, :, mask_base[:data_C.shape[2]]], axis=2, keepdims=True)
    
    data_A -= baseline_A
    data_B -= baseline_B
    data_C -= baseline_C
    
    # 250ms bins matching author's window settings [cite: 10, 11]
    time_windows_AB = np.array([np.arange(0, 1.76, 0.25), np.arange(0.25, 2.01, 0.25)]).T
    time_windows_C = np.array([np.arange(0, 0.76, 0.25), np.arange(0.25, 1.01, 0.25)]).T
    
    binned_data = np.zeros((data.shape[0], num_chan, 20))
    bin_idx = 0
    for d, tw in zip([data_A, data_B], [time_windows_AB, time_windows_AB]):
        for w in tw:
            m = (times_A[:d.shape[2]] > w[0]) & (times_A[:d.shape[2]] < w[1])
            binned_data[:, :, bin_idx] = np.mean(d[:, :, m], axis=2)
            bin_idx += 1
    for w in time_windows_C:
        m = (times_A[:data_C.shape[2]] > w[0]) & (times_A[:data_C.shape[2]] < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_C[:, :, m], axis=2)
        bin_idx += 1
    return binned_data

def create_pseudo_trials(X, y, seed):
    """
    Replicates cosmo_average_samples logic exactly:
    - Splits data into 10 folds (StratifiedKFold)
    - For each fold, randomly samples 4 trials per class (with replacement if needed)
    - Repeats 20 times per class per fold
    - Returns pseudo‑trials, labels, and chunk indices (fold numbers)
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    X_pseudo, y_pseudo, chunks = [], [], []
    np.random.seed(seed)
    for chunk_idx, (_, fold_indices) in enumerate(skf.split(X, y)):
        fold_y, fold_X = y[fold_indices], X[fold_indices]
        for c in np.unique(fold_y):
            c_idx = np.where(fold_y == c)[0]
            replace = len(c_idx) < 4
            for _ in range(20):  # 'repeats' = 20
                samp_idx = np.random.choice(c_idx, size=4, replace=replace)
                X_pseudo.append(np.mean(fold_X[samp_idx], axis=0))
                y_pseudo.append(c)
                chunks.append(chunk_idx)
    return np.array(X_pseudo), np.array(y_pseudo), np.array(chunks)

def run_decoding(max_pairs=None):
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs = len(pairs_to_run)
    
    for p_idx, pair in enumerate(pairs_to_run):
        print(f'Loading pair {p_idx + 1} of {num_pairs} (ID: {pair})')
        sub_str = f'sub-{pair:02d}'
        events = pd.read_csv(os.path.join(path_to_data, sub_str, 'eeg', f'{sub_str}_task-RPS_events.tsv'), sep='\t')
        
        # Build behavioral targets 
        ev_p1 = events[['player1_resp', 'player2_resp', 'outcome']].values
        hist_p1 = np.full((num_trials, 2), np.nan)
        hist_p1[1:] = ev_p1[:-1, :2]
        behav_p1 = np.column_stack([ev_p1, hist_p1])
        
        # Player 2 outcome relative to them 
        ev_p2_raw = ev_p1[:, [1, 0, 2]]
        ev_p2_raw[ev_p1[:, 2] == 2, 2] = 3
        ev_p2_raw[ev_p1[:, 2] == 3, 2] = 2
        hist_p2 = np.full((num_trials, 2), np.nan)
        hist_p2[1:] = ev_p2_raw[:-1, :2]
        behav_p2 = np.column_stack([ev_p2_raw, hist_p2])

        for ppt, behav in zip([1, 2], [behav_p1, behav_p2]):
            print(f'   ppt {ppt}')
            epoch_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
            if not os.path.exists(epoch_file):
                continue
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
            epochs.set_eeg_reference('average', projection=False, verbose=False)
            
            # Remove first trial of each block (40‑trial blocks)
            sel_idx = np.setdiff1d(np.arange(num_trials), np.arange(0, 480, 40))
            behav_data = behav[sel_idx]
            binned_data = get_time_bins(epochs[sel_idx])
            
            # Spatial searchlight neighbors (same as before)
            ch_names = epochs.ch_names
            if pos_dict is not None:
                dist_mat = scipy.spatial.distance.cdist(
                    [pos_dict[c] for c in ch_names],
                    [pos_dict[c] for c in ch_names]
                )
                neighbor_lists = [np.argsort(dist_mat[i])[0:5] for i in range(num_chan)]
            else:
                # Fallback: use 5 nearest by index (only if positions unavailable)
                neighbor_lists = [np.arange(max(0, i-2), min(num_chan, i+3)) for i in range(num_chan)]

            decoding_results = []
            searchlight_results = []
            
            for target_col in [0, 1, 3, 4]:  # self, other, self_prev, other_prev
                y = behav_data[:, target_col]
                valid = ~np.isnan(y) & (y > 0)
                # Dynamic seed per participant prevents synchronized noise dips
                seed = pair * 10 + ppt
                X_ps, y_ps, groups = create_pseudo_trials(binned_data[valid], y[valid], seed)
                
                # Linear SVM (C=1 by default, can be tuned)
                clf = SVC(kernel='linear', random_state=seed)
                
                # StratifiedGroupKFold: 10 folds, shuffle, respecting group labels
                cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
                
                # Time‑resolved decoding
                acc_time = []
                for t in range(20):
                    scores = cross_val_score(clf, X_ps[:, :, t], y_ps, groups=groups, cv=cv, n_jobs=-1)
                    acc_time.append(np.mean(scores))
                decoding_results.append(acc_time)
                
                # Channel searchlight
                sl_acc = np.zeros((num_chan, 20))
                for ch in range(num_chan):
                    for t in range(20):
                        scores = cross_val_score(clf, X_ps[:, neighbor_lists[ch], t], y_ps, groups=groups, cv=cv, n_jobs=-1)
                        sl_acc[ch, t] = np.mean(scores)
                searchlight_results.append(sl_acc)
                
            # Save results
            with open(os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl'), 'wb') as f:
                pickle.dump({
                    'decoding': decoding_results,
                    'searchlight': searchlight_results,
                    'ch_names': ch_names
                }, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    run_decoding(max_pairs=args.test_pairs)