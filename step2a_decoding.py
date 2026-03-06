"""
Decoding script:
  - Decode own & opponent's response for current & previous trial
  - Uses Time-binning, Pseudo-trial averaging, and Temporal + Spatial Searchlight

Libraries needed: mne, numpy, pandas, scikit-learn, scipy
"""

import os
import mne
import numpy as np
import pandas as pd
import scipy.io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score
import pickle

path_to_data = 'project/ds006761'
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_trials = 480
num_chan = 64

# Mapping from Biosemi labels (A1..B32) to standard 10‑20 names
biosemi_to_std = {
    'A1': 'Fp1', 'A2': 'Fpz', 'A3': 'Fp2', 'A4': 'AF7', 'A5': 'AF3', 'A6': 'AFz', 'A7': 'AF4', 'A8': 'AF8',
    'A9': 'F7', 'A10': 'F5', 'A11': 'F3', 'A12': 'F1', 'A13': 'Fz', 'A14': 'F2', 'A15': 'F4', 'A16': 'F6',
    'A17': 'F8', 'A18': 'FT7', 'A19': 'FC5', 'A20': 'FC3', 'A21': 'FC1', 'A22': 'FCz', 'A23': 'FC2', 'A24': 'FC4',
    'A25': 'FC6', 'A26': 'FT8', 'A27': 'T7', 'A28': 'C5', 'A29': 'C3', 'A30': 'C1', 'A31': 'Cz', 'A32': 'C2',
    'B1': 'C4', 'B2': 'C6', 'B3': 'T8', 'B4': 'TP7', 'B5': 'CP5', 'B6': 'CP3', 'B7': 'CP1', 'B8': 'CPz',
    'B9': 'CP2', 'B10': 'CP4', 'B11': 'CP6', 'B12': 'TP8', 'B13': 'P7', 'B14': 'P5', 'B15': 'P3', 'B16': 'P1',
    'B17': 'Pz', 'B18': 'P2', 'B19': 'P4', 'B20': 'P6', 'B21': 'P8', 'B22': 'PO7', 'B23': 'PO5', 'B24': 'PO3',
    'B25': 'POz', 'B26': 'PO4', 'B27': 'PO6', 'B28': 'PO8', 'B29': 'O1', 'B30': 'Oz', 'B31': 'O2', 'B32': 'Iz'
}

# Generate ordered list of standard channel names (A1..A32, B1..B32)
std_names_ordered = [biosemi_to_std[f'A{i}'] for i in range(1,33)] + [biosemi_to_std[f'B{i}'] for i in range(1,33)]

# Load the original BioSemi electrode positions
biosemi_mat = scipy.io.loadmat('biosemi64.mat')   # adjust path if needed
orig_coords = biosemi_mat['biosemi64']            # shape (64, 3)

# Create a dictionary mapping channel name to its 3D position
pos_dict = {name: orig_coords[i] for i, name in enumerate(std_names_ordered)}

def get_time_bins(epochs):
    times = epochs.times
    data = epochs.get_data()
    
    # Masks for each phase
    mask_A = (times >= -0.2) & (times <= 2.0)
    mask_B = (times >= 1.8) & (times <= 4.0)
    mask_C = (times >= 3.8) & (times <= 5.0)
    
    # Baseline masks: 200 ms before each phase onset
    mask_base_A = (times >= -0.2) & (times <= 0)          # before decision
    mask_base_B = (times >= 1.6) & (times <= 1.8)        # before response
    mask_base_C = (times >= 3.6) & (times <= 3.8)        # before feedback
    
    # Extract data for each phase
    data_A = data[:, :, mask_A]
    data_B = data[:, :, mask_B]
    data_C = data[:, :, mask_C]
    
    # Compute baselines
    baseline_A = np.mean(data[:, :, mask_base_A], axis=2, keepdims=True)
    baseline_B = np.mean(data[:, :, mask_base_B], axis=2, keepdims=True)
    baseline_C = np.mean(data[:, :, mask_base_C], axis=2, keepdims=True)
    
    # Subtract baselines
    data_A -= baseline_A
    data_B -= baseline_B
    data_C -= baseline_C
    
    # Shift time axes so that each phase starts at 0
    times_A = times[mask_A]
    times_B = times[mask_B] - 2.0   # response phase starts at 1.8s → shift to 0
    times_C = times[mask_C] - 4.0   # feedback phase starts at 3.8s → shift to 0
    
    # Define bin edges (same as before)
    time_windows_AB = np.array([np.arange(0, 1.76, 0.25), np.arange(0.25, 2.01, 0.25)]).T
    time_windows_C = np.array([np.arange(0, 0.76, 0.25), np.arange(0.25, 1.01, 0.25)]).T
    
    num_tr = data.shape[0]
    binned_data = np.zeros((num_tr, num_chan, len(time_windows_AB)*2 + len(time_windows_C)))
    
    bin_idx = 0
    # Decision phase bins (using times_A)
    for w in time_windows_AB:
        m_A = (times_A > w[0]) & (times_A < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_A[:, :, m_A], axis=2)
        bin_idx += 1
    # Response phase bins (using times_B)
    for w in time_windows_AB:
        m_B = (times_B > w[0]) & (times_B < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_B[:, :, m_B], axis=2)
        bin_idx += 1
    # Feedback phase bins (using times_C)
    for w in time_windows_C:
        m_C = (times_C > w[0]) & (times_C < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_C[:, :, m_C], axis=2)
        bin_idx += 1
        
    return binned_data

def create_pseudo_trials(X, y, n_splits=10, count=4, repeats=20, random_state=1):
    """
    Replicates CoSMoMVPA's cosmo_average_samples.
    Splits data into 'n_splits' chunks. Within each chunk and class, averages 'count' trials 'repeats' times.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    X_pseudo = []
    y_pseudo = []
    chunks = []
    
    np.random.seed(random_state)
    
    for chunk_idx, (_, fold_indices) in enumerate(skf.split(X, y)):
        fold_y = y[fold_indices]
        fold_X = X[fold_indices]
        
        for c in np.unique(fold_y):
            c_idx = np.where(fold_y == c)[0]
            replace = len(c_idx) < count
            
            for _ in range(repeats):
                samp_idx = np.random.choice(c_idx, size=count, replace=replace)
                X_pseudo.append(np.mean(fold_X[samp_idx], axis=0))
                y_pseudo.append(c)
                chunks.append(chunk_idx)
                
    return np.array(X_pseudo), np.array(y_pseudo), np.array(chunks)

def run_decoding(max_pairs=None):
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    os.makedirs(deriv_dir, exist_ok=True)
    
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    total_to_run = len(pairs_to_run)
    
    for p_idx, pair in enumerate(pairs_to_run):
        print(f'Loading pair {p_idx + 1} of {total_to_run} (ID: {pair})')
        sub_str = f'sub-{pair:02d}'
        
        events_file = os.path.join(path_to_data, sub_str, 'eeg', f'{sub_str}_task-RPS_events.tsv')
        if not os.path.exists(events_file): continue
        events = pd.read_csv(events_file, sep='\t')
        
        ev_p1 = events[['player1_resp', 'player2_resp', 'outcome']].values
        ev_p2_outcome = ev_p1[:, 2].copy()
        ev_p2_outcome[ev_p1[:, 2] == 2] = 3
        ev_p2_outcome[ev_p1[:, 2] == 3] = 2
        ev_p2 = np.column_stack([ev_p1[:, 1], ev_p1[:, 0], ev_p2_outcome])
        
        def build_history(ev_array):
            history = np.full((num_trials, 2), np.nan)
            history[1:, :] = ev_array[:-1, :2]
            return np.column_stack([ev_array, history])
            
        behav_p1 = build_history(ev_p1)
        behav_p2 = build_history(ev_p2)
        all_behav = [behav_p1, behav_p2]
        
        for ppt in [1, 2]:
            print(f'   ppt {ppt}')
            epoch_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
            if not os.path.exists(epoch_file):
                print(f"   Missing epoch file for player {ppt}")
                continue
                
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

            # Assign montage – no renaming needed
            montage = mne.channels.make_standard_montage('biosemi64')
            epochs.set_montage(montage, on_missing='ignore')

            epochs.set_eeg_reference('average', projection=False, verbose=False)
            
            rem_idx = np.arange(0, 480, 40)
            sel_idx = np.setdiff1d(np.arange(num_trials), rem_idx)
            
            behav_data = all_behav[ppt-1][sel_idx, :]
            epochs_sel = epochs[sel_idx]
            
            binned_data = get_time_bins(epochs_sel)
            
            # --- Build custom neighbor lists using original BioSemi coordinates ---
            ch_names_epoch = epochs_sel.ch_names
            # Ensure all channels have positions
            missing = [ch for ch in ch_names_epoch if ch not in pos_dict]
            if missing:
                raise ValueError(f"Missing positions for channels: {missing}")
            
            n_chan = len(ch_names_epoch)
            dist_mat = np.zeros((n_chan, n_chan))
            for i, ch1 in enumerate(ch_names_epoch):
                for j, ch2 in enumerate(ch_names_epoch):
                    dist_mat[i, j] = np.linalg.norm(pos_dict[ch1] - pos_dict[ch2])
            
            # For each channel, find the 3 closest other channels (excluding itself)
            neighbor_lists = []
            for i in range(n_chan):
                distances = dist_mat[i].copy()
                distances[i] = np.inf          # exclude self
                closest_3 = np.argsort(distances)[:3]   # indices of 3 nearest neighbours
                neighbor_list = [i] + list(closest_3)   # include the center channel
                neighbor_lists.append(neighbor_list)
            # --------------------------------------------------------------
            
            decoding_accuracy = []
            searchlight_acc = []
            targets_map = [0, 1, 3, 4]   # columns in behav_data: self, other, selfp, otherp
            
            for test_idx in targets_map:
                y = behav_data[:, test_idx]
                valid_mask = ~np.isnan(y) & (y > 0)
                y_valid = y[valid_mask]
                X_valid = binned_data[valid_mask]
                
                if len(y_valid) < 10:
                    decoding_accuracy.append(np.full(20, np.nan))
                    searchlight_acc.append(np.full((num_chan, 20), np.nan))
                    continue
                
                # Apply Pseudo-trial averaging (using fixed seed to match MATLAB)
                X_pseudo, y_pseudo, chunks_pseudo = create_pseudo_trials(
                    X_valid, y_valid, n_splits=10, count=4, repeats=20, random_state=1
                )
                
                # Use LeaveOneGroupOut with the chunks generated by the pseudo-trial function
                cv = LeaveOneGroupOut()
                clf = LinearDiscriminantAnalysis()
                
                acc_time = np.zeros(20)
                for t in range(20):
                    X_t = X_pseudo[:, :, t]
                    acc_time[t] = cross_val_score(clf, X_t, y_pseudo, groups=chunks_pseudo, cv=cv, n_jobs=-1).mean()
                decoding_accuracy.append(acc_time)
                
                sl_acc = np.zeros((num_chan, 20))
                
                # Searchlight using custom neighbour lists
                for ch in range(num_chan):
                    ch_idx = neighbor_lists[ch]   # list of indices (including center)
                    for t in range(20):
                        X_t_sl = X_pseudo[:, ch_idx, t]
                        sl_acc[ch, t] = cross_val_score(clf, X_t_sl, y_pseudo, groups=chunks_pseudo, cv=cv).mean()
                searchlight_acc.append(sl_acc)
                
            out_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            with open(out_file, 'wb') as f:
                pickle.dump({'decoding': decoding_accuracy, 'searchlight': searchlight_acc, 'ch_names': ch_names_epoch}, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    run_decoding(max_pairs=args.test_pairs)