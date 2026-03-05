"""
Decoding script:
  - Decode own & opponent's response for current & previous trial
  - Uses Time-binning, Pseudo-trial averaging, and Temporal + Spatial Searchlight

Libraries needed: mne, numpy, pandas, scikit-learn
"""

import os
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_score
import pickle

path_to_data = 'project/ds006761'
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_trials = 480
num_chan = 64

def get_time_bins(epochs):
    times = epochs.times
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    
    mask_A = (times >= -0.2) & (times <= 2.0)
    mask_B = (times >= 1.8) & (times <= 4.0)
    mask_C = (times >= 3.8) & (times <= 5.0)
    mask_base = (times >= -0.2) & (times <= 0)
    
    data_A = data[:, :, mask_A]
    data_B = data[:, :, mask_B]
    data_C = data[:, :, mask_C]
    
    baseline_A = np.mean(data[:, :, mask_base], axis=2, keepdims=True)
    baseline_B = np.mean(data[:, :, mask_base], axis=2, keepdims=True)
    baseline_C = np.mean(data[:, :, mask_base], axis=2, keepdims=True)
    
    data_A -= baseline_A
    data_B -= baseline_B
    data_C -= baseline_C
    
    times_A = times[mask_A]
    times_B = times[mask_B] - 2.0
    times_C = times[mask_C] - 4.0
    
    time_windows_AB = np.array([np.arange(0, 1.76, 0.25), np.arange(0.25, 2.01, 0.25)]).T
    time_windows_C = np.array([np.arange(0, 0.76, 0.25), np.arange(0.25, 1.01, 0.25)]).T
    
    num_tr = data.shape[0]
    binned_data = np.zeros((num_tr, num_chan, len(time_windows_AB)*2 + len(time_windows_C)))
    
    bin_idx = 0
    for w in time_windows_AB:
        m_A = (times_A > w[0]) & (times_A < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_A[:, :, m_A], axis=2)
        bin_idx += 1
    for w in time_windows_AB:
        m_B = (times_B > w[0]) & (times_B < w[1])
        binned_data[:, :, bin_idx] = np.mean(data_B[:, :, m_B], axis=2)
        bin_idx += 1
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
            
            montage = mne.channels.make_standard_montage('biosemi64')
            mapping = {old: new for old, new in zip(epochs.ch_names[:64], montage.ch_names[:64])}
            epochs.rename_channels(mapping)
            epochs.set_montage(montage, on_missing='ignore')
            
            epochs.set_eeg_reference('average', projection=False, verbose=False)
            
            rem_idx = np.arange(0, 480, 40)
            sel_idx = np.setdiff1d(np.arange(num_trials), rem_idx)
            
            behav_data = all_behav[ppt-1][sel_idx, :]
            epochs_sel = epochs[sel_idx]
            
            binned_data = get_time_bins(epochs_sel)
            adj_matrix, ch_names = mne.channels.find_ch_adjacency(epochs_sel.info, ch_type='eeg')
            
            decoding_accuracy = []
            searchlight_acc = []
            targets_map = [0, 1, 3, 4] 
            
            for test_idx in targets_map:
                y = behav_data[:, test_idx]
                valid_mask = ~np.isnan(y) & (y > 0)
                y_valid = y[valid_mask]
                X_valid = binned_data[valid_mask]
                
                if len(y_valid) < 10:
                    decoding_accuracy.append(np.full(20, np.nan))
                    searchlight_acc.append(np.full((num_chan, 20), np.nan))
                    continue
                
                # Apply Pseudo-trial averaging
                X_pseudo, y_pseudo, chunks_pseudo = create_pseudo_trials(
                    X_valid, y_valid, n_splits=10, count=4, repeats=20, random_state=pair
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
                
                for ch in range(num_chan):
                    row = adj_matrix[ch]
                    row_dense = row.toarray().flatten() if hasattr(row, 'toarray') else np.asarray(row).flatten()
                    neighbors = np.where(row_dense)[0]
                    ch_idx = [ch] + [n for n in neighbors if n != ch]
                    
                    for t in range(20):
                        X_t_sl = X_pseudo[:, ch_idx, t]
                        sl_acc[ch, t] = cross_val_score(clf, X_t_sl, y_pseudo, groups=chunks_pseudo, cv=cv).mean()
                searchlight_acc.append(sl_acc)
                
            out_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            with open(out_file, 'wb') as f:
                pickle.dump({'decoding': decoding_accuracy, 'searchlight': searchlight_acc, 'ch_names': ch_names}, f)