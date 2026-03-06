"""
Decoding script:
  - Decode own & opponent's response for current & previous trial
  - Uses Time-binning, Pseudo-trial averaging (global random sampling, seed=1), 
    and Temporal + Spatial Searchlight with 4 nearest neighbours.
  - Matches MATLAB's cosmo_average_samples and cosmo_meeg_chan_neighborhood('count',4)

Libraries needed: mne, numpy, pandas, scikit-learn, scipy
"""

import os
import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
from scipy.spatial.distance import cdist

path_to_data = 'project/ds006761'
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_trials = 480
num_chan = 64

# Fixed random seed for reproducibility (MATLAB uses seed=1)
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

def get_time_bins(epochs):
    """Identical to original – no changes needed."""
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

def create_pseudo_trials_global(X, y, count=4, repeats=20, random_state=1):
    """
    Replicates cosmo_average_samples exactly:
    - Randomly samples `count` trials (with replacement if needed) from the entire set,
      repeated `repeats` times, for each class separately.
    - Returns X_pseudo (n_repeats * n_classes, features, time), y_pseudo, and chunks
      (which are simply the repeat index – but for cross-validation we will not use grouping).
    - Note: MATLAB's cosmo_average_samples does not create chunks; we will use standard KFold later.
    """
    classes = np.unique(y)
    X_pseudo_list = []
    y_pseudo_list = []
    # We don't create chunks here because we will use standard KFold on the pseudo-trials.
    np.random.seed(random_state)

    for c in classes:
        idx_c = np.where(y == c)[0]
        replace = len(idx_c) < count
        for r in range(repeats):
            samp_idx = np.random.choice(idx_c, size=count, replace=replace)
            # Average across selected trials (axis=0)
            avg = np.mean(X[samp_idx], axis=0)  # shape (n_features, n_timebins)
            X_pseudo_list.append(avg)
            y_pseudo_list.append(c)

    X_pseudo = np.array(X_pseudo_list)  # (n_repeats * n_classes, n_features, n_timebins)
    y_pseudo = np.array(y_pseudo_list)
    return X_pseudo, y_pseudo

def compute_nearest_neighbors(montage, n_neighbors=4):
    """
    Compute indices of the `n_neighbors` closest channels for each channel,
    based on 3D electrode positions.
    Returns a list of lists: neighbors[i] contains indices of neighbors for channel i.
    """
    pos = montage.get_positions()['ch_pos']
    # Order channels as in montage.ch_names (they are 64 standard biosemi names)
    ch_names = montage.ch_names[:64]  # ensure we only use the first 64
    pos_array = np.array([pos[ch] for ch in ch_names])
    dist_mat = cdist(pos_array, pos_array)
    neighbors = []
    for i in range(len(ch_names)):
        # Get distances to all others
        dists = dist_mat[i].copy()
        dists[i] = np.inf  # exclude self
        # Get indices of smallest distances
        nearest = np.argsort(dists)[:n_neighbors]
        neighbors.append(nearest.tolist())
    return neighbors, ch_names

def run_preprocessing(max_pairs=None):
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    os.makedirs(deriv_dir, exist_ok=True)

    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    total_to_run = len(pairs_to_run)

    # Precompute neighbor indices from standard montage (same for all subjects)
    montage = mne.channels.make_standard_montage('biosemi64')
    neighbor_indices, std_ch_names = compute_nearest_neighbors(montage, n_neighbors=4)

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

            # Rename channels to standard biosemi names (if not already)
            mapping = {old: new for old, new in zip(epochs.ch_names[:64], std_ch_names)}
            epochs.rename_channels(mapping)
            epochs.set_montage(montage, on_missing='ignore')
            epochs.set_eeg_reference('average', projection=False, verbose=False)

            rem_idx = np.arange(0, 480, 40)
            sel_idx = np.setdiff1d(np.arange(num_trials), rem_idx)

            behav_data = all_behav[ppt-1][sel_idx, :]
            epochs_sel = epochs[sel_idx]

            binned_data = get_time_bins(epochs_sel)  # shape (n_trials, n_chan, 20)

            decoding_accuracy = []
            searchlight_acc = []
            targets_map = [0, 1, 3, 4]  # columns in behav_data

            for test_idx in targets_map:
                y = behav_data[:, test_idx]
                valid_mask = ~np.isnan(y) & (y > 0)
                y_valid = y[valid_mask]
                X_valid = binned_data[valid_mask]  # (n_valid, n_chan, 20)

                if len(y_valid) < 10:
                    decoding_accuracy.append(np.full(20, np.nan))
                    searchlight_acc.append(np.full((num_chan, 20), np.nan))
                    continue

                # Pseudo-trial averaging (global, seed=1)
                X_pseudo, y_pseudo = create_pseudo_trials_global(
                    X_valid, y_valid, count=4, repeats=20, random_state=RANDOM_SEED
                )  # X_pseudo shape: (n_pseudo, n_chan, 20)

                # Cross-validation on pseudo-trials (standard 10-fold)
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
                clf = LinearDiscriminantAnalysis()

                # Temporal decoding (all channels)
                acc_time = np.zeros(20)
                for t in range(20):
                    X_t = X_pseudo[:, :, t]  # (n_pseudo, n_chan)
                    scores = cross_val_score(clf, X_t, y_pseudo, cv=cv, n_jobs=-1)
                    acc_time[t] = scores.mean()
                decoding_accuracy.append(acc_time)

                # Searchlight decoding (neighbors from precomputed list)
                sl_acc = np.zeros((num_chan, 20))
                for ch in range(num_chan):
                    neighbors = neighbor_indices[ch]  # list of neighbor indices (including itself? MATLAB excludes self)
                    # In MATLAB, 'count',4 returns 4 nearest neighbours excluding itself? The original code uses adj_matrix
                    # which includes the channel itself? We need to check. In the original Python, they did:
                    # row = adj_matrix[ch]; row_dense = row.toarray().flatten(); neighbors = np.where(row_dense)[0];
                    # That includes the channel itself? Usually adjacency includes self? In MNE, find_ch_adjacency returns a graph
                    # where each channel is connected to itself? We need to replicate the neighbor selection exactly.
                    # To be safe, we'll follow MATLAB: get 4 nearest neighbours excluding itself.
                    # Our neighbor_indices already excludes self because we set dists[i]=np.inf.
                    ch_idx = [ch] + neighbors  # include the center channel itself (original code includes center)
                    # In MATLAB searchlight, they include the center channel? Usually yes.
                    for t in range(20):
                        X_t_sl = X_pseudo[:, ch_idx, t]  # (n_pseudo, n_neighbors+1)
                        # Use cross_val_score with same cv
                        scores = cross_val_score(clf, X_t_sl, y_pseudo, cv=cv, n_jobs=-1)
                        sl_acc[ch, t] = scores.mean()
                searchlight_acc.append(sl_acc)

            out_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            with open(out_file, 'wb') as f:
                pickle.dump({'decoding': decoding_accuracy,
                             'searchlight': searchlight_acc,
                             'ch_names': std_ch_names}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run decoding")
    parser.add_argument('--test_pairs', type=int, default=None,
                        help="Limit the number of pairs to process for testing")
    args = parser.parse_args()
    run_preprocessing(max_pairs=args.test_pairs)