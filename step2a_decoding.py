"""
Decoding script (multiple classifiers)
  - Decode own & opponent's response for current & previous trial
  - Choose from: svm (SGDClassifier hinge), lda, logistic, ridge
  - Uses stratified 10‑fold cross‑validation with shuffling,
    preserving the group (chunk) structure of pseudo‑trials
  - Parallelised searchlight (optional) with progress feedback
  - Saves results as pair-XX_player-X_task-RPS_decoding_<clf>.pkl
"""

import os
import mne
import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import pickle
from joblib import Parallel, delayed
import time
import argparse

# ========== USER ADJUSTABLE PARAMETERS ==========
N_JOBS_SEARCHLIGHT = -1   # Use all CPU cores for searchlight. Set to 1 to disable parallelism.
SKIP_SEARCHLIGHT = False  # Set to True to skip searchlight (faster for testing).
DEFAULT_CLASSIFIERS = ['svm', 'lda', 'logistic', 'ridge']  # can be overridden by command line
# =================================================

# Optional: tqdm for fancy progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

path_to_data = 'project/ds006761'
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

# Load exact 3D coordinates from the provided biosemi64.mat
try:
    biosemi_mat = scipy.io.loadmat('biosemi64.mat')
    orig_coords = biosemi_mat['biosemi64']
    pos_dict = {label: orig_coords[i] for i, label in enumerate(matlab_layout_labels)}
    print("Loaded biosemi64.mat coordinates.")
except Exception as e:
    pos_dict = None
    print("Could not load biosemi64.mat, using fallback neighbour lists.")

def get_time_bins(epochs):
    print("   Binning data into 250ms windows...")
    times = epochs.times
    data = epochs.get_data() * 1e6  # Convert to microvolts 
    
    mask_A = (times >= -0.2) & (times <= 2.0)
    mask_B = (times >= 1.8) & (times <= 4.0)
    mask_C = (times >= 3.8) & (times <= 5.0)
    
    data_A, data_B, data_C = data[:, :, mask_A], data[:, :, mask_B], data[:, :, mask_C]
    times_A = times[mask_A]
    
    mask_base = (times_A >= -0.2) & (times_A <= 0)
    baseline_A = np.mean(data_A[:, :, mask_base], axis=2, keepdims=True)
    baseline_B = np.mean(data_B[:, :, mask_base], axis=2, keepdims=True)
    baseline_C = np.mean(data_C[:, :, mask_base[:data_C.shape[2]]], axis=2, keepdims=True)
    
    data_A -= baseline_A
    data_B -= baseline_B
    data_C -= baseline_C
    
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
    print(f"   Binning done. Shape: {binned_data.shape}")
    return binned_data

def create_pseudo_trials(X, y, seed):
    from sklearn.model_selection import StratifiedKFold
    print(f"      Creating pseudo-trials with seed {seed}...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    X_pseudo, y_pseudo, chunks = [], [], []
    np.random.seed(seed)
    total_chunks = 0
    for chunk_idx, (_, fold_indices) in enumerate(skf.split(X, y)):
        fold_y, fold_X = y[fold_indices], X[fold_indices]
        for c in np.unique(fold_y):
            c_idx = np.where(fold_y == c)[0]
            replace = len(c_idx) < 4
            for _ in range(20):
                samp_idx = np.random.choice(c_idx, size=4, replace=replace)
                X_pseudo.append(np.mean(fold_X[samp_idx], axis=0))
                y_pseudo.append(c)
                chunks.append(chunk_idx)
                total_chunks += 1
    print(f"      Created {total_chunks} pseudo-trials.")
    return np.array(X_pseudo), np.array(y_pseudo), np.array(chunks)

def get_classifier(name, seed):
    """Return a pipeline (scaler + classifier) for the given name."""
    if name == 'svm':
        clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001,
                            max_iter=1000, tol=1e-3, random_state=seed, n_jobs=1)
    elif name == 'lda':
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.01)
    elif name == 'logistic':
        clf = LogisticRegression(l1_ratio=0, C=1.0, solver='lbfgs',
                                 max_iter=1000, random_state=seed)
    elif name == 'ridge':
        clf = RidgeClassifier(alpha=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return make_pipeline(StandardScaler(), clf)

def compute_searchlight_for_condition(X_ps, y_ps, groups, neighbor_lists, pipeline, cv, n_jobs_outer=N_JOBS_SEARCHLIGHT):
    """Parallel computation of searchlight accuracies with progress reporting."""
    n_chan = X_ps.shape[1]
    n_times = X_ps.shape[2]
    total_pairs = n_chan * n_times

    print(f"      Starting searchlight: {n_chan} channels × {n_times} time bins = {total_pairs} combinations.")
    print(f"      Using {n_jobs_outer} parallel workers.")

    def score_one(ch, t):
        X_chan = X_ps[:, neighbor_lists[ch], t]
        scores = cross_val_score(pipeline, X_chan, y_ps, groups=groups, cv=cv, n_jobs=1)
        return np.mean(scores)

    ch_t_pairs = [(ch, t) for ch in range(n_chan) for t in range(n_times)]

    if HAS_TQDM:
        results = Parallel(n_jobs=n_jobs_outer)(
            delayed(score_one)(ch, t) for ch, t in tqdm(ch_t_pairs, desc="      Searchlight", unit="pair")
        )
    else:
        results = []
        step = max(1, total_pairs // 10)
        for i, (ch, t) in enumerate(ch_t_pairs):
            results.append(score_one(ch, t))
            if (i+1) % step == 0:
                print(f"      ... {i+1}/{total_pairs} done")
        print(f"      Searchlight completed.")

    sl_acc = np.array(results).reshape(n_chan, n_times)
    return sl_acc

def run_decoding(max_pairs=None, classifiers=None):
    if classifiers is None:
        classifiers = DEFAULT_CLASSIFIERS
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs = len(pairs_to_run)
    
    # Mapping from column index to descriptive name
    target_names = {0: 'self', 1: 'other', 3: 'self_prev', 4: 'other_prev'}
    target_cols = [0, 1, 3, 4]  # order as in original paper
    
    for p_idx, pair in enumerate(pairs_to_run):
        print(f'Loading pair {p_idx + 1} of {num_pairs} (ID: {pair})')
        sub_str = f'sub-{pair:02d}'
        events = pd.read_csv(os.path.join(path_to_data, sub_str, 'eeg', f'{sub_str}_task-RPS_events.tsv'), sep='\t')
        
        ev_p1 = events[['player1_resp', 'player2_resp', 'outcome']].values
        hist_p1 = np.full((num_trials, 2), np.nan)
        hist_p1[1:] = ev_p1[:-1, :2]
        behav_p1 = np.column_stack([ev_p1, hist_p1])
        
        ev_p2_raw = ev_p1[:, [1, 0, 2]]
        ev_p2_raw[ev_p1[:, 2] == 2, 2] = 3
        ev_p2_raw[ev_p1[:, 2] == 3, 2] = 2
        hist_p2 = np.full((num_trials, 2), np.nan)
        hist_p2[1:] = ev_p2_raw[:-1, :2]
        behav_p2 = np.column_stack([ev_p2_raw, hist_p2])

        for ppt, behav in zip([1, 2], [behav_p1, behav_p2]):
            print(f'   Processing participant {ppt}')
            epoch_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
            if not os.path.exists(epoch_file):
                print(f"      File {epoch_file} not found, skipping.")
                continue
            epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
            epochs.set_eeg_reference('average', projection=False, verbose=False)
            
            sel_idx = np.setdiff1d(np.arange(num_trials), np.arange(0, 480, 40))
            behav_data = behav[sel_idx]
            binned_data = get_time_bins(epochs[sel_idx])
            
            ch_names = epochs.ch_names
            if pos_dict is not None:
                print("      Computing neighbour lists from coordinates...")
                dist_mat = scipy.spatial.distance.cdist(
                    [pos_dict[c] for c in ch_names],
                    [pos_dict[c] for c in ch_names]
                )
                neighbor_lists = [np.argsort(dist_mat[i])[0:5] for i in range(num_chan)]
            else:
                print("      Using fallback neighbour lists (by index).")
                neighbor_lists = [np.arange(max(0, i-2), min(num_chan, i+3)) for i in range(num_chan)]

            # Loop over requested classifiers
            for clf_name in classifiers:
                print(f"      Running classifier: {clf_name}")
                decoding_results = []
                searchlight_results = []
                
                for target_col in target_cols:
                    target_name = target_names[target_col]
                    print(f"         Decoding target: {target_name}")
                    y = behav_data[:, target_col]
                    valid = ~np.isnan(y) & (y > 0)
                    print(f"            Valid trials: {np.sum(valid)}")
                    seed = pair * 10 + ppt
                    X_ps, y_ps, groups = create_pseudo_trials(binned_data[valid], y[valid], seed)
                    
                    unique_classes = np.unique(y_ps)
                    if len(unique_classes) < 2:
                        print(f"            WARNING: Only {len(unique_classes)} class(es) in y_ps. Skipping this target.")
                        decoding_results.append([np.nan]*20)
                        if not SKIP_SEARCHLIGHT:
                            searchlight_results.append(np.full((num_chan,20), np.nan))
                        continue
                    
                    pipeline = get_classifier(clf_name, seed)
                    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
                    
                    # Time‑resolved decoding (simple average over channels)
                    print("            Running time-resolved decoding...")
                    acc_time = []
                    for t in range(20):
                        print(f"               Time bin {t+1}/20... ", end='', flush=True)
                        start_t = time.time()
                        try:
                            scores = cross_val_score(pipeline, X_ps[:, :, t], y_ps, groups=groups, cv=cv, n_jobs=1)
                            mean_score = np.mean(scores)
                            acc_time.append(mean_score)
                            elapsed = time.time() - start_t
                            print(f"done in {elapsed:.2f}s (mean acc={mean_score:.4f})")
                        except Exception as e:
                            print(f"ERROR: {e}")
                            acc_time.append(np.nan)
                    decoding_results.append(acc_time)
                    print(f"            Time-resolved done. Overall mean acc: {np.nanmean(acc_time):.4f}")
                    
                    # Searchlight – parallelised (skip if flag set)
                    if SKIP_SEARCHLIGHT:
                        print("            Skipping searchlight as requested.")
                        searchlight_results.append(np.full((num_chan,20), np.nan))
                    else:
                        sl_acc = compute_searchlight_for_condition(
                            X_ps, y_ps, groups, neighbor_lists, pipeline, cv
                        )
                        searchlight_results.append(sl_acc)
                        print("            Searchlight done.")
                    
                # Save results with classifier name in filename
                out_file = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{clf_name}.pkl')
                if os.path.exists(out_file):
                    print(f"      Skipping classifier {clf_name} – output already exists.")
                    continue

                print(f"      Running classifier: {clf_name}")
                with open(out_file, 'wb') as f:
                    pickle.dump({
                        'decoding': decoding_results,
                        'searchlight': searchlight_results,
                        'ch_names': ch_names,
                        'classifier': clf_name
                    }, f)
                print(f"      Saved results for classifier {clf_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    parser.add_argument('--classifiers', nargs='+', default=DEFAULT_CLASSIFIERS,
                        choices=['svm', 'lda', 'logistic', 'ridge'],
                        help='List of classifiers to run')
    parser.add_argument('--n_jobs', type=int, default=N_JOBS_SEARCHLIGHT,
                        help='Number of parallel jobs for searchlight')
    parser.add_argument('--skip_searchlight', action='store_true',
                        help='Skip searchlight computation')
    args = parser.parse_args()
    
    N_JOBS_SEARCHLIGHT = args.n_jobs
    SKIP_SEARCHLIGHT = args.skip_searchlight
    print(f"Using {N_JOBS_SEARCHLIGHT} parallel workers for searchlight.")
    print(f"SKIP_SEARCHLIGHT = {SKIP_SEARCHLIGHT}")
    print(f"Classifiers to run: {args.classifiers}")
    
    run_decoding(max_pairs=args.test_pairs, classifiers=args.classifiers)