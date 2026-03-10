"""
Permutation‑based statistics for decoding results (winners vs. losers).
For each classifier, condition, and time bin, computes:
  - One‑sample permutation test (against chance 33.33%) for winners and losers separately.
  - Independent‑samples permutation test (winners vs. losers).
  - Cohen's d effect sizes.
  - FDR correction across time bins for each test family.
Results are printed and saved as CSV files.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import permutation_test, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import glob

path_to_data = 'project/ds006761'
deriv_dir = os.path.join(path_to_data, 'derivatives')
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_tests = 4
num_time_bins = 20
conditions = ['Own response', "Opponent's response",
              'Own previous response', "Opponent's previous response"]
chance = 33.33

def cohen_d(x, y):
    """Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def run_permutation_stats(classifier, max_pairs=None):
    """Run permutation tests and effect sizes for one classifier."""
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs_run = len(pairs_to_run)
    
    all_decoding = np.full((num_pairs_run * 2, num_tests, num_time_bins), np.nan)
    winners_idx, losers_idx = [], []
    
    # Load data
    for p_idx, pair in enumerate(pairs_to_run):
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if os.path.exists(events_file):
            events = pd.read_csv(events_file, sep='\t')
            w1, w2 = sum(events['outcome'] == 2), sum(events['outcome'] == 3)
            p1_idx, p2_idx = p_idx * 2, p_idx * 2 + 1
            if w1 > w2:
                winners_idx.append(p1_idx); losers_idx.append(p2_idx)
            else:
                winners_idx.append(p2_idx); losers_idx.append(p1_idx)
        
        for ppt in [1, 2]:
            idx = p_idx * 2 + (ppt - 1)
            file_path = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{classifier}.pkl')
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found")
                continue
            with open(file_path, 'rb') as f: 
                res = pickle.load(f)
            for t_idx in range(num_tests):
                all_decoding[idx, t_idx, :] = res['decoding'][t_idx]
    
    if np.all(np.isnan(all_decoding)):
        print(f"No valid data for classifier {classifier}")
        return
    
    # Prepare results containers
    results = []
    
    for t_idx, cond in enumerate(conditions):
        data_win = all_decoding[winners_idx, t_idx, :] * 100
        data_los = all_decoding[losers_idx, t_idx, :] * 100
        
        # One‑sample permutation tests (against chance)
        p_win = np.ones(num_time_bins) * np.nan
        p_los = np.ones(num_time_bins) * np.nan
        d_win = np.ones(num_time_bins) * np.nan
        d_los = np.ones(num_time_bins) * np.nan
        
        for b in range(num_time_bins):
            # Winners
            win_b = data_win[:, b][~np.isnan(data_win[:, b])]
            if len(win_b) >= 3:
                res = permutation_test((win_b,), lambda x, axis: np.mean(x, axis=axis) - chance,
                                       vectorized=True, n_resamples=10000, alternative='greater')
                p_win[b] = res.pvalue
                d_win[b] = (np.mean(win_b) - chance) / np.std(win_b, ddof=1)
            
            # Losers
            los_b = data_los[:, b][~np.isnan(data_los[:, b])]
            if len(los_b) >= 3:
                res = permutation_test((los_b,), lambda x, axis: np.mean(x, axis=axis) - chance,
                                       vectorized=True, n_resamples=10000, alternative='greater')
                p_los[b] = res.pvalue
                d_los[b] = (np.mean(los_b) - chance) / np.std(los_b, ddof=1)
        
        # Independent‑samples permutation test (winners vs losers)
        p_diff = np.ones(num_time_bins) * np.nan
        d_diff = np.ones(num_time_bins) * np.nan
        for b in range(num_time_bins):
            win_b = data_win[:, b][~np.isnan(data_win[:, b])]
            los_b = data_los[:, b][~np.isnan(data_los[:, b])]
            if len(win_b) >= 3 and len(los_b) >= 3:
                res = permutation_test((win_b, los_b), lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                                       vectorized=True, n_resamples=10000, alternative='two-sided')
                p_diff[b] = res.pvalue
                d_diff[b] = cohen_d(win_b, los_b)
        
        # FDR correction across time bins
        # For winners
        valid = ~np.isnan(p_win)
        if np.any(valid):
            rejected, p_corr = fdrcorrection(p_win[valid])
            p_win_corr = np.full(num_time_bins, np.nan)
            p_win_corr[valid] = p_corr
        else:
            p_win_corr = p_win.copy()
        
        # For losers
        valid = ~np.isnan(p_los)
        if np.any(valid):
            rejected, p_corr = fdrcorrection(p_los[valid])
            p_los_corr = np.full(num_time_bins, np.nan)
            p_los_corr[valid] = p_corr
        else:
            p_los_corr = p_los.copy()
        
        # For difference
        valid = ~np.isnan(p_diff)
        if np.any(valid):
            rejected, p_corr = fdrcorrection(p_diff[valid])
            p_diff_corr = np.full(num_time_bins, np.nan)
            p_diff_corr[valid] = p_corr
        else:
            p_diff_corr = p_diff.copy()
        
        # Store results for this condition
        for b in range(num_time_bins):
            results.append({
                'classifier': classifier,
                'condition': cond,
                'time_bin': b+1,
                'n_winners': np.sum(~np.isnan(data_win[:, b])),
                'n_losers': np.sum(~np.isnan(data_los[:, b])),
                'mean_win': np.nanmean(data_win[:, b]),
                'mean_los': np.nanmean(data_los[:, b]),
                'p_win': p_win[b],
                'p_win_corr': p_win_corr[b],
                'd_win': d_win[b],
                'p_los': p_los[b],
                'p_los_corr': p_los_corr[b],
                'd_los': d_los[b],
                'p_diff': p_diff[b],
                'p_diff_corr': p_diff_corr[b],
                'd_diff': d_diff[b]
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    out_file = os.path.join(deriv_dir, f'permutation_stats_{classifier}.csv')
    df.to_csv(out_file, index=False)
    print(f"\nSaved permutation results for {classifier} to {out_file}")
    
    # Print summary
    print(f"\n=== Permutation test results for {classifier} ===")
    for cond in conditions:
        cond_df = df[df['condition'] == cond]
        print(f"\n{cond}:")
        print("Time\tp_diff_corr\td_diff")
        for _, row in cond_df.iterrows():
            if not np.isnan(row['p_diff_corr']):
                print(f"{int(row['time_bin'])}\t{row['p_diff_corr']:.4f}\t{row['d_diff']:.3f}")

def main(max_pairs=None):
    # Find all available classifiers
    pattern = os.path.join(deriv_dir, 'pair-*_player-*_task-RPS_decoding_*.pkl')
    files = glob.glob(pattern)
    classifiers = set()
    for f in files:
        base = os.path.basename(f)
        suffix = base.split('decoding_')[1].replace('.pkl', '')
        classifiers.add(suffix)
    
    if not classifiers:
        print("No decoding files found.")
        return
    
    classifiers = sorted(classifiers)
    print(f"Found classifiers: {classifiers}")
    
    for clf in classifiers:
        print(f"\n--- Processing classifier: {clf} ---")
        run_permutation_stats(clf, max_pairs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    main(max_pairs=args.test_pairs)