"""
Debugging script for decoding results.
Loads all .pkl files for each classifier, computes mean accuracies per condition,
and prints statistics. Provides direct overall Winner vs Loser comparison.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

path_to_data = 'project/ds006761'
deriv_dir = os.path.join(path_to_data, 'derivatives')

pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_tests = 4
num_time_bins = 20
conditions = ['Own response', "Opponent's response",
              'Own previous response', "Opponent's previous response"]

def load_data_for_classifier(clf_suffix):
    """Load all decoding files for a given classifier suffix (e.g., 'svm', 'lda')."""
    pattern = os.path.join(deriv_dir, f'pair-*_player-*_task-RPS_decoding_{clf_suffix}.pkl')
    files = glob.glob(pattern)
    all_decoding = []
    subject_ids = []
    winners_idx = []
    losers_idx = []
    
    for fpath in sorted(files):
        # Extract pair and player from filename
        base = os.path.basename(fpath)
        parts = base.split('_')
        pair = int(parts[0].split('-')[1])
        player = int(parts[1].split('-')[1])
        
        # Determine winner for this pair
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        winner_ppt = None
        if os.path.exists(events_file):
            events = pd.read_csv(events_file, sep='\t')
            w1 = sum(events['outcome'] == 2)
            w2 = sum(events['outcome'] == 3)
            winner_ppt = 1 if w1 > w2 else 2
        
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        decoding = np.array(data['decoding'])  # shape (4,20)
        
        current_idx = len(all_decoding)
        all_decoding.append(decoding)
        subject_ids.append(f"{pair}_{player}")
        
        if winner_ppt is not None:
            if player == winner_ppt:
                winners_idx.append(current_idx)
            else:
                losers_idx.append(current_idx)
    
    if not all_decoding:
        return None, None, None, None
    all_decoding = np.array(all_decoding)  # (n_subj,4,20)
    return all_decoding, subject_ids, winners_idx, losers_idx

def run_debug():
    # Find all unique classifier suffixes
    pattern = os.path.join(deriv_dir, 'pair-*_player-*_task-RPS_decoding_*.pkl')
    files = glob.glob(pattern)
    clf_suffixes = set()
    for f in files:
        base = os.path.basename(f)
        # extract suffix after 'decoding_'
        suffix = base.split('decoding_')[1].replace('.pkl', '')
        clf_suffixes.add(suffix)
    
    if not clf_suffixes:
        print("No decoding files found.")
        return
    
    print("\n" + "="*75)
    print(" DEBUG DECODING STATISTICS FOR ALL CLASSIFIERS")
    print("="*75)
    
    for clf in sorted(clf_suffixes):
        print(f"\n--- Classifier: {clf.upper()} ---")
        data = load_data_for_classifier(clf)
        if data[0] is None:
            print("  No data found.")
            continue
        all_decoding, subject_ids, winners_idx, losers_idx = data
        n_subj = all_decoding.shape[0]
        print(f"  Loaded {n_subj} subjects")
        
        # Basic stats
        mean_acc = np.nanmean(all_decoding * 100, axis=0)   # (4,20)
        std_acc = np.nanstd(all_decoding * 100, axis=0)
        sem_acc = std_acc / np.sqrt(n_subj)
        
        # Winner / Loser stats
        mean_win = np.nanmean(all_decoding[winners_idx] * 100, axis=0) if winners_idx else None
        mean_los = np.nanmean(all_decoding[losers_idx] * 100, axis=0) if losers_idx else None
        
        # Check for any NaNs
        nan_counts = np.isnan(all_decoding).sum(axis=(1,2))
        print(f"  Subjects with any NaN: {np.sum(nan_counts>0)} / {n_subj}")
        
        # OVERALL AVERAGES COMPARISON
        chance = 33.33
        print("\n  OVERALL ACCURACY (Average across all 5 seconds)")
        print(f"  Chance level = {chance}%")
        
        for t in range(num_tests):
            mn_all = mean_acc[t].mean()
            print(f"\n  {conditions[t]}:")
            print(f"    Overall: {mn_all:.2f}%")
            if mean_win is not None and mean_los is not None:
                mn_win_val = mean_win[t].mean()
                mn_los_val = mean_los[t].mean()
                diff = mn_win_val - mn_los_val
                print(f"    Winners: {mn_win_val:.2f}%")
                print(f"    Losers:  {mn_los_val:.2f}%")
                print(f"    -> Diff: {diff:+.2f}%")
        
        # Optional quick plot
        fig, axes = plt.subplots(2,2, figsize=(10,8))
        x = np.linspace(0.125, 4.875, 20)  # bin centres
        for t in range(num_tests):
            ax = axes.flat[t]
            ax.axhline(chance, color='k', ls='--')
            ax.plot(x, mean_acc[t], 'o-', color='blue', label='Overall')
            ax.fill_between(x, mean_acc[t]-sem_acc[t], mean_acc[t]+sem_acc[t], color='blue', alpha=0.2)
            if mean_win is not None and mean_los is not None:
                ax.plot(x, mean_win[t], '^-', color='#0072BD', alpha=0.8, label='Winners')
                ax.plot(x, mean_los[t], 's-', color='#77AC30', alpha=0.8, label='Losers')
                ax.legend(loc='lower right', fontsize=8)
            ax.set_title(conditions[t])
            ax.set_ylim(30, 40)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Accuracy (%)')
        plt.tight_layout()
        out_plot = os.path.join(deriv_dir, f'debug_decoding_{clf}.png')
        plt.savefig(out_plot)
        print(f"\n  Debug plot saved to: {out_plot}")

if __name__ == '__main__':
    run_debug()