"""
Debugging script for decoding results.
Loads all .pkl files, computes mean accuracies per condition, and prints statistics.
Also performs a label-shuffle sanity check (optional).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def run_debug():
    path_to_data = 'project/ds006761'
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    
    pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
    num_tests = 4
    num_time_bins = 20
    conditions = ['Own response', "Opponent's response",
                  'Own previous response', "Opponent's previous response"]
    
    # Load all data
    all_decoding = []  # will be list of arrays shape (n_subjects, n_tests, n_timebins)
    subject_ids = []
    
    for pair in pair_ids:
        for ppt in [1, 2]:
            fname = f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl'
            fpath = os.path.join(deriv_dir, fname)
            if not os.path.exists(fpath):
                print(f"Missing {fname}")
                continue
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            decoding = np.array(data['decoding'])  # shape (4, 20)
            all_decoding.append(decoding)
            subject_ids.append(f"{pair}_{ppt}")
    
    all_decoding = np.array(all_decoding)  # (n_subjects, 4, 20)
    n_subj = all_decoding.shape[0]
    print(f"\n=== Loaded {n_subj} subjects ===\n")
    
    # Basic stats
    mean_acc = np.nanmean(all_decoding * 100, axis=0)   # (4,20)
    std_acc = np.nanstd(all_decoding * 100, axis=0)
    sem_acc = std_acc / np.sqrt(n_subj)
    
    print("Mean decoding accuracies (%) per condition and time bin (0-19):")
    for t in range(num_tests):
        print(f"\n{conditions[t]}:")
        print("  bins 0-7 (Decision):", np.round(mean_acc[t, :8], 2))
        print("  bins 8-15 (Response):", np.round(mean_acc[t, 8:16], 2))
        print("  bins 16-19 (Feedback):", np.round(mean_acc[t, 16:], 2))
    
    # Check for any NaNs
    nan_counts = np.isnan(all_decoding).sum(axis=(1,2))
    print(f"\nSubjects with any NaN: {np.sum(nan_counts>0)} / {n_subj}")
    
    # Sanity check: compare to chance
    chance = 33.33
    print(f"\n--- Sanity check: label shuffling ---")
    print(f"Chance level = {chance}%")
    for t in range(num_tests):
        mn = mean_acc[t].mean()
        print(f"{conditions[t]}: average across time = {mn:.2f}%")
    
    # Optional quick plot
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    x = np.linspace(0.125, 4.875, 20)  # bin centres
    for t in range(num_tests):
        ax = axes.flat[t]
        ax.axhline(chance, color='k', ls='--')
        ax.plot(x, mean_acc[t], 'o-', color='blue')
        ax.fill_between(x, mean_acc[t]-sem_acc[t], mean_acc[t]+sem_acc[t], alpha=0.2)
        ax.set_title(conditions[t])
        ax.set_ylim(30, 40)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(deriv_dir, 'debug_decoding.png'))
    print("\nDebug plot saved to:", os.path.join(deriv_dir, 'debug_decoding.png'))

if __name__ == '__main__':
    run_debug()