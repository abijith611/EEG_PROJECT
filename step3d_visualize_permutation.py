"""
Visualize permutation test results (Cohen's d and significance) from CSV files.
For each classifier, creates a 2x2 figure with each condition's effect size across time.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

plot_dir = os.path.join('EEG-PROJECT', 'results', 'plots')
os.makedirs(plot_dir, exist_ok=True)

deriv_dir = 'project/ds006761/derivatives'

time_bins = np.arange(1, 21)
time_sec = np.linspace(0.125, 4.875, 20)

# Condition order (matching the paper)
conditions = ['Own response', "Opponent's response",
              'Own previous response', "Opponent's previous response"]
titles = ['A) Own response', "B) Opponent's response",
          'C) Own previous response', "D) Opponent's previous response"]

def plot_classifier_combined(csv_file):
    """Create a 2x2 figure for one classifier's four conditions."""
    df = pd.read_csv(csv_file)
    classifier = df['classifier'].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        cond_df = df[df['condition'] == cond].sort_values('time_bin')
        t = cond_df['time_bin'].values
        d = cond_df['d_diff'].values
        p_corr = cond_df['p_diff_corr'].values

        ax.plot(t, d, 'o-', color='blue', linewidth=2, markersize=4)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

        # Highlight significant bins
        sig_mask = p_corr < 0.05
        if np.any(sig_mask):
            ax.fill_between(t, -1, 1, where=sig_mask, color='red', alpha=0.2)

        ax.set_title(titles[idx], fontweight='bold', fontsize=12)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0.5, 20.5)
        ax.grid(True, alpha=0.3)

        if idx >= 2:  # bottom row
            ax.set_xlabel('Time (s)')
        if idx % 2 == 0:  # left column
            ax.set_ylabel("Cohen's d\n(Winners – Losers)")

    # Set common x-ticks (seconds) for bottom plots
    for ax in axes[2:]:
        ax.set_xticks([1, 5, 9, 13, 17])
        ax.set_xticklabels(['0', '1', '2', '3', '4'])

    plt.suptitle(f'Classifier: {classifier.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # make room for suptitle
    out_file = os.path.join(plot_dir, f'permutation_{classifier}.png')
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved combined figure: {out_file}")

def main():
    pattern = os.path.join(deriv_dir, 'permutation_stats_*.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        print("No permutation stats CSV files found.")
        return

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        plot_classifier_combined(csv_file)

if __name__ == '__main__':
    main()