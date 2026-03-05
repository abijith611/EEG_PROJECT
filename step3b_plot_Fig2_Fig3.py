"""
Plot Decoding Results and Compute Bayes Factors
Replaces R's ttestBF with `pingouin.bayesfactor_ttest`

Libraries needed: mne, matplotlib, pingouin, numpy, scipy
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import mne
from scipy import stats

path_to_data = 'project/ds006761'
plot_dir = os.path.join(path_to_data, 'derivatives', 'plots')
os.makedirs(plot_dir, exist_ok=True)

pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_tests = 4
num_time_bins = 20

def plot_decoding(max_pairs=None):
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs_run = len(pairs_to_run)
    
    all_decoding = np.full((num_pairs_run * 2, num_tests, num_time_bins), np.nan)
    searchlight_all = np.full((num_pairs_run * 2, num_tests, 64, num_time_bins), np.nan)
    
    ch_names = None
    
    for p_idx, pair in enumerate(pairs_to_run):
        for ppt in [1, 2]:
            idx = p_idx * 2 + (ppt - 1)
            file_path = os.path.join(path_to_data, 'derivatives', f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            if not os.path.exists(file_path): continue
                
            with open(file_path, 'rb') as f:
                res = pickle.load(f)
                
            ch_names = res['ch_names']
            for t_idx in range(num_tests):
                all_decoding[idx, t_idx, :] = res['decoding'][t_idx]
                searchlight_all[idx, t_idx, :, :] = res['searchlight'][t_idx]

    bfs = np.zeros((num_tests, num_time_bins))
    for t in range(num_tests):
        for w in range(num_time_bins):
            data_slice = all_decoding[:, t, w]
            data_slice = data_slice[~np.isnan(data_slice)] # Drop NaNs if any missing
            if len(data_slice) > 2:
                # FIX: Calculate T-value first, then pass T-value and sample size (nx). 
                # Explicitly use 0.707 (the numeric value for 'medium' prior)
                t_stat, _ = stats.ttest_1samp(data_slice, 1/3)
                bf_val = pg.bayesfactor_ttest(t_stat, nx=len(data_slice), r=0.707)
                bfs[t, w] = bf_val if isinstance(bf_val, float) else 1.0

    fig, axes = plt.subplots(num_tests, 1, figsize=(10, 12))
    titles = ['A) Own response', "B) Opponent's response", 'C) Own previous response', "D) Opponent's previous response"]
    
    for t in range(num_tests):
        ax = axes[t]
        
        # Calculate mean ignoring NaNs
        data_mean = np.nanmean(all_decoding[:, t, :] * 100, axis=0)
        ci = np.nanstd(all_decoding[:, t, :] * 100, axis=0) / np.sqrt(num_pairs_run * 2) * 1.96
        
        x_axis = np.arange(1, 21)
        ax.axhline(33.33, color='k', linestyle='--')
        
        ax.plot(x_axis, data_mean, 'k-', linewidth=2)
        ax.fill_between(x_axis, data_mean - ci, data_mean + ci, color='gray', alpha=0.3)
        
        ax.axvspan(0.5, 8.5, color='b', alpha=0.05)
        ax.axvspan(8.5, 16.5, color='r', alpha=0.05)
        ax.axvspan(16.5, 20.5, color='g', alpha=0.05)
        
        ax.set_ylim(31, 40)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(titles[t])
        
        ax2 = ax.twinx()
        log_bfs = np.log10(bfs[t, :] + 1e-10)
        ax2.scatter(x_axis, log_bfs, c=log_bfs, cmap='coolwarm', s=50)
        ax2.set_ylim(-6, 6)
        ax2.set_ylabel('Log(BF10)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Figure2.png'), dpi=300)
    print("Figure 2 saved.")
    
    if ch_names is not None:
        # Guarantee we use the correct 10-20 names from the Biosemi64 template 
        # so the topomap matches the 3D coords, even if older files were loaded.
        montage = mne.channels.make_standard_montage('biosemi64')
        safe_ch_names = montage.ch_names[:64]
        
        info = mne.create_info(ch_names=safe_ch_names, sfreq=256, ch_types='eeg')
        info.set_montage(montage)
        
        sl_mean = np.nanmean(searchlight_all[:, 0, :, :], axis=0)
        
        fig_topo, axes_topo = plt.subplots(1, 5, figsize=(15, 3))
        selected_bins = [4, 8, 12, 16, 19]
        
        for idx, t_bin in enumerate(selected_bins):
            mne.viz.plot_topomap(sl_mean[:, t_bin], info, axes=axes_topo[idx], show=False, vlim=(0.33, 0.36), cmap='hot')
            axes_topo[idx].set_title(f'Time bin {t_bin}')
            
        plt.savefig(os.path.join(plot_dir, 'Figure2_Topos.png'), dpi=300)
        print("Figure 2 Topos saved.")

if __name__ == '__main__':
    plot_decoding()