"""
Plot Decoding Results and Compute Bayes Factors
Replicates the 2x2 grid layout, phase shading, and time-bin topographies from the paper.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    
    winners_idx = []
    losers_idx = []
    
    # Load decoding data and identify winners/losers
    for p_idx, pair in enumerate(pairs_to_run):
        # Determine Winner/Loser
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if os.path.exists(events_file):
            events = pd.read_csv(events_file, sep='\t')
            w1, w2 = sum(events['outcome'] == 2), sum(events['outcome'] == 3)
            p1_idx = p_idx * 2
            p2_idx = p_idx * 2 + 1
            if w1 > w2:
                winners_idx.append(p1_idx); losers_idx.append(p2_idx)
            else:
                winners_idx.append(p2_idx); losers_idx.append(p1_idx) # Includes ties gracefully
        
        for ppt in [1, 2]:
            idx = p_idx * 2 + (ppt - 1)
            file_path = os.path.join(path_to_data, 'derivatives', f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            if not os.path.exists(file_path): continue
                
            with open(file_path, 'rb') as f: res = pickle.load(f)
            for t_idx in range(num_tests):
                all_decoding[idx, t_idx, :] = res['decoding'][t_idx]
                searchlight_all[idx, t_idx, :, :] = res['searchlight'][t_idx]

    # Pre-calculate Topomap template to avoid errors
    montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(ch_names=montage.ch_names[:64], sfreq=256, ch_types='eeg')
    info.set_montage(montage)

    # -------------------------------------------------------------
    # FIGURE 2: ALL SUBJECTS (2x2 Grid)
    # -------------------------------------------------------------
    fig2 = plt.figure(figsize=(15, 12), layout='constrained')
    # FIX: Added figure=fig2 to silence the layoutgrids warning
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.2)
    titles = ['A) Own response', "B) Opponent's response", 'C) Own previous response', "D) Opponent's previous response"]
    
    for t in range(num_tests):
        row, col = t // 2, t % 2
        inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs2[row, col], height_ratios=[3, 0.5, 0.8], hspace=0.1)
        
        ax_main = fig2.add_subplot(inner_gs[0])
        ax_bf = fig2.add_subplot(inner_gs[1], sharex=ax_main)
        
        data_t = all_decoding[:, t, :]
        data_mean = np.nanmean(data_t * 100, axis=0)
        ci = np.nanstd(data_t * 100, axis=0) / np.sqrt(num_pairs_run * 2) * 1.96
        x_axis = np.arange(1, 21)
        
        # 1. Main Axis
        ax_main.axhline(33.33, color='k', linestyle='--')
        ax_main.plot(x_axis, data_mean, 'k-', linewidth=2)
        ax_main.fill_between(x_axis, data_mean - ci, data_mean + ci, color='gray', alpha=0.3)
        ax_main.axvspan(0.5, 8.5, color='#EDB120', alpha=0.1) # Phase A (Decision)
        ax_main.axvspan(8.5, 16.5, color='#D95319', alpha=0.1) # Phase B (Response)
        ax_main.axvspan(16.5, 20.5, color='#7E2F8E', alpha=0.1) # Phase C (Feedback)
        ax_main.set_ylim(31, 40)
        ax_main.set_ylabel('Accuracy (%)')
        ax_main.set_title(titles[t], loc='left', fontweight='bold')
        ax_main.set_xlim(0, 21)
        ax_main.set_xticks(np.arange(0.5, 21, 4))
        ax_main.set_xticklabels([])
        
        # 2. BF Axis
        bfs = np.ones(num_time_bins) # Changed to np.ones to plot grey when skipped
        for w in range(num_time_bins):
            slice_w = data_t[:, w][~np.isnan(data_t[:, w])]
            if len(slice_w) > 2:
                t_stat, _ = stats.ttest_1samp(slice_w, 1/3)
                bf = pg.bayesfactor_ttest(t_stat, nx=len(slice_w), r=0.707)
                bfs[w] = bf if isinstance(bf, float) else 1.0
                
        log_bfs = np.log10(bfs)
        sc = ax_bf.scatter(x_axis, [0]*20, c=log_bfs, cmap='coolwarm', vmin=-6, vmax=6, s=80)
        ax_bf.set_yticks([])
        ax_bf.set_ylabel('Log(BF10)')
        
        # 3. Topographies (5 bins)
        topo_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=inner_gs[2], wspace=0.05)
        sl_mean = np.nanmean(searchlight_all[:, t, :, :], axis=0)
        timebins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)] # 1-second chunks
        
        for idx, (t_start, t_end) in enumerate(timebins):
            ax_topo = fig2.add_subplot(topo_gs[0, idx])
            topo_data = np.nanmean(sl_mean[:, t_start:t_end], axis=1)
            mne.viz.plot_topomap(topo_data, info, axes=ax_topo, show=False, vlim=(0.33, 0.36), cmap='hot')
            ax_topo.set_title(f'{idx+1}s', pad=0)

    plt.savefig(os.path.join(plot_dir, 'Figure2.png'), dpi=300, bbox_inches='tight')
    print("Figure 2 saved.")

    # -------------------------------------------------------------
    # FIGURE 3: WINNERS VS LOSERS
    # -------------------------------------------------------------
    if len(winners_idx) >= 1: # Lowered threshold to plot even with subset testing
        fig3 = plt.figure(figsize=(15, 12), layout='constrained')
        # FIX: Added figure=fig3 to silence the layoutgrids warning
        gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.2)
        
        for t in range(num_tests):
            row, col = t // 2, t % 2
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs3[row, col], height_ratios=[3, 1], hspace=0.1)
            ax_main = fig3.add_subplot(inner_gs[0])
            ax_bf = fig3.add_subplot(inner_gs[1], sharex=ax_main)
            
            data_win = all_decoding[winners_idx, t, :] * 100
            data_los = all_decoding[losers_idx, t, :] * 100
            
            # 1. Main Axis
            ax_main.axhline(33.33, color='k', linestyle='--')
            for d, c, lbl in zip([data_win, data_los], ['#0072BD', '#77AC30'], ['Winners', 'Losers']):
                mean_d = np.nanmean(d, axis=0)
                ci_d = np.nanstd(d, axis=0) / np.sqrt(len(d)) * 1.96
                ax_main.plot(x_axis, mean_d, color=c, linewidth=2, label=lbl)
                ax_main.fill_between(x_axis, mean_d - ci_d, mean_d + ci_d, color=c, alpha=0.15)
                ax_main.scatter(x_axis, mean_d, color=c, s=30)
                
            ax_main.axvspan(0.5, 8.5, color='#EDB120', alpha=0.1)
            ax_main.axvspan(8.5, 16.5, color='#D95319', alpha=0.1)
            ax_main.axvspan(16.5, 20.5, color='#7E2F8E', alpha=0.1)
            ax_main.set_ylim(31, 40)
            ax_main.set_ylabel('Accuracy (%)')
            ax_main.set_title(titles[t], loc='left', fontweight='bold')
            ax_main.legend(loc='lower right')
            ax_main.set_xlim(0, 21)
            
            # 2. BF Axis (3 rows: Winners, Losers, Diff)
            bfs_win, bfs_los, bfs_diff = np.ones(num_time_bins), np.ones(num_time_bins), np.ones(num_time_bins)
            for w in range(num_time_bins):
                sw = data_win[:, w][~np.isnan(data_win[:, w])] / 100
                sl = data_los[:, w][~np.isnan(data_los[:, w])] / 100
                
                if len(sw) > 2:
                    t_win, _ = stats.ttest_1samp(sw, 1/3)
                    bf_w = pg.bayesfactor_ttest(t_win, nx=len(sw), r=0.707)
                    bfs_win[w] = bf_w if isinstance(bf_w, float) else 1.0
                    
                if len(sl) > 2:
                    t_los, _ = stats.ttest_1samp(sl, 1/3)
                    bf_l = pg.bayesfactor_ttest(t_los, nx=len(sl), r=0.707)
                    bfs_los[w] = bf_l if isinstance(bf_l, float) else 1.0
                    
                if len(sw) > 2 and len(sl) > 2:
                    t_diff, _ = stats.ttest_ind(sw, sl)
                    bf_d = pg.bayesfactor_ttest(t_diff, nx=len(sw), ny=len(sl), r=0.707)
                    bfs_diff[w] = bf_d if isinstance(bf_d, float) else 1.0

            ax_bf.scatter(x_axis, [2]*20, c=np.log10(bfs_win), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            ax_bf.scatter(x_axis, [1]*20, c=np.log10(bfs_los), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            ax_bf.scatter(x_axis, [0]*20, c=np.log10(bfs_diff), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            
            ax_bf.set_yticks([0, 1, 2])
            ax_bf.set_yticklabels(['Diff', 'Losers', 'Winners'])
            ax_bf.set_ylim(-0.5, 2.5)

        plt.savefig(os.path.join(plot_dir, 'Figure3.png'), dpi=300, bbox_inches='tight')
        print("Figure 3 saved.")

if __name__ == '__main__':
    plot_decoding()