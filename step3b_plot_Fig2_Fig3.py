"""
Plot Decoding Results and Compute Bayes Factors
Replicates the grid layout, time-bin topographies, and directional Bayes Factors.
Includes the text labels over the phase windows.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
import warnings

R_AVAILABLE = False
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    conv = numpy2ri.converter

    robjects.r('version')
    bf_pkg = importr('BayesFactor')
    R_AVAILABLE = True
    print("R and BayesFactor package found – using exact Bayes factors.")
except Exception as e:
    R_AVAILABLE = False
    print(f"R/BayesFactor not available ({e}). Using pingouin approximation.")
    import pingouin as pg

path_to_data = 'project/ds006761'
root_dir = 'EEG-PROJECT'
plot_dir = os.path.join(root_dir, 'results', 'plots')
os.makedirs(plot_dir, exist_ok=True)

pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_tests = 4
num_time_bins = 20

def calc_bayes_factor(data, mu=1/3, rscale="medium", null_interval="c(0.5, Inf)"):
    if not R_AVAILABLE:
        from scipy import stats
        t_stat, _ = stats.ttest_1samp(data, mu)
        return pg.bayesfactor_ttest(t_stat, nx=len(data), r=rscale)

    try:
        with conv.context():
            robjects.globalenv['data'] = data
            robjects.r(f'bf = BayesFactor::ttestBF(x=data, mu={mu}, rscale="{rscale}", nullInterval={null_interval})')
            bf_val = robjects.r('as.vector(bf[1])')[0]
        return float(bf_val)
    except Exception as e:
        warnings.warn(f"R call failed, falling back to pingouin: {e}")
        from scipy import stats
        t_stat, _ = stats.ttest_1samp(data, mu)
        return pg.bayesfactor_ttest(t_stat, nx=len(data), r=rscale)


def calc_bayes_factor_ind(data_win, data_los, rscale="medium", null_interval="c(-0.5, 0.5)"):
    if not R_AVAILABLE:
        from scipy import stats
        t_stat, _ = stats.ttest_ind(data_win, data_los)
        return pg.bayesfactor_ttest(t_stat, nx=len(data_win), ny=len(data_los), r=rscale)

    try:
        with conv.context():
            robjects.globalenv['data_win'] = data_win
            robjects.globalenv['data_los'] = data_los
            robjects.r(f'bf = BayesFactor::ttestBF(x=data_win, y=data_los, rscale="{rscale}", nullInterval={null_interval})')
            bf_val = robjects.r('as.vector(bf[2])')[0] 
        return float(bf_val)
    except Exception as e:
        warnings.warn(f"R call failed, falling back to pingouin: {e}")
        from scipy import stats
        t_stat, _ = stats.ttest_ind(data_win, data_los)
        return pg.bayesfactor_ttest(t_stat, nx=len(data_win), ny=len(data_los), r=rscale)


def plot_decoding(max_pairs=None):
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs_run = len(pairs_to_run)
    
    all_decoding = np.full((num_pairs_run * 2, num_tests, num_time_bins), np.nan)
    searchlight_all = np.full((num_pairs_run * 2, num_tests, 64, num_time_bins), np.nan)
    winners_idx, losers_idx = [], []
    
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
            file_path = os.path.join(path_to_data, 'derivatives', f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
            if not os.path.exists(file_path): continue
            with open(file_path, 'rb') as f: res = pickle.load(f)
            for t_idx in range(num_tests):
                all_decoding[idx, t_idx, :] = res['decoding'][t_idx]
                searchlight_all[idx, t_idx, :, :] = res['searchlight'][t_idx]

    montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(ch_names=montage.ch_names[:64], sfreq=256, ch_types='eeg')
    info.set_montage(montage)

    # FIGURE 2
    fig2 = plt.figure(figsize=(15, 12), layout='constrained')
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.2)
    titles = ['A) Own response', "B) Opponent's response", 'C) Own previous response', "D) Opponent's previous response"]
    
    # Using 1-indexed ticks locally but 0.125 increments matching the bins
    x_axis_time = np.linspace(0.125, 4.875, 20) 
    
    for t in range(num_tests):
        row, col = t // 2, t % 2
        inner_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs2[row, col],
                                                     height_ratios=[3, 0.5, 0.8], hspace=0.1)
        ax_main = fig2.add_subplot(inner_gs[0])
        ax_bf = fig2.add_subplot(inner_gs[1], sharex=ax_main)
        
        data_t = all_decoding[:, t, :]
        data_mean = np.nanmean(data_t * 100, axis=0)
        ci = np.nanstd(data_t * 100, axis=0) / np.sqrt(num_pairs_run * 2) * 1.96
        
        ax_main.axhline(33.33, color='k', linestyle='--', zorder=0)
        
        # Segment plotting to match paper's distinct lines
        colors = ['#EDB120', '#D95319', '#7E2F8E']
        for idx, (start, end) in enumerate([(0, 8), (8, 16), (16, 20)]):
            ax_main.plot(x_axis_time[start:end], data_mean[start:end], color=colors[idx], linewidth=2)
            ax_main.fill_between(x_axis_time[start:end], data_mean[start:end] - ci[start:end], data_mean[start:end] + ci[start:end], color=colors[idx], alpha=0.2)
            ax_main.scatter(x_axis_time[start:end], data_mean[start:end], color=colors[idx], s=40)
        
        ax_main.axvspan(0, 2, color='#EDB120', alpha=0.1)
        ax_main.axvspan(2, 4, color='#D95319', alpha=0.1)
        ax_main.axvspan(4, 5, color='#7E2F8E', alpha=0.1)
        
        ax_main.text(1.0, 39.7, 'Decision', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(3.0, 39.7, 'Response', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(4.5, 39.7, 'Feedback', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
        
        ax_main.set_ylim(31, 40)
        ax_main.set_ylabel('Accuracy (%)')
        ax_main.set_title(titles[t], loc='left', fontweight='bold')
        ax_main.set_xlim(0, 5)
        ax_main.set_xticks(np.arange(0, 5.1, 1))
        ax_main.set_xticklabels([])
        
        bfs = np.ones(num_time_bins)
        for w in range(num_time_bins):
            slice_w = data_t[:, w][~np.isnan(data_t[:, w])]
            if len(slice_w) > 2:
                bfs[w] = calc_bayes_factor(slice_w)
                
        ax_bf.scatter(x_axis_time, [0]*20, c=np.log10(bfs), cmap='coolwarm', vmin=-8, vmax=8, s=80)
        ax_bf.set_yticks([])
        ax_bf.set_ylabel('Log(BF10)')
        
        topo_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=inner_gs[2], wspace=0.05)
        sl_mean = np.nanmean(searchlight_all[:, t, :, :], axis=0)
        timebins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)] 
        
        for idx, (t_start, t_end) in enumerate(timebins):
            ax_topo = fig2.add_subplot(topo_gs[0, idx])
            topo_data = np.nanmean(sl_mean[:, t_start:t_end], axis=1)
            mne.viz.plot_topomap(topo_data, info, axes=ax_topo, show=False, vlim=(0.33, 0.36), 
                                 cmap='hot', extrapolate='box', sphere=(0.0, 0.0, 0.0, 0.095))
            ax_topo.set_title(f'{idx+1}s', pad=0)

    plt.savefig(os.path.join(plot_dir, 'Figure2.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure 2 saved.")

    # FIGURE 3
    if len(winners_idx) >= 1:
        fig3 = plt.figure(figsize=(15, 12), layout='constrained')
        gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.2)
        
        for t in range(num_tests):
            row, col = t // 2, t % 2
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs3[row, col],
                                                         height_ratios=[3, 1], hspace=0.1)
            ax_main = fig3.add_subplot(inner_gs[0])
            ax_bf = fig3.add_subplot(inner_gs[1], sharex=ax_main)
            
            data_win = all_decoding[winners_idx, t, :] * 100
            data_los = all_decoding[losers_idx, t, :] * 100
            
            ax_main.axhline(33.33, color='k', linestyle='--', zorder=0)
            for d, c, lbl in zip([data_win, data_los], ['#0072BD', '#77AC30'], ['Winners', 'Losers']):
                mean_d = np.nanmean(d, axis=0)
                ci_d = np.nanstd(d, axis=0) / np.sqrt(len(d)) * 1.96
                for start, end in [(0, 8), (8, 16), (16, 20)]:
                    line_lbl = lbl if start == 0 else None
                    ax_main.plot(x_axis_time[start:end], mean_d[start:end], color=c, linewidth=2, label=line_lbl)
                    ax_main.fill_between(x_axis_time[start:end], mean_d[start:end] - ci_d[start:end], mean_d[start:end] + ci_d[start:end], color=c, alpha=0.15)
                    ax_main.scatter(x_axis_time[start:end], mean_d[start:end], color=c, s=30)
                
            ax_main.axvspan(0, 2, color='#EDB120', alpha=0.1)
            ax_main.axvspan(2, 4, color='#D95319', alpha=0.1)
            ax_main.axvspan(4, 5, color='#7E2F8E', alpha=0.1)
            
            ax_main.text(1.0, 39.7, 'Decision', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
            ax_main.text(3.0, 39.7, 'Response', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
            ax_main.text(4.5, 39.7, 'Feedback', ha='center', va='top', fontsize=11, fontweight='bold', color='black', zorder=10)
            
            ax_main.set_ylim(31, 40)
            ax_main.set_ylabel('Accuracy (%)')
            ax_main.set_title(titles[t], loc='left', fontweight='bold')
            ax_main.legend(loc='lower right')
            ax_main.set_xlim(0, 5)
            
            bfs_win, bfs_los, bfs_diff = np.ones(num_time_bins), np.ones(num_time_bins), np.ones(num_time_bins)
            for w in range(num_time_bins):
                sw = data_win[:, w][~np.isnan(data_win[:, w])] / 100
                sl = data_los[:, w][~np.isnan(data_los[:, w])] / 100
                
                if len(sw) > 2: bfs_win[w] = calc_bayes_factor(sw)
                if len(sl) > 2: bfs_los[w] = calc_bayes_factor(sl)
                if len(sw) > 2 and len(sl) > 2: bfs_diff[w] = calc_bayes_factor_ind(sw, sl)

            ax_bf.scatter(x_axis_time, [2]*20, c=np.log10(bfs_win), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            ax_bf.scatter(x_axis_time, [1]*20, c=np.log10(bfs_los), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            ax_bf.scatter(x_axis_time, [0]*20, c=np.log10(bfs_diff), cmap='coolwarm', vmin=-6, vmax=6, s=80)
            
            ax_bf.set_yticks([0, 1, 2])
            ax_bf.set_yticklabels(['Diff', 'Losers', 'Winners'])
            ax_bf.set_ylim(-0.5, 2.5)

        plt.savefig(os.path.join(plot_dir, 'Figure3.png'), dpi=300, bbox_inches='tight', facecolor='white')
        print("Figure 3 saved.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None)
    args = parser.parse_args()
    plot_decoding(max_pairs=args.test_pairs)