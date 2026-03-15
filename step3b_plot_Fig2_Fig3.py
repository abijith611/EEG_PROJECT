# step3b_plot_Fig2_Fig3.py
"""
Plot Decoding Results and Compute Bayes Factors
Replicates the grid layout, time-bin topographies, and directional Bayes Factors.
Includes perfectly aligned Topoplots and dedicated Colorbar axes.
Now supports a single classifier argument; if omitted, plots all available classifiers.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import mne
import warnings
import glob
import logging
from typing import Optional, List, Tuple
from config import (PATH_TO_DATA, PLOT_DIR, PAIR_IDS, DERIV_DIR,
                    SEARCHLIGHT_CLASSIFIERS, NUM_TESTS, NUM_TIME_BINS, NUM_CHAN,
                    get_logger, setup_root_logger)

# Suppress verbose rpy2 logs and R console outputs (like "t approximation invoked")
logging.getLogger('rpy2').setLevel(logging.ERROR)
logging.getLogger('rpy2.rinterface_lib.callbacks').setLevel(logging.ERROR)

logger = get_logger(__name__)

R_AVAILABLE = False
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    conv = numpy2ri.converter
    robjects.r('version')
    bf_pkg = importr('BayesFactor')
    R_AVAILABLE = True
    logger.info("R and BayesFactor package found – using exact Bayes factors.")
except Exception as e:
    R_AVAILABLE = False
    logger.warning(f"R/BayesFactor not available ({e}). Using pingouin approximation.")
    import pingouin as pg

os.makedirs(PLOT_DIR, exist_ok=True)

# Custom hot colormap to match MATLAB (removes the glaring white tip)
cmap_hot = plt.get_cmap('hot')
custom_hot = mcolors.ListedColormap(cmap_hot(np.linspace(0, 0.9, 256)))


def calc_bayes_factor(data: np.ndarray, mu: float = 1/3,
                      rscale: str = "medium", null_interval: str = "c(0.5, Inf)") -> float:
    """
    Calculate directional Bayes factor for one-sample test against mu.

    Args:
        data: Array of values (proportions, not percentages).
        mu: Chance level (default 1/3).
        rscale: Prior scale for BayesFactor.
        null_interval: R expression for the null interval.

    Returns:
        Bayes factor (BF10).
    """
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
        logger.warning(f"R BayesFactor failed: {e}. Falling back to pingouin.")
        from scipy import stats
        t_stat, _ = stats.ttest_1samp(data, mu)
        return pg.bayesfactor_ttest(t_stat, nx=len(data), r=rscale)


def calc_bayes_factor_ind(data_win: np.ndarray, data_los: np.ndarray,
                          rscale: str = "medium", null_interval: str = "c(-0.5, 0.5)") -> float:
    """
    Calculate Bayes factor for independent samples (winners vs losers).

    Args:
        data_win: Values for winners.
        data_los: Values for losers.
        rscale: Prior scale.
        null_interval: R expression for the null interval.

    Returns:
        Bayes factor (BF10 for alternative hypothesis).
    """
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
        logger.warning(f"R BayesFactor independent failed: {e}. Falling back to pingouin.")
        from scipy import stats
        t_stat, _ = stats.ttest_ind(data_win, data_los)
        return pg.bayesfactor_ttest(t_stat, nx=len(data_win), ny=len(data_los), r=rscale)


def plot_decoding(max_pairs: Optional[int] = None, classifier: str = 'svm') -> None:
    """
    Generate Figures 2 and 3 for a given classifier.

    Args:
        max_pairs: If given, use only the first `max_pairs` pairs.
        classifier: Classifier name (e.g., 'svm').
    """
    pairs_to_run = PAIR_IDS[:max_pairs] if max_pairs is not None else PAIR_IDS
    num_pairs_run = len(pairs_to_run)
    logger.info(f"Plotting decoding for classifier {classifier} using {num_pairs_run} pairs.")

    all_decoding = np.full((num_pairs_run * 2, NUM_TESTS, NUM_TIME_BINS), np.nan)
    searchlight_all = np.full((num_pairs_run * 2, NUM_TESTS, NUM_CHAN, NUM_TIME_BINS), np.nan)
    winners_idx, losers_idx = [], []

    for p_idx, pair in enumerate(pairs_to_run):
        events_file = os.path.join(PATH_TO_DATA, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if os.path.exists(events_file):
            events = pd.read_csv(events_file, sep='\t')
            w1, w2 = sum(events['outcome'] == 2), sum(events['outcome'] == 3)
            p1_idx, p2_idx = p_idx * 2, p_idx * 2 + 1
            if w1 > w2:
                winners_idx.append(p1_idx)
                losers_idx.append(p2_idx)
            else:
                winners_idx.append(p2_idx)
                losers_idx.append(p1_idx)
        else:
            logger.warning(f"Events file missing for pair {pair}, cannot assign winner/loser.")

        for ppt in [1, 2]:
            idx = p_idx * 2 + (ppt - 1)
            file_path = os.path.join(DERIV_DIR, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{classifier}.pkl')
            if not os.path.exists(file_path):
                logger.debug(f"File not found: {file_path}")
                continue
            with open(file_path, 'rb') as f:
                res = pickle.load(f)
            for t_idx in range(NUM_TESTS):
                all_decoding[idx, t_idx, :] = res['decoding'][t_idx]
                searchlight_all[idx, t_idx, :, :] = res['searchlight'][t_idx]

    # Use first available file to get ch_names
    first_file = None
    for p_idx, pair in enumerate(pairs_to_run):
        for ppt in [1, 2]:
            test_path = os.path.join(DERIV_DIR, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{classifier}.pkl')
            if os.path.exists(test_path):
                first_file = test_path
                break
        if first_file:
            break
    if first_file is None:
        logger.error(f"No decoding files found for classifier {classifier}")
        return

    with open(first_file, 'rb') as f:
        res_example = pickle.load(f)
    ch_names_ordered = res_example['ch_names']

    info = mne.create_info(ch_names=ch_names_ordered, sfreq=256, ch_types='eeg')
    montage = mne.channels.make_standard_montage('biosemi64')
    info.set_montage(montage)

    has_searchlight = classifier in SEARCHLIGHT_CLASSIFIERS

    # ---------------------------------------------------------
    # FIGURE 2
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(16, 13), layout='constrained')
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.15)
    titles = ['A) Own response', "B) Opponent's response",
              'C) Own previous response', "D) Opponent's previous response"]

    x_axis_bins = np.arange(1, NUM_TIME_BINS + 1)

    for t in range(NUM_TESTS):
        row, col = t // 2, t % 2

        if has_searchlight:
            inner_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs2[row, col],
                                                         height_ratios=[4, 0.5, 1.2],
                                                         width_ratios=[21, 1],
                                                         hspace=0.2, wspace=0.05)
            ax_topo_cb = fig2.add_subplot(inner_gs[2, 1])
        else:
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs2[row, col],
                                                         height_ratios=[4, 0.5],
                                                         width_ratios=[21, 1],
                                                         hspace=0.2, wspace=0.05)

        ax_main = fig2.add_subplot(inner_gs[0, 0])
        ax_bf = fig2.add_subplot(inner_gs[1, 0], sharex=ax_main)
        ax_bf_cb = fig2.add_subplot(inner_gs[1, 1])

        data_t = all_decoding[:, t, :]
        # Sanity check: if all NaN, skip plotting
        if np.all(np.isnan(data_t)):
            logger.warning(f"All data NaN for condition {t}, skipping subplot.")
            ax_main.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_main.transAxes)
            continue

        data_mean = np.nanmean(data_t * 100, axis=0)
        ci = np.nanstd(data_t * 100, axis=0) / np.sqrt(num_pairs_run * 2) * 1.96

        ax_main.axhline(33.33, color='k', linestyle='--', zorder=0)

        colors = ['#EDB120', '#D95319', '#7E2F8E']
        for idx, (start, end) in enumerate([(0, 8), (8, 16), (16, 20)]):
            ax_main.plot(x_axis_bins[start:end], data_mean[start:end],
                         color=colors[idx], linewidth=2)
            ax_main.fill_between(x_axis_bins[start:end],
                                 data_mean[start:end] - ci[start:end],
                                 data_mean[start:end] + ci[start:end],
                                 color=colors[idx], alpha=0.2)
            ax_main.scatter(x_axis_bins[start:end], data_mean[start:end],
                            color=colors[idx], s=45)

        ax_main.axvspan(0.75, 8.25, color='#EDB120', alpha=0.1)
        ax_main.axvspan(8.75, 16.25, color='#D95319', alpha=0.1)
        ax_main.axvspan(16.75, 20.25, color='#7E2F8E', alpha=0.1)

        ax_main.text(4.5, 39.5, 'Decision', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(12.5, 39.5, 'Response', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(18.5, 39.5, 'Feedback', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)

        ax_main.set_ylim(31, 40)
        ax_main.set_ylabel('Decoding accuracy (%)', fontsize=11)
        ax_main.set_title(titles[t], loc='left', fontweight='bold', fontsize=14)
        ax_main.set_xlim(0, 21)
        ax_main.set_xticks([0.5, 4.5, 8.5, 12.5, 16.5, 20.5])
        ax_main.set_xticklabels([])

        bfs = np.ones(NUM_TIME_BINS)
        for w in range(NUM_TIME_BINS):
            slice_w = data_t[:, w][~np.isnan(data_t[:, w])]
            if len(slice_w) > 2:
                bfs[w] = calc_bayes_factor(slice_w / 100)   # convert to proportion

        ax_bf.scatter(x_axis_bins, [0] * 20, color='lightgray', s=90, alpha=0.8)
        sc = ax_bf.scatter(x_axis_bins, [0] * 20, c=np.log10(bfs),
                           cmap='RdBu_r', vmin=-6, vmax=6, s=90)

        cb_bf = fig2.colorbar(sc, cax=ax_bf_cb)
        cb_bf.set_ticks([-6, 0, 6])
        cb_bf.set_ticklabels(['10⁻⁶', '1', '10⁶'])
        cb_bf.ax.tick_params(labelsize=10)

        if col == 0:
            ax_bf_cb.set_visible(False)
        else:
            cb_bf.set_label('BF (log scale)', fontsize=11, fontweight='bold')

        ax_bf.set_yticks([])
        ax_bf.set_ylabel('BF10', fontsize=11)
        ax_bf.set_xticks([0.5, 4.5, 8.5, 12.5, 16.5, 20.5])
        ax_bf.set_xticklabels(['0', '1', '2', '3', '4', '5'])
        ax_bf.set_xlabel('Time (s)', fontsize=12)

        # Topoplots – only if searchlight data exists
        if has_searchlight:
            topo_gs = gridspec.GridSpecFromSubplotSpec(1, 21, subplot_spec=inner_gs[2, 0], wspace=0.0)
            sl_mean = np.nanmean(searchlight_all[:, t, :, :], axis=0)

            col_starts = [1, 5, 9, 13, 17]
            col_ends = [4, 8, 12, 16, 20]
            timebins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]

            for idx, (t_start, t_end) in enumerate(timebins):
                ax_topo = fig2.add_subplot(topo_gs[0, col_starts[idx]:col_ends[idx]])
                topo_data = np.nanmean(sl_mean[:, t_start:t_end], axis=1)
                mne.viz.plot_topomap(topo_data, info, axes=ax_topo, show=False,
                                     vlim=(0.3333, 0.36), cmap=custom_hot,
                                     contours=0, sphere=None)
                ax_topo.set_title(f'{idx+1}s', pad=2, fontsize=12, fontweight='bold')

            # Topo colorbar
            sm_topo = plt.cm.ScalarMappable(cmap=custom_hot, norm=plt.Normalize(vmin=0.3333, vmax=0.36))
            sm_topo.set_array([])
            cb_topo = fig2.colorbar(sm_topo, cax=ax_topo_cb)
            cb_topo.set_ticks([0.3333, 0.36])
            cb_topo.set_ticklabels(['33.3%', '36.0%'])
            cb_topo.ax.tick_params(labelsize=10)

            if col == 0:
                ax_topo_cb.set_visible(False)
            else:
                cb_topo.set_label('Accuracy (%)', fontsize=11, fontweight='bold', labelpad=10)
        else:
            if 'ax_topo_cb' in locals():
                ax_topo_cb.set_visible(False)

    out_path = os.path.join(PLOT_DIR, f'Figure2_{classifier}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    logger.info(f"Figure2_{classifier}.png saved.")
    plt.close(fig2)

    # ---------------------------------------------------------
    # FIGURE 3 (winners vs losers)
    # ---------------------------------------------------------
    if len(winners_idx) < 1 or len(losers_idx) < 1:
        logger.warning("Insufficient winners/losers groups. Skipping Figure 3.")
        return

    fig3 = plt.figure(figsize=(16, 13), layout='constrained')
    gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.15)

    color_win = plt.get_cmap('winter')(0.2)
    color_los = plt.get_cmap('winter')(0.7)

    for t in range(NUM_TESTS):
        row, col = t // 2, t % 2

        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs3[row, col],
                                                     height_ratios=[3, 1.2],
                                                     width_ratios=[21, 1],
                                                     hspace=0.15, wspace=0.05)

        ax_main = fig3.add_subplot(inner_gs[0, 0])
        ax_bf = fig3.add_subplot(inner_gs[1, 0], sharex=ax_main)
        ax_bf_cb = fig3.add_subplot(inner_gs[1, 1])

        data_win = all_decoding[winners_idx, t, :] * 100
        data_los = all_decoding[losers_idx, t, :] * 100

        # Sanity check
        if np.all(np.isnan(data_win)) or np.all(np.isnan(data_los)):
            logger.warning(f"All NaN for winners/losers in condition {t}, skipping.")
            ax_main.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_main.transAxes)
            continue

        ax_main.axhline(33.33, color='k', linestyle='--', zorder=0)

        for d, c, lbl in zip([data_win, data_los], [color_win, color_los], ['Winners', 'Losers']):
            mean_d = np.nanmean(d, axis=0)
            ci_d = np.nanstd(d, axis=0) / np.sqrt(len(d)) * 1.96
            for start, end in [(0, 8), (8, 16), (16, 20)]:
                line_lbl = lbl if start == 0 else None
                ax_main.plot(x_axis_bins[start:end], mean_d[start:end],
                             color=c, linewidth=2, label=line_lbl)
                ax_main.fill_between(x_axis_bins[start:end],
                                     mean_d[start:end] - ci_d[start:end],
                                     mean_d[start:end] + ci_d[start:end],
                                     color=c, alpha=0.15)
                ax_main.scatter(x_axis_bins[start:end], mean_d[start:end],
                                color=c, s=45)

        ax_main.axvspan(0.75, 8.25, color='#EDB120', alpha=0.1)
        ax_main.axvspan(8.75, 16.25, color='#D95319', alpha=0.1)
        ax_main.axvspan(16.75, 20.25, color='#7E2F8E', alpha=0.1)

        ax_main.text(4.5, 39.5, 'Decision', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(12.5, 39.5, 'Response', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)
        ax_main.text(18.5, 39.5, 'Feedback', ha='center', va='top',
                     fontsize=11, fontweight='bold', color='black', zorder=10)

        ax_main.set_ylim(31, 40)
        ax_main.set_ylabel('Decoding accuracy (%)', fontsize=11)
        ax_main.set_title(titles[t], loc='left', fontweight='bold', fontsize=14)
        ax_main.legend(loc='lower right')
        ax_main.set_xlim(0, 21)
        ax_main.set_xticks([0.5, 4.5, 8.5, 12.5, 16.5, 20.5])
        ax_main.set_xticklabels([])

        bfs_win, bfs_los, bfs_diff = np.ones(NUM_TIME_BINS), np.ones(NUM_TIME_BINS), np.ones(NUM_TIME_BINS)
        for w in range(NUM_TIME_BINS):
            sw = data_win[:, w][~np.isnan(data_win[:, w])] / 100
            sl = data_los[:, w][~np.isnan(data_los[:, w])] / 100

            if len(sw) > 2:
                bfs_win[w] = calc_bayes_factor(sw)
            if len(sl) > 2:
                bfs_los[w] = calc_bayes_factor(sl)
            if len(sw) > 2 and len(sl) > 2:
                bfs_diff[w] = calc_bayes_factor_ind(sw, sl)

        # Background gray circles
        ax_bf.scatter(x_axis_bins, [2] * 20, color='lightgray', s=90, alpha=0.8)
        ax_bf.scatter(x_axis_bins, [1] * 20, color='lightgray', s=90, alpha=0.8)
        ax_bf.scatter(x_axis_bins, [0] * 20, color='lightgray', s=90, alpha=0.8)

        ax_bf.scatter(x_axis_bins, [2] * 20, c=np.log10(bfs_win),
                      cmap='RdBu_r', vmin=-6, vmax=6, s=90)
        ax_bf.scatter(x_axis_bins, [1] * 20, c=np.log10(bfs_los),
                      cmap='RdBu_r', vmin=-6, vmax=6, s=90)
        sc3 = ax_bf.scatter(x_axis_bins, [0] * 20, c=np.log10(bfs_diff),
                            cmap='RdBu_r', vmin=-6, vmax=6, s=90)

        cb3 = fig3.colorbar(sc3, cax=ax_bf_cb)
        cb3.set_ticks([-6, 0, 6])
        cb3.set_ticklabels(['10⁻⁶', '1', '10⁶'])
        cb3.ax.tick_params(labelsize=10)

        if col == 0:
            ax_bf_cb.set_visible(False)
        else:
            cb3.set_label('BF (log scale)', fontsize=11, fontweight='bold')

        ax_bf.set_yticks([0, 1, 2])
        ax_bf.set_yticklabels(['Diff', 'Losers', 'Winners'], fontsize=10)
        ax_bf.set_ylim(-1, 3)
        ax_bf.set_xticks([0.5, 4.5, 8.5, 12.5, 16.5, 20.5])
        ax_bf.set_xticklabels(['0', '1', '2', '3', '4', '5'])
        ax_bf.set_xlabel('Time (s)', fontsize=12)

    out_path = os.path.join(PLOT_DIR, f'Figure3_{classifier}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    logger.info(f"Figure3_{classifier}.png saved.")
    plt.close(fig3)


if __name__ == '__main__':
    import argparse
    import sys
    setup_root_logger(log_to_file=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None)
    parser.add_argument('--classifier', type=str, default=None,
                        choices=['svm', 'lda', 'logistic', 'ridge'],
                        help='Classifier to plot. If omitted, plots all available classifiers.')
    args = parser.parse_args()

    # Find all available classifiers from existing decoding files
    if args.classifier is None:
        pattern = os.path.join(PATH_TO_DATA, 'derivatives', 'pair-*_player-*_task-RPS_decoding_*.pkl')
        files = glob.glob(pattern)
        classifiers = set()
        for f in files:
            base = os.path.basename(f)
            suffix = base.split('decoding_')[1].replace('.pkl', '')
            classifiers.add(suffix)
        if not classifiers:
            logger.error("No decoding files found.")
            sys.exit(1)
        classifiers = sorted(classifiers)
        logger.info(f"Found classifiers: {classifiers}")
    else:
        classifiers = [args.classifier]

    for clf in classifiers:
        logger.info(f"\n--- Plotting for classifier: {clf} ---")
        plot_decoding(max_pairs=args.test_pairs, classifier=clf)