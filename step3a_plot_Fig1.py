# step3a_plot_Fig1.py
"""
Plot the behavioural responses
Matches Figure 1 layout using Matplotlib.
Implements custom Raincloud plots (half-violin, thick boxplot, left-scatter)
to exactly match the paper's visual style.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
from typing import List, Optional
from config import PATH_TO_DATA, PLOT_DIR, PAIR_IDS, DERIV_DIR, get_logger, setup_root_logger

logger = get_logger(__name__)
os.makedirs(PLOT_DIR, exist_ok=True)


def draw_raincloud(ax: plt.Axes, data_list: List[np.ndarray], colors: List[str], x_labels: List[str]) -> None:
    """
    Custom function to draw raincloud plots with extended KDE tails.
    If a dataset has fewer than 2 points, the half‑violin is omitted.

    Args:
        ax: Matplotlib axes to draw on.
        data_list: List of 1D arrays for each condition.
        colors: List of colors for each condition.
        x_labels: Tick labels for x‑axis.
    """
    x_pos = np.arange(len(data_list))

    # 1. Half-violins using custom KDE for extended tails
    for i, data in enumerate(data_list):
        y = data[~np.isnan(data)]
        if len(y) >= 2:
            kde = gaussian_kde(y)
            y_min, y_max = y.min(), y.max()
            y_range = y_max - y_min
            y_eval = np.linspace(y_min - 0.3 * y_range, y_max + 0.3 * y_range, 200)
            x_eval = kde(y_eval)
            x_eval = (x_eval / x_eval.max()) * 0.35
            ax.fill_betweenx(y_eval, x_pos[i], x_pos[i] + x_eval,
                              facecolor=colors[i], edgecolor='black', alpha=0.6)
        else:
            logger.warning(f"Not enough data points (n={len(y)}) to draw half-violin for condition {i}")

    # 2. Boxplots
    bp = ax.boxplot(data_list, positions=x_pos, widths=0.12, patch_artist=True,
                    showfliers=False, medianprops={'color': 'white', 'linewidth': 2},
                    boxprops={'facecolor': 'black', 'edgecolor': 'black'},
                    whiskerprops={'color': 'black', 'linewidth': 1.5},
                    capprops={'color': 'black', 'linewidth': 1.5})

    # 3. Scatter points (rain)
    for i, data in enumerate(data_list):
        y = data[~np.isnan(data)]
        if len(y) > 0:
            x = np.random.normal(loc=x_pos[i] - 0.2, scale=0.03, size=len(y))
            ax.scatter(x, y, color='black', alpha=0.4, s=15, edgecolors='none', zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)


def plot_behavior(max_pairs: Optional[int] = None) -> None:
    """
    Generate Figure 1: behavioural results.

    Args:
        max_pairs: If given, use only the first `max_pairs` pairs.
    """
    pairs_to_run = PAIR_IDS[:max_pairs] if max_pairs is not None else PAIR_IDS
    num_pairs_run = len(pairs_to_run)
    logger.info(f"Plotting behaviour for {num_pairs_run} pairs.")

    outcome_summary = np.zeros((num_pairs_run, 3))
    ranked_resp = np.zeros((3, num_pairs_run * 2))
    all_played_rank = np.zeros((3, num_pairs_run * 2))
    prop_change = np.zeros((3, num_pairs_run * 2))

    for p_idx, pair in enumerate(pairs_to_run):
        events_file = os.path.join(PATH_TO_DATA, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if not os.path.exists(events_file):
            logger.warning(f"Events file missing for pair {pair}, skipping in behaviour plots.")
            continue
        events = pd.read_csv(events_file, sep='\t')

        # Winner index logic
        w1, w2 = sum(events['outcome'] == 2), sum(events['outcome'] == 3)
        winner_idx = 0 if w1 > w2 else 1

        ev_r = events[(events['player1_resp'] > 0) & (events['player2_resp'] > 0)]
        tot = len(ev_r)

        if tot > 0:
            outcome_summary[p_idx, 0] = sum(ev_r['outcome'] == 1) / tot * 100
            outcome_summary[p_idx, 1] = sum(ev_r['outcome'] == (2 if winner_idx == 0 else 3)) / tot * 100
            outcome_summary[p_idx, 2] = sum(ev_r['outcome'] == (3 if winner_idx == 0 else 2)) / tot * 100

            played = ev_r[['player1_resp', 'player2_resp']].values
            for ppt in [0, 1]:
                counts = np.bincount(played[:, ppt], minlength=4)[1:] / tot * 100
                ranked_resp[:, p_idx * 2 + ppt] = np.sort(counts)[::-1]
                all_played_rank[:, p_idx * 2 + ppt] = np.argsort(counts)[::-1] + 1

        # Game-to-game response change
        for ppt in [1, 2]:
            resp = events[f'player{ppt}_resp'].values
            outcome = events['outcome'].values

            if ppt == 2:
                outcome_aligned = outcome.copy()
                outcome_aligned[outcome == 2] = 3
                outcome_aligned[outcome == 3] = 2
            else:
                outcome_aligned = outcome

            stay_win, stay_loss, stay_draw = [], [], []

            for i in range(1, len(resp)):
                if resp[i] > 0 and resp[i-1] > 0:
                    changed = (resp[i] != resp[i-1])
                    prev_out = outcome_aligned[i-1]

                    if prev_out == 2:
                        stay_win.append(changed)
                    elif prev_out == 3:
                        stay_loss.append(changed)
                    elif prev_out == 1:
                        stay_draw.append(changed)

            ppt_idx = p_idx * 2 + (ppt - 1)
            if stay_win:
                prop_change[0, ppt_idx] = np.mean(stay_win) * 100
            if stay_loss:
                prop_change[1, ppt_idx] = np.mean(stay_loss) * 100
            if stay_draw:
                prop_change[2, ppt_idx] = np.mean(stay_draw) * 100

    # Load Markov Chain data
    mc_file = os.path.join(DERIV_DIR, 'markov_chain_pred.npy')
    if os.path.exists(mc_file):
        mc_data = np.load(mc_file, allow_pickle=True).item()
        pred_acc = mc_data['Mean_Accuracy'][:, :, 5:] * 100
        pred_acc = pred_acc.reshape(-1, 96)
        logger.info("Markov chain data loaded.")
    else:
        pred_acc = np.zeros((num_pairs_run * 2, 96))
        logger.warning("Markov chain file not found; using zeros.")

    # Figure setup
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout='constrained')

    # Plot 1: Outcomes Violin
    ax = axes[0, 0]
    ax.set_title("Game outcome", loc='left', fontweight='bold')
    data_outcomes = [outcome_summary[:, 1], outcome_summary[:, 2], outcome_summary[:, 0]]
    draw_raincloud(ax, data_outcomes, ['#D95319', '#EDB120', '#7E2F8E'],
                   ['Winner\nwins', 'Loser\nwins', 'Draw'])
    ax.axhline(33.33, color='k', linestyle='--', zorder=0)
    ax.set_ylabel('Percentage')
    ax.set_ylim(18, 48)

    # Plot 2: Ranked Resp Raincloud + Pie Insets
    ax = axes[0, 1]
    ax.set_title("Response Played", loc='left', fontweight='bold')
    data_ranked = [ranked_resp[0, :], ranked_resp[1, :], ranked_resp[2, :]]
    draw_raincloud(ax, data_ranked, ['#B30000', '#E64D00', '#FFB300'],
                   ['Most\nplayed', 'Mid\nplayed', 'Least\nplayed'])
    ax.axhline(33.33, color='k', linestyle='--', zorder=0)
    ax.set_ylim(18, 48)

    rps_colors = ['#4DBEEE', '#77AC30', '#EDB120']
    for i in range(3):
        ax_inset = ax.inset_axes([0.15 + i*0.31, 0.85, 0.15, 0.15])
        rank_data = all_played_rank[i, :]
        counts = [np.sum(rank_data == 1), np.sum(rank_data == 2), np.sum(rank_data == 3)]
        ax_inset.pie(counts, labels=['R', 'P', 'S'], colors=rps_colors, textprops={'fontsize': 8})
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(rps_colors, ['Rock', 'Paper', 'Scissors'])]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.75),
              ncol=1, frameon=False, fontsize=10)

    # Plot 3: Response Change (Switch rate)
    ax = axes[1, 0]
    ax.set_title("Game-to-game response change", loc='left', fontweight='bold')
    data_change = [prop_change[0, :], prop_change[1, :], prop_change[2, :]]
    draw_raincloud(ax, data_change, ['#4DBEEE', '#77AC30', '#7E2F8E'],
                   ['After\nwin', 'After\nloss', 'After\ndraw'])
    ax.axhline(66.67, color='k', linestyle='--', zorder=0)
    ax.set_ylabel('Percentage')
    ax.set_ylim(20, 103)

    # Plot 4: Predictability (Markov Chain)
    ax = axes[1, 1]
    ax.set_title("Markov chain response predictability", loc='left', fontweight='bold')
    x_axis = np.arange(5, 101)

    valid_pred_acc = pred_acc[~np.isnan(pred_acc).all(axis=1)]
    if len(valid_pred_acc) > 0:
        mean_acc = np.nanmean(valid_pred_acc, axis=0)
        ci = stats.t.ppf(0.975, valid_pred_acc.shape[0]-1) * np.nanstd(valid_pred_acc, axis=0) / np.sqrt(valid_pred_acc.shape[0])

        for i in range(valid_pred_acc.shape[0]):
            ax.plot(x_axis, valid_pred_acc[i, :], color='gray', alpha=0.15)
        ax.plot(x_axis, mean_acc, color='#0072BD', linewidth=2)
        ax.fill_between(x_axis, mean_acc - ci, mean_acc + ci, color='#0072BD', alpha=0.2)

    ax.axhline(33.33, color='k', linestyle='--', zorder=0)
    ax.set_xlabel('N previous games')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(25, 65)

    out_path = os.path.join(PLOT_DIR, 'Figure1.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Figure 1 saved to {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    setup_root_logger(log_to_file=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    plot_behavior(max_pairs=args.test_pairs)