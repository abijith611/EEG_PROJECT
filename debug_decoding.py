# debug_decoding.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from typing import Optional, Tuple, List, Dict, Any
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS, NUM_TESTS, NUM_TIME_BINS, PLOT_DIR, get_logger, setup_root_logger

if __name__ == '__main__':
    setup_root_logger(log_to_file=False)

logger = get_logger(__name__)

conditions = ['Own response', "Opponent's response",
              'Own previous response', "Opponent's previous response"]


def load_data_for_classifier(clf_suffix: str) -> Tuple[Optional[np.ndarray],
                                                        Optional[List[str]],
                                                        Optional[List[int]],
                                                        Optional[List[int]]]:
    pattern = os.path.join(DERIV_DIR, f'pair-*_player-*_task-RPS_decoding_{clf_suffix}.pkl')
    files = glob.glob(pattern)
    if not files:
        return None, None, None, None

    all_decoding = []
    subject_ids = []
    winners_idx = []
    losers_idx = []

    for fpath in sorted(files):
        base = os.path.basename(fpath)
        parts = base.split('_')
        pair = int(parts[0].split('-')[1])
        player = int(parts[1].split('-')[1])

        events_file = os.path.join(PATH_TO_DATA, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
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

    all_decoding = np.array(all_decoding)
    return all_decoding, subject_ids, winners_idx, losers_idx


def get_summary_stats() -> Dict[str, Any]:
    """
    Compute overall mean accuracies per condition for each classifier,
    and winner/loser differences.

    Returns:
        dict: classifier -> {
            'mean_acc': (4,20) array of mean accuracies (percent),
            'sem_acc': (4,20) array of standard errors,
            'mean_win': (4,20) array for winners (or None),
            'mean_los': (4,20) array for losers (or None),
            'diff': (4,20) array of (winners - losers) or None
        }
    """
    pattern = os.path.join(DERIV_DIR, 'pair-*_player-*_task-RPS_decoding_*.pkl')
    files = glob.glob(pattern)
    clf_suffixes = set()
    for f in files:
        base = os.path.basename(f)
        suffix = base.split('decoding_')[1].replace('.pkl', '')
        clf_suffixes.add(suffix)

    results = {}
    for clf in sorted(clf_suffixes):
        data = load_data_for_classifier(clf)
        if data[0] is None:
            continue
        all_decoding, _, winners_idx, losers_idx = data
        all_decoding_pct = all_decoding * 100
        n_subj = all_decoding_pct.shape[0]

        mean_acc = np.nanmean(all_decoding_pct, axis=0)
        std_acc = np.nanstd(all_decoding_pct, axis=0)
        sem_acc = std_acc / np.sqrt(n_subj)

        mean_win = np.nanmean(all_decoding_pct[winners_idx], axis=0) if winners_idx else None
        mean_los = np.nanmean(all_decoding_pct[losers_idx], axis=0) if losers_idx else None
        diff = (mean_win - mean_los) if (mean_win is not None and mean_los is not None) else None

        results[clf] = {
            'mean_acc': mean_acc,
            'sem_acc': sem_acc,
            'mean_win': mean_win,
            'mean_los': mean_los,
            'diff': diff
        }
    return results


def run_debug() -> None:
    stats = get_summary_stats()
    if not stats:
        logger.error("No decoding files found.")
        return

    logger.info("=" * 75)
    logger.info(" DEBUG DECODING STATISTICS FOR ALL CLASSIFIERS")
    logger.info("=" * 75)

    for clf, s in stats.items():
        logger.info(f"\n--- Classifier: {clf.upper()} ---")
        logger.info(f"  Subjects: {s['mean_acc'].shape[0] if hasattr(s['mean_acc'], 'shape') else 'N/A'}")
        nan_count = np.isnan(s['mean_acc']).sum()
        logger.info(f"  NaNs in mean: {nan_count}")

        chance = 33.33
        logger.info("\n  OVERALL ACCURACY (Average across all 5 seconds)")
        logger.info(f"  Chance level = {chance}%")

        for t in range(NUM_TESTS):
            mn_all = s['mean_acc'][t].mean()
            logger.info(f"\n  {conditions[t]}:")
            logger.info(f"    Overall: {mn_all:.2f}%")
            if s['diff'] is not None:
                mn_win = s['mean_win'][t].mean()
                mn_los = s['mean_los'][t].mean()
                diff = mn_win - mn_los
                logger.info(f"    Winners: {mn_win:.2f}%")
                logger.info(f"    Losers:  {mn_los:.2f}%")
                logger.info(f"    -> Diff: {diff:+.2f}%")

        # Plot and save
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        x = np.linspace(0.125, 4.875, NUM_TIME_BINS)
        for t in range(NUM_TESTS):
            ax = axes.flat[t]
            ax.axhline(chance, color='k', ls='--')
            ax.plot(x, s['mean_acc'][t], 'o-', color='blue', label='Overall')
            ax.fill_between(x, s['mean_acc'][t] - s['sem_acc'][t],
                            s['mean_acc'][t] + s['sem_acc'][t],
                            color='blue', alpha=0.2)
            if s['diff'] is not None:
                ax.plot(x, s['mean_win'][t], '^-', color='#0072BD', alpha=0.8, label='Winners')
                ax.plot(x, s['mean_los'][t], 's-', color='#77AC30', alpha=0.8, label='Losers')
                ax.legend(loc='lower right', fontsize=8)
            ax.set_title(conditions[t])
            ax.set_ylim(30, 40)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Accuracy (%)')
        plt.tight_layout()
        out_plot = os.path.join(PLOT_DIR, f'debug_decoding_{clf}.png')
        plt.savefig(out_plot)
        logger.info(f"\n  Debug plot saved to: {out_plot}")
        plt.close(fig)


if __name__ == '__main__':
    run_debug()