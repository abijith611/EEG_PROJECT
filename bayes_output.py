# bayes_output.py
import os
import pickle
import numpy as np
import pandas as pd
import glob
import pingouin as pg
import logging
from typing import List, Tuple, Optional, Dict, Any
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS, get_logger, setup_root_logger

if __name__ == '__main__':
    setup_root_logger(log_to_file=False)

logging.getLogger('rpy2').setLevel(logging.WARNING)
logger = get_logger(__name__)

R_AVAILABLE = False
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    conv = numpy2ri.converter
    bf_pkg = importr('BayesFactor')
    R_AVAILABLE = True
    logger.info("R and BayesFactor package found – using exact Bayes factors.")
except Exception as e:
    logger.warning(f"R/BayesFactor not available: {e}. Using pingouin approximation.")
    R_AVAILABLE = False


def format_bf(bf: float) -> str:
    """
    Format Bayes factor for display – matches paper style (full numbers, no scientific).
    - NaN → "N/A"
    - bf ≥ 1000 → integer with commas (e.g., "1,234,567")
    - 1 ≤ bf < 1000 → integer with commas if effectively integer, else 2 decimals
    - bf < 1 → up to 4 decimals (no commas, no scientific)
    """
    if np.isnan(bf):
        return "N/A"
    if bf >= 1000:
        # integer with commas
        return f"{bf:,.0f}"
    elif bf >= 1:
        # If very close to an integer, show as integer
        if abs(bf - round(bf)) < 1e-6:
            return f"{round(bf):,}"
        else:
            # two decimals, but use comma for thousands if applicable
            # Since bf < 1000, no comma needed, but keep for consistency
            return f"{bf:,.2f}"
    else:  # bf < 1
        # Determine precision: up to 4 decimals, but avoid trailing zeros
        if bf >= 0.01:
            return f"{bf:.3f}".rstrip('0').rstrip('.')
        elif bf >= 0.0001:
            return f"{bf:.4f}".rstrip('0').rstrip('.')
        else:
            return f"{bf:.6f}".rstrip('0').rstrip('.')


def calc_bf_1samp(data: np.ndarray, mu: float = 100/3) -> float:
    if len(data) < 3:
        return 1.0
    if R_AVAILABLE:
        try:
            with conv.context():
                robjects.globalenv['data'] = data
                robjects.r(f'bf = BayesFactor::ttestBF(x=data, mu={mu}, rscale="medium", nullInterval=c(0, 0.5))')
                bf_val = robjects.r('as.vector(bf[1])')[0]
            return float(bf_val)
        except Exception as e:
            logger.warning(f"R BayesFactor failed: {e}. Falling back to pingouin.")
    from scipy import stats
    t_stat, _ = stats.ttest_1samp(data, mu)
    return pg.bayesfactor_ttest(t_stat, nx=len(data), r='medium')


def get_winner_loser_stats() -> Dict[str, Any]:
    """
    Compute peak accuracies and maximum Bayes factors for winners and losers
    across all classifiers, conditions, and phases.

    Returns a nested dictionary:
        classifier -> condition_name -> phase_name -> {
            'overall': (peak_acc, max_bf),
            'winners': (peak_acc, max_bf) or (nan, nan),
            'losers': (peak_acc, max_bf) or (nan, nan)
        }
    """
    conditions = ['Own response', "Opponent's response",
                  'Own previous response', "Opponent's previous response"]
    phases = {
        'Decision (0-2s)': (0, 8),
        'Response (2-4s)': (8, 16),
        'Feedback (4-5s)': (16, 20)
    }

    pattern = os.path.join(DERIV_DIR, 'pair-*_player-*_task-RPS_decoding_*.pkl')
    files = glob.glob(pattern)
    clf_suffixes = set()
    for f in files:
        base = os.path.basename(f)
        suffix = base.split('decoding_')[1].replace('.pkl', '')
        clf_suffixes.add(suffix)

    results = {}

    for clf in sorted(clf_suffixes):
        all_decoding = []
        winners_idx = []
        losers_idx = []

        for pair in PAIR_IDS:
            events_file = os.path.join(PATH_TO_DATA, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
            winner_ppt = None
            if os.path.exists(events_file):
                events = pd.read_csv(events_file, sep='\t')
                w1 = sum(events['outcome'] == 2)
                w2 = sum(events['outcome'] == 3)
                winner_ppt = 1 if w1 > w2 else 2

            for ppt in [1, 2]:
                fpath = os.path.join(DERIV_DIR, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{clf}.pkl')
                if not os.path.exists(fpath):
                    continue
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                current_idx = len(all_decoding)
                all_decoding.append(np.array(data['decoding']) * 100)
                if winner_ppt is not None:
                    if ppt == winner_ppt:
                        winners_idx.append(current_idx)
                    else:
                        losers_idx.append(current_idx)

        if not all_decoding:
            continue

        all_decoding = np.array(all_decoding)          # (n_subj,4,20)
        mean_acc_all = np.nanmean(all_decoding, axis=0)
        data_win = all_decoding[winners_idx] if winners_idx else np.array([])
        data_los = all_decoding[losers_idx] if losers_idx else np.array([])
        mean_win = np.nanmean(data_win, axis=0) if data_win.size else np.full((4,20), np.nan)
        mean_los = np.nanmean(data_los, axis=0) if data_los.size else np.full((4,20), np.nan)

        clf_res = {}
        for t_idx, cond in enumerate(conditions):
            cond_res = {}
            for phase_name, (start, end) in phases.items():
                # Overall
                phase_vals_all = mean_acc_all[t_idx, start:end]
                max_acc_all = np.max(phase_vals_all)
                bfs_all = []
                for bin_idx in range(start, end):
                    bin_data = all_decoding[:, t_idx, bin_idx]
                    bin_data = bin_data[~np.isnan(bin_data)]
                    bfs_all.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
                max_bf_all = np.max(bfs_all)

                # Winners
                if data_win.size:
                    phase_vals_win = mean_win[t_idx, start:end]
                    max_acc_win = np.max(phase_vals_win)
                    bfs_win = []
                    for bin_idx in range(start, end):
                        bin_data = data_win[:, t_idx, bin_idx]
                        bin_data = bin_data[~np.isnan(bin_data)]
                        bfs_win.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
                    max_bf_win = np.max(bfs_win)
                else:
                    max_acc_win, max_bf_win = np.nan, np.nan

                # Losers
                if data_los.size:
                    phase_vals_los = mean_los[t_idx, start:end]
                    max_acc_los = np.max(phase_vals_los)
                    bfs_los = []
                    for bin_idx in range(start, end):
                        bin_data = data_los[:, t_idx, bin_idx]
                        bin_data = bin_data[~np.isnan(bin_data)]
                        bfs_los.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
                    max_bf_los = np.max(bfs_los)
                else:
                    max_acc_los, max_bf_los = np.nan, np.nan

                cond_res[phase_name] = {
                    'overall': (max_acc_all, max_bf_all),
                    'winners': (max_acc_win, max_bf_win),
                    'losers': (max_acc_los, max_bf_los)
                }
            clf_res[cond] = cond_res
        results[clf] = clf_res

    return results


def extract_winner_loser_stats() -> None:
    """Print formatted table (original behavior) with paper‑style Bayes numbers."""
    stats = get_winner_loser_stats()
    if not stats:
        logger.error("No data found.")
        return

    phases = ['Decision (0-2s)', 'Response (2-4s)', 'Feedback (4-5s)']
    for clf, clf_data in stats.items():
        logger.info("=" * 75)
        logger.info(f" 🏆 WINNERS VS. LOSERS STATS FOR CLASSIFIER: {clf.upper()} 🏆")
        logger.info("=" * 75)
        for cond, cond_data in clf_data.items():
            logger.info(f"\n--- {cond} ---")
            for phase in phases:
                d = cond_data[phase]
                overall_acc, overall_bf = d['overall']
                win_acc, win_bf = d['winners']
                los_acc, los_bf = d['losers']

                overall_bf_str = format_bf(overall_bf)
                win_bf_str = format_bf(win_bf)
                los_bf_str = format_bf(los_bf)

                logger.info(f"  {phase}:")
                logger.info(f"    Overall -> Peak Acc: {overall_acc:.2f}% | Max BF10: {overall_bf_str}")
                logger.info(f"    Winners -> Peak Acc: {win_acc:.2f}% | Max BF10: {win_bf_str}")
                logger.info(f"    Losers  -> Peak Acc: {los_acc:.2f}% | Max BF10: {los_bf_str}")


if __name__ == '__main__':
    extract_winner_loser_stats()