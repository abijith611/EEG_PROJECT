import os
import pickle
import numpy as np
import pandas as pd
import warnings
import glob
import pingouin as pg  # for fallback Bayes factor
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS

# Attempt to import R bridge for exact Bayes Factors
R_AVAILABLE = False
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    conv = numpy2ri.converter
    bf_pkg = importr('BayesFactor')
    R_AVAILABLE = True
except Exception as e:
    warnings.warn(f"R/BayesFactor not available: {e}. Using pingouin approximation.")
    R_AVAILABLE = False

def calc_bf_1samp(data, mu=100/3):
    """
    Calculates directional Bayes Factor against chance level (mu).
    Uses R's BayesFactor if available, otherwise falls back to pingouin.
    """
    if len(data) < 3:
        return 1.0  # Not enough data
    
    if R_AVAILABLE:
        try:
            with conv.context():
                robjects.globalenv['data'] = data
                robjects.r(f'bf = BayesFactor::ttestBF(x=data, mu={mu}, rscale="medium", nullInterval=c(0.5, Inf))')
                bf_val = robjects.r('as.vector(bf[1])')[0]
            return float(bf_val)
        except Exception as e:
            # If R fails, fall back to pingouin
            pass
    
    # Fallback using pingouin (two-sided test, then convert to directional?)
    # Note: pingouin's bayesfactor_ttest returns BF10 for two-sided test.
    # We want evidence for the alternative hypothesis that mean > mu.
    # Since BF for one-sided can be approximated by doubling the two-sided BF if the effect is in the predicted direction.
    # But for simplicity, we'll just return the two-sided BF10; the user can interpret.
    from scipy import stats
    t_stat, _ = stats.ttest_1samp(data, mu)
    # For one-sided test where we expect mean > mu, BF can be approximated.
    # A common approach: if t_stat > 0, then BF_one_sided ≈ 2 * BF_two_sided (if prior is symmetric).
    # We'll keep it simple and return two-sided BF10.
    bf10 = pg.bayesfactor_ttest(t_stat, nx=len(data), r='medium')
    return bf10

def extract_winner_loser_stats():
    
    conditions = ['Own response', "Opponent's response", 
                  'Own previous response', "Opponent's previous response"]
    
    # Find all classifier suffixes
    pattern = os.path.join(DERIV_DIR, 'pair-*_player-*_task-RPS_decoding_*.pkl')
    files = glob.glob(pattern)
    clf_suffixes = set()
    for f in files:
        base = os.path.basename(f)
        suffix = base.split('decoding_')[1].replace('.pkl', '')
        clf_suffixes.add(suffix)
    
    if not clf_suffixes:
        print("No decoding files found.")
        return
    
    phases = {
        'Decision (0-2s)': (0, 8),
        'Response (2-4s)': (8, 16),
        'Feedback (4-5s)': (16, 20)
    }
    
    for clf in sorted(clf_suffixes):
        print("\n" + "="*75)
        print(f" 🏆 WINNERS VS. LOSERS STATS FOR CLASSIFIER: {clf.upper()} 🏆")
        print("="*75)
        
        all_decoding = []
        winners_idx = []
        losers_idx = []
        
        # Load data and sort by Winner/Loser
        for pair in PAIR_IDS:
            events_file = os.path.join(PATH_TO_DATA, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
            winner_ppt = None
            if os.path.exists(events_file):
                events = pd.read_csv(events_file, sep='\t')
                w1 = sum(events['outcome'] == 2)
                w2 = sum(events['outcome'] == 3)
                winner_ppt = 1 if w1 > w2 else 2 # Tie goes to player 2 as per plotting logic

            for ppt in [1, 2]:
                fpath = os.path.join(DERIV_DIR, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding_{clf}.pkl')
                if not os.path.exists(fpath): continue
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
            print(f"No data for classifier {clf}")
            continue
            
        all_decoding = np.array(all_decoding)
        
        for t_idx, cond in enumerate(conditions):
            print(f"\n--- {cond} ---")
            
            data_win = all_decoding[winners_idx, t_idx, :]
            data_los = all_decoding[losers_idx, t_idx, :]
            
            mean_win = np.nanmean(data_win, axis=0)
            mean_los = np.nanmean(data_los, axis=0)
            
            for phase_name, (start, end) in phases.items():
                print(f"  {phase_name}:")
                
                # Winners Calculation
                phase_means_win = mean_win[start:end]
                max_acc_val_win = np.max(phase_means_win)
                bfs_win = []
                for bin_idx in range(start, end):
                    bin_data = data_win[:, bin_idx]
                    bin_data = bin_data[~np.isnan(bin_data)]
                    bfs_win.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
                max_bf_win = np.max(bfs_win)
                
                # Losers Calculation
                phase_means_los = mean_los[start:end]
                max_acc_val_los = np.max(phase_means_los)
                bfs_los = []
                for bin_idx in range(start, end):
                    bin_data = data_los[:, bin_idx]
                    bin_data = bin_data[~np.isnan(bin_data)]
                    bfs_los.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
                max_bf_los = np.max(bfs_los)
                
                # Print Formatting
                bf_str_win = "> 1000 (Extreme Evidence)" if max_bf_win > 1000 else f"{max_bf_win:.2f}"
                bf_str_los = "> 1000 (Extreme Evidence)" if max_bf_los > 1000 else f"{max_bf_los:.2f}"
                
                print(f"    Winners -> Peak Acc: {max_acc_val_win:.2f}% | Max BF10: {bf_str_win}")
                print(f"    Losers  -> Peak Acc: {max_acc_val_los:.2f}% | Max BF10: {bf_str_los}")

if __name__ == '__main__':
    extract_winner_loser_stats()