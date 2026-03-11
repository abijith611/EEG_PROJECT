import os
import pickle
import numpy as np
import pandas as pd
import warnings
import glob
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS

# Import R bridge for exact Bayes Factors
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
conv = numpy2ri.converter
bf_pkg = importr('BayesFactor')

def calc_bf_1samp(data, mu=100/3):
    """Calculates directional Bayes Factor against 33.33% chance using R"""
    try:
        with conv.context():
            robjects.globalenv['data'] = data
            robjects.r(f'bf = BayesFactor::ttestBF(x=data, mu={mu}, rscale="medium", nullInterval=c(0.5, Inf))')
            bf_val = robjects.r('as.vector(bf[1])')[0]
        return float(bf_val)
    except Exception as e:
        return 1.0 # Fallback if R fails on a specific bin

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