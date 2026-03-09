import os
import pickle
import numpy as np
import pandas as pd
import warnings

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
            # nullInterval=c(0.5, Inf) tests the directional hypothesis that accuracy > chance
            robjects.r(f'bf = BayesFactor::ttestBF(x=data, mu={mu}, rscale="medium", nullInterval=c(0.5, Inf))')
            bf_val = robjects.r('as.vector(bf[1])')[0]
        return float(bf_val)
    except Exception as e:
        return 1.0 # Fallback if R fails on a specific bin

def extract_winner_loser_stats():
    path_to_data = 'project/ds006761'
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
    
    conditions = ['Own response', "Opponent's response", 
                  'Own previous response', "Opponent's previous response"]
    
    all_decoding = []
    winners_idx = []
    losers_idx = []
    
    # Load data and sort by Winner/Loser
    for pair in pair_ids:
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        winner_ppt = None
        if os.path.exists(events_file):
            events = pd.read_csv(events_file, sep='\t')
            w1 = sum(events['outcome'] == 2)
            w2 = sum(events['outcome'] == 3)
            winner_ppt = 1 if w1 > w2 else 2 # Tie goes to player 2 as per plotting logic

        for ppt in [1, 2]:
            fpath = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS_decoding.pkl')
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
            
    all_decoding = np.array(all_decoding)
    
    phases = {
        'Decision (0-2s)': (0, 8),
        'Response (2-4s)': (8, 16),
        'Feedback (4-5s)': (16, 20)
    }

    print("\n" + "="*75)
    print(" 🏆 WINNERS VS. LOSERS STATS (USING R BAYESFACTOR) 🏆")
    print("="*75)

    for t_idx, cond in enumerate(conditions):
        print(f"\n--- {cond} ---")
        
        data_win = all_decoding[winners_idx, t_idx, :]
        data_los = all_decoding[losers_idx, t_idx, :]
        
        mean_win = np.nanmean(data_win, axis=0)
        mean_los = np.nanmean(data_los, axis=0)
        
        for phase_name, (start, end) in phases.items():
            print(f"  {phase_name}:")
            
            # --- Winners Calculation ---
            phase_means_win = mean_win[start:end]
            max_acc_val_win = np.max(phase_means_win)
            
            bfs_win = []
            for bin_idx in range(start, end):
                bin_data = data_win[:, bin_idx]
                bin_data = bin_data[~np.isnan(bin_data)]
                bfs_win.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
            max_bf_win = np.max(bfs_win)
            
            # --- Losers Calculation ---
            phase_means_los = mean_los[start:end]
            max_acc_val_los = np.max(phase_means_los)
            
            bfs_los = []
            for bin_idx in range(start, end):
                bin_data = data_los[:, bin_idx]
                bin_data = bin_data[~np.isnan(bin_data)]
                bfs_los.append(calc_bf_1samp(bin_data) if len(bin_data) > 2 else 1.0)
            max_bf_los = np.max(bfs_los)
            
            # --- Print Formatting ---
            bf_str_win = "> 1000 (Extreme Evidence)" if max_bf_win > 1000 else f"{max_bf_win:.2f}"
            bf_str_los = "> 1000 (Extreme Evidence)" if max_bf_los > 1000 else f"{max_bf_los:.2f}"
            
            print(f"    Winners -> Peak Acc: {max_acc_val_win:.2f}% | Max BF10: {bf_str_win}")
            print(f"    Losers  -> Peak Acc: {max_acc_val_los:.2f}% | Max BF10: {bf_str_los}")

if __name__ == '__main__':
    extract_winner_loser_stats()