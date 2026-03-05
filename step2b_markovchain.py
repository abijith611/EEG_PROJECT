"""
RPS Markov Chain
Predict the response of the player based on N previous trials.
Window sizes from 5 to 100.
"""

import os
import numpy as np
import pandas as pd

path_to_data = 'project/ds006761'
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_trials = 480
num_windows = 100

def run_markov(max_pairs=None):
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs_run = len(pairs_to_run)
    
    Mean_Accuracy = np.zeros((num_pairs_run, 2, num_windows + 1))
    M_pred = np.zeros((num_pairs_run, 2, num_windows + 1, num_trials, 4))
    
    for p_idx, pair in enumerate(pairs_to_run):
        print(f'Loading pair {p_idx + 1} of {num_pairs_run} (ID: {pair})')
        sub_str = f'sub-{pair:02d}'
        
        events_file = os.path.join(path_to_data, sub_str, 'eeg', f'{sub_str}_task-RPS_events.tsv')
        if not os.path.exists(events_file):
            continue
            
        events = pd.read_csv(events_file, sep='\t')
        
        for ppt in [1, 2]:
            ppt_col = 'player1_resp' if ppt == 1 else 'player2_resp'
            resp = events[ppt_col].values
            
            prob_data = np.full((num_trials, 13), np.nan)
            prob_data[0, :] = [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1]
            
            for i in range(1, num_trials):
                prob_data[i, :] = prob_data[i-1, :]
                prob_data[i, 0] = i + 1
                
                r_prev = resp[i-1]
                r_curr = resp[i]
                
                if r_prev == 1:
                    prob_data[i, 1] += 1
                    if r_curr == 1: prob_data[i, 2] += 1
                    elif r_curr == 2: prob_data[i, 3] += 1
                    else: prob_data[i, 4] += 1
                elif r_prev == 2:
                    prob_data[i, 5] += 1
                    if r_curr == 1: prob_data[i, 6] += 1
                    elif r_curr == 2: prob_data[i, 7] += 1
                    else: prob_data[i, 8] += 1
                elif r_prev == 3:
                    prob_data[i, 9] += 1
                    if r_curr == 1: prob_data[i, 10] += 1
                    elif r_curr == 2: prob_data[i, 11] += 1
                    else: prob_data[i, 12] += 1
                    
            prob_res = np.full((num_trials, 4), np.nan)
            m_Prob = np.full((3, 3), 1/3)
            inter_prob_data = np.full((num_trials, 13), np.nan)
            
            for window_size in range(5, 101):
                for i in range(2, num_trials):
                    if i < window_size:
                        inter_prob_data[i, :] = prob_data[i-1, :]
                    else:
                        inter_prob_data[i, :] = prob_data[i-1, :] - prob_data[i-window_size, :]
                    
                    inter_prob_data[i, 0] = i + 1
                    
                    if inter_prob_data[i, 1] > 0: m_Prob[0, :] = inter_prob_data[i, 2:5] / inter_prob_data[i, 1]
                    else: m_Prob[0, :] = [1/3, 1/3, 1/3]
                        
                    if inter_prob_data[i, 5] > 0: m_Prob[1, :] = inter_prob_data[i, 6:9] / inter_prob_data[i, 5]
                    else: m_Prob[1, :] = [1/3, 1/3, 1/3]
                        
                    if inter_prob_data[i, 9] > 0: m_Prob[2, :] = inter_prob_data[i, 10:13] / inter_prob_data[i, 9]
                    else: m_Prob[2, :] = [1/3, 1/3, 1/3]
                        
                    prob_res[i, 0] = resp[i]
                    
                    idx = i
                    if resp[i-1] > 0: idx = i
                    elif i>1 and resp[i-2] > 0: idx = i-1
                    else: idx = i-2
                    
                    if idx > 0:
                        last_r = resp[idx-1] - 1
                        if last_r >= 0 and last_r <= 2:
                            best_pred = np.argmax(m_Prob[int(last_r), :]) + 1
                            prob_res[i, 1] = best_pred
                            prob_res[i, 2] = np.max(m_Prob[int(last_r), :])
                            
                    if np.isnan(prob_res[i, 2]):
                        prob_res[i, 3] = np.nan
                    elif prob_res[i, 0] == prob_res[i, 1]:
                        prob_res[i, 3] = 1
                    else:
                        prob_res[i, 3] = 0
                        
                data_mean = prob_res[2:480, 3]
                data_mean = data_mean[np.isfinite(data_mean)]
                if len(data_mean) > 0:
                    Mean_Accuracy[p_idx, ppt-1, window_size] = np.mean(data_mean)
                M_pred[p_idx, ppt-1, window_size, :, :] = prob_res
                
    deriv_dir = os.path.join(path_to_data, 'derivatives')
    os.makedirs(deriv_dir, exist_ok=True)
    out_file = os.path.join(deriv_dir, 'markov_chain_pred.npy')
    np.save(out_file, {'M_pred': M_pred, 'Mean_Accuracy': Mean_Accuracy})