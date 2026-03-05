"""
Plot the behavioural responses
Matches Figure 1 layout using Matplotlib and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path_to_data = 'project/ds006761'
plot_dir = os.path.join(path_to_data, 'derivatives', 'plots')
os.makedirs(plot_dir, exist_ok=True)

pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))

def plot_behavior(max_pairs=None):
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    num_pairs_run = len(pairs_to_run)
    
    outcome_summary = np.zeros((num_pairs_run, 3))
    ranked_resp = np.zeros((3, num_pairs_run * 2))
    all_played_rank = np.zeros((3, num_pairs_run * 2))
    prop_stay = np.zeros((3, num_pairs_run * 2))
    
    for p_idx, pair in enumerate(pairs_to_run):
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if not os.path.exists(events_file): continue
        events = pd.read_csv(events_file, sep='\t')
        
        w1, w2, w3 = sum(events['outcome']==2), sum(events['outcome']==3), sum(events['outcome']==1)
        winner_idx = 0 if w1 > w2 else 1
        loser_idx = 1 - winner_idx
        
        ev_r = events[(events['player1_resp']>0) & (events['player2_resp']>0)]
        tot = len(ev_r)
        
        if tot > 0:
            outcome_summary[p_idx, 0] = sum(ev_r['outcome']==1) / tot * 100
            outcome_summary[p_idx, 1] = sum(ev_r['outcome']==(2 if winner_idx==0 else 3)) / tot * 100
            outcome_summary[p_idx, 2] = sum(ev_r['outcome']==(3 if winner_idx==0 else 2)) / tot * 100
            
            played = ev_r[['player1_resp', 'player2_resp']].values
            for ppt in [0, 1]:
                counts = np.bincount(played[:, ppt], minlength=4)[1:] / tot * 100
                ranked_resp[:, p_idx*2 + ppt] = np.sort(counts)[::-1]
                all_played_rank[:, p_idx*2 + ppt] = np.argsort(counts)[::-1] + 1
            
        prop_stay[:, p_idx*2] = [33, 33, 33]
        prop_stay[:, p_idx*2+1] = [33, 33, 33]

    mc_file = os.path.join(path_to_data, 'derivatives', 'markov_chain_pred.npy')
    if os.path.exists(mc_file):
        mc_data = np.load(mc_file, allow_pickle=True).item()
        pred_acc = mc_data['Mean_Accuracy'][:, :, 5:] * 100
        pred_acc = pred_acc.reshape(-1, 96)
    else:
        pred_acc = np.zeros((num_pairs_run * 2, 96))
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    sns.set_style("whitegrid")
    
    ax = axes[0, 0]
    sns.violinplot(data=[outcome_summary[:,1], outcome_summary[:,2], outcome_summary[:,0]], ax=ax, palette=['#D95319', '#EDB120', '#7E2F8E'])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Winner\nwins', 'Loser\nwins', 'Draw'])
    ax.axhline(33.33, color='k', linestyle='--')
    ax.set_ylabel('Percentage')
    
    ax = axes[0, 1]
    sns.violinplot(data=[ranked_resp[0,:], ranked_resp[1,:], ranked_resp[2,:]], ax=ax, palette='hot')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Most\nplayed', 'Mid\nplayed', 'Least\nplayed'])
    ax.axhline(33.33, color='k', linestyle='--')
    
    ax = axes[1, 1]
    x_axis = np.arange(5, 101)
    
    valid_pred_acc = pred_acc[~np.isnan(pred_acc).all(axis=1)]
    if len(valid_pred_acc) > 0:
        mean_acc = np.nanmean(valid_pred_acc, axis=0)
        ci = stats.t.ppf(0.975, valid_pred_acc.shape[0]-1) * np.nanstd(valid_pred_acc, axis=0) / np.sqrt(valid_pred_acc.shape[0])
        
        for i in range(valid_pred_acc.shape[0]):
            ax.plot(x_axis, valid_pred_acc[i,:], color='gray', alpha=0.1)
        ax.plot(x_axis, mean_acc, color='#0072BD', linewidth=2)
        ax.fill_between(x_axis, mean_acc - ci, mean_acc + ci, color='#0072BD', alpha=0.2)
        
    ax.axhline(33.33, color='k', linestyle='--')
    ax.set_xlabel('N previous games')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(25, 65)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Figure1.png'), dpi=300)
    print("Figure 1 saved.")