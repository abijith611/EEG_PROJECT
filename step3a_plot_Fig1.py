"""
Plot the behavioural responses
Matches Figure 1 layout using Matplotlib and Seaborn.
Includes pie chart insets, custom color palettes, and game-to-game response changes.
Overlays raw scatter data on violins to mimic the paper's Raincloud visual style.
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
    prop_change = np.zeros((3, num_pairs_run * 2)) # [After Win, After Loss, After Draw]
    
    for p_idx, pair in enumerate(pairs_to_run):
        events_file = os.path.join(path_to_data, f'sub-{pair:02d}', 'eeg', f'sub-{pair:02d}_task-RPS_events.tsv')
        if not os.path.exists(events_file): continue
        events = pd.read_csv(events_file, sep='\t')
        
        # Winner index logic
        w1, w2 = sum(events['outcome']==2), sum(events['outcome']==3)
        winner_idx = 0 if w1 > w2 else 1
        
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
                
        # Calculate Game-to-game response change (Stay/Switch)
        for ppt in [1, 2]:
            resp = events[f'player{ppt}_resp'].values
            outcome = events['outcome'].values
            
            # Align outcome to current player perspective (1=draw, 2=win, 3=loss)
            if ppt == 2:
                outcome_aligned = outcome.copy()
                outcome_aligned[outcome == 2] = 3
                outcome_aligned[outcome == 3] = 2
            else:
                outcome_aligned = outcome
                
            stay_win, stay_loss, stay_draw = [], [], []
            
            for i in range(1, len(resp)):
                if resp[i] > 0 and resp[i-1] > 0: # valid trials
                    changed = (resp[i] != resp[i-1])
                    prev_out = outcome_aligned[i-1]
                    
                    if prev_out == 2: stay_win.append(changed)
                    elif prev_out == 3: stay_loss.append(changed)
                    elif prev_out == 1: stay_draw.append(changed)
            
            ppt_idx = p_idx * 2 + (ppt - 1)
            if stay_win: prop_change[0, ppt_idx] = np.mean(stay_win) * 100
            if stay_loss: prop_change[1, ppt_idx] = np.mean(stay_loss) * 100
            if stay_draw: prop_change[2, ppt_idx] = np.mean(stay_draw) * 100

    # Load Markov Chain data
    mc_file = os.path.join(path_to_data, 'derivatives', 'markov_chain_pred.npy')
    if os.path.exists(mc_file):
        mc_data = np.load(mc_file, allow_pickle=True).item()
        pred_acc = mc_data['Mean_Accuracy'][:, :, 5:] * 100
        pred_acc = pred_acc.reshape(-1, 96)
    else:
        pred_acc = np.zeros((num_pairs_run * 2, 96))
    
    # Figure setup
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout='constrained')
    sns.set_style("white")
    
    # Plot 1: Outcomes Violin
    ax = axes[0, 0]
    df_outcome = pd.DataFrame({
        'Winner\nwins': outcome_summary[:, 1],
        'Loser\nwins': outcome_summary[:, 2],
        'Draw': outcome_summary[:, 0]
    })
    sns.violinplot(data=df_outcome, ax=ax, palette=['#D95319', '#EDB120', '#7E2F8E'], inner='box', cut=0)
    sns.stripplot(data=df_outcome, ax=ax, color='k', alpha=0.5, jitter=True, size=4)
    ax.axhline(33.33, color='k', linestyle='--')
    ax.set_ylabel('Percentage')
    
    # Plot 2: Ranked Resp Violin + Pie Insets
    ax = axes[0, 1]
    df_ranked = pd.DataFrame({
        'Most\nplayed': ranked_resp[0, :],
        'Mid\nplayed': ranked_resp[1, :],
        'Least\nplayed': ranked_resp[2, :]
    })
    sns.violinplot(data=df_ranked, ax=ax, palette=['#B30000', '#E64D00', '#FFB300'], inner='box', cut=0)
    sns.stripplot(data=df_ranked, ax=ax, color='k', alpha=0.5, jitter=True, size=4)
    ax.axhline(33.33, color='k', linestyle='--')
    
    # Add Pie Charts above Violins
    rps_colors = ['#4DBEEE', '#77AC30', '#EDB120'] # Rock, Paper, Scissors colors
    for i in range(3):
        ax_inset = ax.inset_axes([0.15 + i*0.31, 0.85, 0.15, 0.15])
        rank_data = all_played_rank[i, :]
        counts = [np.sum(rank_data == 1), np.sum(rank_data == 2), np.sum(rank_data == 3)]
        ax_inset.pie(counts, labels=['R', 'P', 'S'], colors=rps_colors, textprops={'fontsize': 8})

    # Plot 3: Response Change (Switch rate)
    ax = axes[1, 0]
    df_change = pd.DataFrame({
        'After\nwin': prop_change[0, :],
        'After\nloss': prop_change[1, :],
        'After\ndraw': prop_change[2, :]
    })
    sns.violinplot(data=df_change, ax=ax, palette=['#4DBEEE', '#77AC30', '#7E2F8E'], inner='box', cut=0)
    sns.stripplot(data=df_change, ax=ax, color='k', alpha=0.5, jitter=True, size=4)
    ax.axhline(66.67, color='k', linestyle='--')
    ax.set_ylabel('Percentage')
    
    # Plot 4: Predictability (Markov Chain)
    ax = axes[1, 1]
    x_axis = np.arange(5, 101)
    
    valid_pred_acc = pred_acc[~np.isnan(pred_acc).all(axis=1)]
    if len(valid_pred_acc) > 0:
        mean_acc = np.nanmean(valid_pred_acc, axis=0)
        ci = stats.t.ppf(0.975, valid_pred_acc.shape[0]-1) * np.nanstd(valid_pred_acc, axis=0) / np.sqrt(valid_pred_acc.shape[0])
        
        for i in range(valid_pred_acc.shape[0]):
            ax.plot(x_axis, valid_pred_acc[i,:], color='gray', alpha=0.15)
        ax.plot(x_axis, mean_acc, color='#0072BD', linewidth=2)
        ax.fill_between(x_axis, mean_acc - ci, mean_acc + ci, color='#0072BD', alpha=0.2)
        
    ax.axhline(33.33, color='k', linestyle='--')
    ax.set_xlabel('N previous games')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(25, 65)
    
    plt.savefig(os.path.join(plot_dir, 'Figure1.png'), dpi=300, bbox_inches='tight')
    print("Figure 1 saved.")

if __name__ == '__main__':
    plot_behavior()