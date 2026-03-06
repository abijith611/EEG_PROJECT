import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_markov_analysis(path_to_data, pair_ids):
    """Predicts player responses based on history (Window size 5-100)[cite: 553]."""
    results = {}
    for pair in pair_ids:
        events_path = os.path.join(path_to_data, f"sub-{pair:02d}", 'eeg', f"sub-{pair:02d}_task-RPS_events.tsv")
        events = pd.read_csv(events_path, sep='\t')
        for ppt in [1, 2]:
            resp = events[f'player{ppt}_resp'].values
            accs = []
            for N in range(5, 101):
                correct = 0
                count = 0
                for i in range(2, len(resp)):
                    history = resp[max(0, i-N):i]
                    if resp[i] == 0 or resp[i-1] == 0: continue
                    # Find most likely next move based on history
                    occ = np.where(history[:-1] == resp[i-1])[0]
                    if len(occ) > 0:
                        pred = pd.Series(history[occ+1]).mode().iloc[0]
                        if pred == resp[i]: correct += 1
                        count += 1
                accs.append(correct/count if count > 0 else 0)
            results[f"pair_{pair:02d}_ppt_{ppt}"] = accs
    return pd.DataFrame(results, index=range(5, 101))

def run_decoding_analysis(all_epochs, path_to_data, pair_ids):
    """Decodes own response using LDA across 250ms bins[cite: 546, 558, 567]."""
    decoding_results = {}
    times = np.arange(0, 5.25, 0.25) # 250ms windows [cite: 546]
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    for key, epochs in all_epochs.items():
        pair_num = int(key.split('_')[1])
        ppt_num = int(key.split('_')[3])
        events_path = os.path.join(path_to_data, f"sub-{pair_num:02d}", 'eeg', f"sub-{pair_num:02d}_task-RPS_events.tsv")
        events = pd.read_csv(events_path, sep='\t')
        y = events['player1_resp' if ppt_num == 1 else 'player2_resp'].values
        
        # Slicing data into bins and running 10-fold CV [cite: 567]
        data = epochs.get_data()
        accuracies = []
        for i in range(len(times)-1):
            t_idx = np.where((epochs.times >= times[i]) & (epochs.times < times[i+1]))[0]
            X = data[:, :, t_idx].mean(axis=2)
            valid = np.where(y > 0)[0] # Exclude no-response [cite: 564]
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(lda, X[valid], y[valid], cv=cv)
            accuracies.append(scores.mean())
        decoding_results[key] = accuracies
    return decoding_results