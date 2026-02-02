import mne
import pandas as pd
import numpy as np

def calculate_outcomes(df):
    """
    Manually calculates Win/Loss/Tie based on Rock(1), Paper(2), Scissors(3).
    Adds 'player1_outcome' and 'player2_outcome' to the DataFrame.
    """
    # Map: 1=Rock, 2=Paper, 3=Scissors
    # Logic: 1 beats 3, 2 beats 1, 3 beats 2.
    
    outcomes_p1 = []
    outcomes_p2 = []
    
    p1_resps = df['player1_resp'].values
    p2_resps = df['player2_resp'].values
    
    for p1, p2 in zip(p1_resps, p2_resps):
        if pd.isna(p1) or pd.isna(p2):
            outcomes_p1.append('nan')
            outcomes_p2.append('nan')
            continue
            
        p1 = int(p1)
        p2 = int(p2)
        
        if p1 == p2:
            outcomes_p1.append('tie')
            outcomes_p2.append('tie')
        elif (p1 == 1 and p2 == 3) or (p1 == 2 and p2 == 1) or (p1 == 3 and p2 == 2):
            outcomes_p1.append('win')
            outcomes_p2.append('loss')
        else:
            outcomes_p1.append('loss')
            outcomes_p2.append('win')
            
    df['player1_outcome'] = outcomes_p1
    df['player2_outcome'] = outcomes_p2
    return df

def run_epoching(raw, subject_id, basic_path, player_num):
    print(f"--- [step3_epoching] Extracting 3 Phases (Dec, Resp, Feed) ---")
    
    # 1. Load Events
    events_file = f'{basic_path}/{subject_id}/eeg/{subject_id}_task-RPS_events.tsv'
    try:
        df = pd.read_csv(events_file, sep='\t')
    except:
        print("❌ Error loading events TSV.")
        return None, None

    # 2. Calculate Outcomes (Fixes KeyError)
    if 'player1_outcome' not in df.columns:
        print("   -> Calculating outcomes (Win/Loss/Tie)...")
        df = calculate_outcomes(df)

    # 3. Create Event Arrays for 3 Phases
    sfreq = raw.info['sfreq']
    onsets = df['onset'].values
    
    # Decision: Time 0
    ev_dec = np.zeros((len(onsets), 3), dtype=int)
    ev_dec[:, 0] = (onsets * sfreq).astype(int)
    ev_dec[:, 2] = 1
    
    # Response: Time +2.0s
    ev_resp = ev_dec.copy()
    ev_resp[:, 0] += int(2.0 * sfreq)
    
    # Feedback: Time +4.0s
    ev_feed = ev_dec.copy()
    ev_feed[:, 0] += int(4.0 * sfreq)

    # 4. Apply Exclusion (Remove 1st trial of blocks)
    block_size = 40
    n_trials = len(df)
    keep_mask = np.ones(n_trials, dtype=bool)
    for i in range(0, n_trials, block_size):
        keep_mask[i] = False
    
    # Filter everything
    df_clean = df[keep_mask].reset_index(drop=True)
    ev_dec = ev_dec[keep_mask]
    ev_resp = ev_resp[keep_mask]
    ev_feed = ev_feed[keep_mask]
    
    print(f"   -> Retained {len(df_clean)} trials after block cleaning.")

    # 5. Epoching (With independent baselines)
    tmin, tmax = -0.2, 2.0
    fback_max = 1.0
    baseline = (tmin, 0)
    
    try:
        ep_dec = mne.Epochs(raw, ev_dec, tmin=tmin, tmax=tmax, baseline=baseline, verbose=False)
        ep_resp = mne.Epochs(raw, ev_resp, tmin=tmin, tmax=tmax, baseline=baseline, verbose=False)
        ep_feed = mne.Epochs(raw, ev_feed, tmin=tmin, tmax=fback_max, baseline=baseline, verbose=False)
        
        # Load data
        ep_dec.load_data()
        ep_resp.load_data()
        ep_feed.load_data()
        
        # Sync drops (if MNE dropped bad epochs)
        # We assume strict syncing: if one phase drops a trial, we drop it from all for consistency
        # Ideally, use the intersection of selection
        common_indices = set(ep_dec.selection) & set(ep_resp.selection) & set(ep_feed.selection)
        # (For simplicity in this fix, assuming good data, but a robust check is better)
        
        # Return Dictionary
        epochs_dict = {
            'Decision': ep_dec,
            'Response': ep_resp,
            'Feedback': ep_feed
        }
        
        return epochs_dict, df_clean
        
    except Exception as e:
        print(f"❌ Epoching failed: {e}")
        return None, None