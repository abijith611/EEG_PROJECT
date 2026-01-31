import mne
import pandas as pd
import numpy as np

# step3_epoching.py - MODIFY run_epoching function

def run_epoching(raw, subject_id, basic_path, player_num):
    print(f"--- [step3_epoching] Extracting 3 Phases (Dec, Resp, Feed) ---")
    
    events_file = f'{basic_path}/{subject_id}/eeg/{subject_id}_task-RPS_events.tsv'
    df = pd.read_csv(events_file, sep='\t')
    
    # ================================================================
    # CRITICAL: REMOVE FIRST TRIAL OF EACH BLOCK (MATLAB PAPER'S METHOD)
    # ================================================================
    # The experiment has 480 trials, divided into 12 blocks of 40 trials
    # Paper removes the first trial of each block
    block_size = 40
    n_blocks = 12
    
    # Create a mask to identify first trial of each block
    # These are indices: 0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440
    remove_mask = np.zeros(len(df), dtype=bool)
    for block in range(n_blocks):
        block_start = block * block_size
        if block_start < len(df):
            remove_mask[block_start] = True
    
    n_removed = np.sum(remove_mask)
    print(f"   Removing {n_removed} trials (first of each block)")
    
    # Keep only non-removed trials
    df_keep = df[~remove_mask].copy().reset_index(drop=True)
    print(f"   Remaining trials: {len(df_keep)} (was {len(df)})")
    # ================================================================
    
    # Now use df_keep instead of df for the rest of the function
    # ... rest of your existing code, but use df_keep
    
    resp_col = f'player{player_num}_resp'
    sfreq = raw.info['sfreq']
    
    events_list = []
    valid_indices = []
    
    # Use df_keep instead of df
    for index, row in df_keep.iterrows():
        try:
            val = int(row[resp_col])
            if val in [1, 2, 3]:  # Rock, Paper, Scissors only
                onset_sample = int(row['onset'] * sfreq)
                events_list.append([onset_sample, 0, val])
                valid_indices.append(index)
        except:
            continue
    
    # Convert to standard MNE event array
    base_events = np.array(events_list, dtype=int)
    print(f"   Found {len(base_events)} valid trials after removal")
    
    # Filter DataFrame to match valid events
    df_clean = df_keep.loc[valid_indices].reset_index(drop=True)

    # 2. CREATE SHIFTED EVENTS (Simulating Screen Onsets)
    # The paper assumes fixed timing: Decision(0s), Response(+2s), Feedback(+4s)
    
    # Phase 1: Decision (Original Onset)
    ev_dec = base_events.copy()
    
    # Phase 2: Response (Shift +2.0 seconds)
    ev_resp = base_events.copy()
    ev_resp[:, 0] += int(2.0 * sfreq)
    
    # Phase 3: Feedback (Shift +4.0 seconds)
    ev_feed = base_events.copy()
    ev_feed[:, 0] += int(4.0 * sfreq)

    # 3. EPOCHING (The Critical Fix)
    event_id = {'Rock': 1, 'Paper': 2, 'Scissors': 3}
    common_baseline = (-0.2, 0) # Apply baseline -200ms before EACH phase start
    
    print("   -> Epoching Decision Phase (-0.2 to 2.0s)...")
    epochs_dec = mne.Epochs(raw, ev_dec, event_id, tmin=-0.2, tmax=2.0, 
                            baseline=common_baseline, preload=True, verbose=False)

    print("   -> Epoching Response Phase (-0.2 to 2.0s)...")
    epochs_resp = mne.Epochs(raw, ev_resp, event_id, tmin=-0.2, tmax=2.0, 
                             baseline=common_baseline, preload=True, verbose=False)

    print("   -> Epoching Feedback Phase (-0.2 to 1.0s)...")
    epochs_feed = mne.Epochs(raw, ev_feed, event_id, tmin=-0.2, tmax=1.0, 
                             baseline=common_baseline, preload=True, verbose=False)
    
    # Return a TUPLE containing all 3 phases
    return (epochs_dec, epochs_resp, epochs_feed), df_clean