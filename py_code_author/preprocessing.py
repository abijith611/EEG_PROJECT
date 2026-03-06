import mne
import pandas as pd
import numpy as np
import os

def run_preprocessing(path_to_data, pair_ids, participants_df):
    """
    Cleans and epochs raw EEG data for each pair.
    Returns: dict of { 'pair_XX_ppt_X': mne.Epochs }
    """
    all_epochs = {}
    FS_NEW = 256
    standard_montage = mne.channels.make_standard_montage('biosemi64')

    for pair in pair_ids:
        pair_str = f"sub-{pair:02d}"
        raw_file = os.path.join(path_to_data, pair_str, 'eeg', f"{pair_str}_task-RPS_eeg.bdf")
        events_file = os.path.join(path_to_data, pair_str, 'eeg', f"{pair_str}_task-RPS_events.tsv")
        
        # Load events and raw data [cite: 1, 535, 540]
        events_df = pd.read_csv(events_file, sep='\t')
        raw = mne.io.read_raw_bdf(raw_file, preload=True, verbose=False)
        
        # Player 1 is recorded as 2-A/B; Player 2 as 1-A/B 
        for ppt in [1, 2]:
            pattern = '2-' if ppt == 1 else '1-'
            
            # FIXED: Specifically target EEG channels (A1-A32, B1-B32) 
            # This excludes auxiliary channels like GSR, Resp, and Temp 
            ch_names = [ch for ch in raw.ch_names if (pattern + 'A' in ch) or (pattern + 'B' in ch)]
            
            # Ensure we have exactly 64 channels to match the montage 
            ch_names = ch_names[:64]
            
            ppt_raw = raw.copy().pick_channels(ch_names)
            
            # Map raw labels (e.g., '2-A1') to montage labels (e.g., 'A1') 
            ppt_raw.rename_channels({old: new for old, new in zip(ch_names, standard_montage.ch_names)})
            ppt_raw.set_montage(standard_montage)
            
            # Re-reference to common average [cite: 1, 536]
            ppt_raw.set_eeg_reference('average', projection=False)
            
            # Interpolate bad channels [cite: 1, 542, 543]
            bad_info = participants_df.loc[participants_df['participant_id'] == pair_str]
            bad_chans = bad_info.iloc[0, 6 if ppt == 1 else 11] 
            if pd.notna(bad_chans):
                ppt_raw.info['bads'] = bad_chans.split(', ')
                ppt_raw.interpolate_bads(reset_bads=True)
            
            # Create epochs: -0.2 to 5.0s relative to decision onset [cite: 1, 536, 544]
            event_matrix = np.column_stack([events_df['onset_sample'].values, 
                                            np.zeros(len(events_df), dtype=int), 
                                            np.ones(len(events_df), dtype=int)])
            
            epochs = mne.Epochs(ppt_raw, event_matrix, tmin=-0.2, tmax=5.0, 
                                baseline=(-0.2, 0), preload=True, verbose=False)
            epochs.resample(FS_NEW)
            
            all_epochs[f"pair_{pair:02d}_ppt_{ppt}"] = epochs
            
    return all_epochs