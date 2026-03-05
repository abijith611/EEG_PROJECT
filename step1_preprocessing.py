"""
Pre-processing script:
  - Load the data and select correct channels for Player 1 and Player 2
  - Epoch the data (-0.2 to 5 seconds)
  - Interpolate noisy channels (using participants.tsv)
  - Down-sample to 256 Hz
  - Save as MNE .fif format

Libraries needed: mne, pandas, numpy, argparse
"""

import os
import argparse
import mne
import numpy as np
import pandas as pd

# Set paths (Updated to match your directory structure)
path_to_data = 'project/ds006761'
identify_bad_channels = False
interpolate_bad_channels = True

# Parameters
num_trials = 480
pair_ids = list(range(1, 10)) + list(range(11, 23)) + list(range(25, 35))
num_pairs = len(pair_ids)
FS = 2048 # Original Biosemi sampling frequency

# Load demographics (to identify bad channels)
part_file = os.path.join(path_to_data, 'participants.tsv')
if os.path.exists(part_file):
    participants = pd.read_csv(part_file, sep='\t')
else:
    print("Warning: participants.tsv not found.")

def run_preprocessing(max_pairs=None):
    # Slice the pair_ids array if max_pairs is provided for testing
    pairs_to_run = pair_ids[:max_pairs] if max_pairs is not None else pair_ids
    total_to_run = len(pairs_to_run)
    
    for p_idx, pair in enumerate(pairs_to_run):
        print(f'Loading pair {p_idx + 1} of {total_to_run} (ID: {pair})')
        
        # Paths
        sub_str = f'sub-{pair:02d}'
        eeg_dir = os.path.join(path_to_data, sub_str, 'eeg')
        events_filename = os.path.join(eeg_dir, f'{sub_str}_task-RPS_events.tsv')
        raw_filename = os.path.join(eeg_dir, f'{sub_str}_task-RPS_eeg.bdf')
        
        if not os.path.exists(events_filename) or not os.path.exists(raw_filename):
            print(f"Skipping {sub_str} - Data files missing")
            continue
            
        # Get triggers
        events_df = pd.read_csv(events_filename, sep='\t')
        stimonsample = events_df['onset_sample'].values
        
        # MNE events array: [sample, 0, trigger_id]
        mne_events = np.column_stack([stimonsample, np.zeros_like(stimonsample), np.ones_like(stimonsample)])
        
        # Load raw BDF data
        raw = mne.io.read_raw_bdf(raw_filename, preload=True, verbose=False)
        ch_names = raw.ch_names
        
        # Process each player (1 and 2)
        for ppt in [1, 2]:
            print(f'  Processing player {ppt}...')
            
            # Identify channels. MATLAB code maps Player 1 to 2-A/2-B and Player 2 to 1-A/1-B
            if ppt == 1:
                target_prefix = '2-'
            else:
                target_prefix = '1-'
                
            # Filter channels based on target prefix (A and B blocks of Biosemi)
            ppt_chans = [ch for ch in ch_names if ch.startswith(f'{target_prefix}A') or ch.startswith(f'{target_prefix}B')]
            
            # Create a copy of raw, pick only the relevant participant's channels
            # Updated to use .pick() to avoid MNE legacy warnings
            raw_ppt = raw.copy().pick(ppt_chans)
            
            # Rename channels to standard Biosemi names (e.g., '2-A1' -> 'A1')
            rename_dict = {ch: ch.replace(target_prefix, '') for ch in ppt_chans}
            raw_ppt.rename_channels(rename_dict)
            
            # Set standard 64-channel biosemi montage (equivalent to biosemi64.mat)
            montage = mne.channels.make_standard_montage('biosemi64')
            raw_ppt.set_montage(montage, on_missing='ignore')
            
            # Epoch the data: -0.2 to 5 seconds
            epochs = mne.Epochs(raw_ppt, mne_events, tmin=-0.2, tmax=5.0, baseline=None, preload=True, verbose=False)
            
            if identify_bad_channels:
                # Plot data to visually identify bad channels
                epochs.copy().filter(l_freq=0.1, h_freq=100.0).plot(n_channels=64, scalings='auto')
                
            if interpolate_bad_channels:
                # Get channels to fix from participants.tsv
                # Column 7 = ppt1 bad chans, Column 12 = ppt2 bad chans (0-indexed: 6 and 11)
                try:
                    ppt_row = participants[participants['participant_id'] == sub_str].iloc[0]
                    col_name = 'player1_bad_channels' if ppt == 1 else 'player2_bad_channels' # adjust column names if needed
                    bad_str = ppt_row[col_name] 
                    
                    if pd.notna(bad_str) and isinstance(bad_str, str) and bad_str.strip() != '':
                        bad_chans = bad_str.split(', ')
                        epochs.info['bads'] = bad_chans
                        
                        # Interpolate bad channels using spherical spline (replicates ft_channelrepair)
                        epochs.interpolate_bads(reset_bads=True, verbose=False)
                        print(f'    Fixed channels: {bad_chans}')
                except Exception as e:
                    pass # Ignore if columns don't match perfectly, proceed without fixing
                
                # Resample to 256 Hz
                epochs.resample(256)
                
                # Save derivatives
                deriv_dir = os.path.join(path_to_data, 'derivatives')
                os.makedirs(deriv_dir, exist_ok=True)
                out_filename = os.path.join(deriv_dir, f'pair-{pair:02d}_player-{ppt}_task-RPS-epo.fif')
                epochs.save(out_filename, overwrite=True)

if __name__ == '__main__':
    # Add a dynamic main to easily test on a subset of subjects
    parser = argparse.ArgumentParser(description="Run EEG preprocessing")
    parser.add_argument('--test_pairs', type=int, default=None, 
                        help="Limit the number of pairs to process for testing (e.g., 4)")
    args = parser.parse_args()
    
    # Run the processing logic
    run_preprocessing(max_pairs=args.test_pairs)