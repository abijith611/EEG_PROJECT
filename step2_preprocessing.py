import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import pandas as pd
import os

def load_bad_channels_from_participants(subject_id, basic_path, player_num):
    """
    Load bad channels from participants.tsv - EXACT COLUMN NAMES
    """
    participants_file = f'{basic_path}/participants.tsv'
    
    if not os.path.exists(participants_file):
        print(f"   participants.tsv not found at {participants_file}")
        return []
    
    try:
        df = pd.read_csv(participants_file, sep='\t')
        
        # Find the subject
        subject_row = df[df['participant_id'] == subject_id]
        if subject_row.empty:
            print(f"   Subject {subject_id} not found in participants.tsv")
            return []
        
        # Get the correct column name
        col_name = f'player{player_num}_pre_processing_channels_fixed'
        
        if col_name not in subject_row.columns:
            print(f"   Column {col_name} not found in participants.tsv")
            return []
        
        # Get the value
        value = subject_row[col_name].values[0]
        
        # Check if it's NaN or empty
        if pd.isna(value):
            print(f"   No bad channels specified for {subject_id} (NaN)")
            return []
        
        # Convert to string and clean
        channels_str = str(value).strip()
        
        if channels_str == '' or channels_str.lower() == 'nan':
            print(f"   No bad channels specified for {subject_id} (empty)")
            return []
        
        # Parse comma-separated channels
        # Example: "FC5, T7, POz, P2"
        channels = []
        for ch in channels_str.split(','):
            ch_clean = ch.strip()
            if ch_clean:  # Not empty
                channels.append(ch_clean)
        
        print(f"   Loaded {len(channels)} bad channels from {col_name}: {channels}")
        return channels
        
    except Exception as e:
        print(f"   Error loading bad channels: {e}")
        return []

def run_preprocessing(raw, subject_id=None, basic_path=None, player_num=None, target_rate=256):
    """
    Updated version that accepts optional bad channel parameters
    """
    print("\n" + "="*50)
    print(f"--- [step2_preprocessing] Cleaning Pipeline ---")
    if subject_id and player_num:
        print(f"   Subject: {subject_id}, Player: {player_num}")
    
    raw = raw.copy()

    # =========================================================================
    # 1. TRANSLATION LAYER (The Fix for BioSemi Names)
    # =========================================================================
    # We map the 64 raw channels (A1..A32, B1..B32) to standard 10-20 locations.
    # This list follows the standard BioSemi 64-channel layout order.
    standard_64_names = [
        # === A-SIDE (Left Hemisphere / Center) ===
        # Matches indices 0 to 31 (A1 to A32)
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 
        'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 
        'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',

        # === B-SIDE (Right Hemisphere / Center) ===
        # Matches indices 32 to 63 (B1 to B32)
        'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 
        'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 
        'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'Fpz' 
    ]


    print(f"   Original First Name: {raw.ch_names[0]}")
    
    # We assume the file channels are ordered A1..A32 then B1..B32
    # We rename the first 64 channels in the file to the standard list.
    rename_dict = {}
    if len(raw.ch_names) >= 64:
        for i in range(64):
            original_name = raw.ch_names[i]
            # Only rename if it looks like a BioSemi name (contains 'A' or 'B' or '1-'/'2-')
            if any(x in original_name for x in ['A', 'B', '1-', '2-']):
                rename_dict[original_name] = standard_64_names[i]
    
    if rename_dict:
        print(f"   -> Translating {len(rename_dict)} BioSemi names to Standard 10-20 system...")
        try:
            raw.rename_channels(rename_dict)
            print(f"   -> Success! New First Name: {raw.ch_names[0]} (Should be Fp1)")
        except Exception as e:
            print(f"   -> Renaming Warning: {e}")

    # ======================================================
    # 2. LOAD BAD CHANNELS
    # ======================================================
    if subject_id and basic_path and player_num:
        import pandas as pd
        import os
        
        participants_file = f'{basic_path}/participants.tsv'
        if os.path.exists(participants_file):
            try:
                df = pd.read_csv(participants_file, sep='\t')
                subject_row = df[df['participant_id'] == subject_id]
                
                if not subject_row.empty:
                    col_name = f'player{player_num}_pre_processing_channels_fixed'
                    if col_name in subject_row.columns:
                        value = subject_row[col_name].values[0]
                        if pd.notna(value):
                            channels = [ch.strip() for ch in str(value).split(',') if ch.strip()]
                            raw.info['bads'] = channels
                            print(f"   Set bad channels: {channels}")
            except Exception as e:
                print(f"   Error loading bad channels: {e}")
    # =========================================================================
    # 3. APPLY MONTAGE
    # =========================================================================
    try:
        # Define EEG channels (First 64 are now standard names)
        # We ensure anything named like "Fp1" is treated as EEG.
        # We filter 'valid_map' to only include channels actually in raw.ch_names,
        # so we don't need the 'on_missing' argument anymore.
        valid_map = {ch: 'eeg' for ch in raw.ch_names if ch in standard_64_names}
        
        # FIX: Removed on_missing='ignore'
        if valid_map:
            raw.set_channel_types(valid_map)

        # Apply the standard 10-20 map
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore') 
        print("   -> 10-20 System Montage applied.")
        
    except Exception as e:
        print(f"   -> ❌ WARNING: Could not set montage. ICA Topologies will NOT work. Error: {e}")

    # =========================================================================
    # 4. INTERPOLATION
    # =========================================================================
    try:
        # Use only EEG channels for std calculation
        picks = mne.pick_types(raw.info, eeg=True)
        data = raw.get_data(picks=picks)
        
        if data.size > 0:
            stds = np.std(data, axis=1)
            median_std = np.median(stds[stds > 0]) if np.any(stds > 0) else 1.0
            
            # Find bad indices relative to the picks
            bad_indices_local = [i for i, std in enumerate(stds) if std == 0 or std > 15 * median_std]
            # Convert local pick indices to channel names
            bad_names = [raw.ch_names[picks[i]] for i in bad_indices_local]
            
        if raw.info['bads']:
            print(f"   Interpolating {len(raw.info['bads'])} bad channels...")
            raw.interpolate_bads(reset_bads=True, verbose=False)
            
    except Exception as e:
        print(f"   Montage/Interpolation error: {e}")


    # =========================================================================
    # 5. CAR & RESAMPLE
    # =========================================================================
    print("   -> Applying CAR...")
    raw.set_eeg_reference('average', projection=False, verbose=False)
    
    if raw.info['sfreq'] != target_rate:
        print(f"   -> Resampling to {target_rate}Hz...")
        raw.resample(target_rate, npad="auto")
        
    print("="*50 + "\n")
    return raw

def visualize_clean_data(raw, title="Clean Data"):
    print(f"--- Visualizing Cleaned {title} ---")
    try:
        raw.compute_psd(fmax=60).plot(average=True, picks='all', exclude='bads')
        plt.title(f"{title} - Frequency Spectrum")
        plt.show(block=False)
    except Exception as e:
        print(f"Could not plot PSD: {e}")

def visualize_ica_components(raw, n_components=15):
    print(f"--- [step2_preprocessing] Calculating ICA ({n_components} comps) ---")
    
    # 1. FINAL SAFETY CHECK
    if raw.info['dig'] is None or len(raw.info['dig']) == 0:
        print("❌ SKIPPING ICA PLOT: No digitization points found.")
        print("   -> Renaming failed. Check raw.ch_names output.")
        return None

    # 2. Filter & Fit
    try:
        raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica = ICA(n_components=n_components, method='fastica', random_state=97)
        ica.fit(raw_ica, verbose=False)
        
        print("   -> Plotting ICA Topomaps...")
        # Plotting
        ica.plot_components(show=False) 
        plt.suptitle("Independent Components", y=1.02)
        plt.show(block=False)
        return ica
        
    except Exception as e:
        print(f"❌ ICA ERROR: {e}")
        return None