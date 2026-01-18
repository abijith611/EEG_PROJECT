import mne
import os

def load_and_split_data(subject_id, basic_path):
    """
    Reads BioSemi file and dynamically splits into P1 and P2 
    based on the total channel count.
    
    Layouts handled:
    1. Standard (143+ ch): P1(0-63) --Gap(7)-- P2(71-134)
    2. Compact (129 ch):   P1(0-63) --No Gap-- P2(64-127)
    """
    bdf_file_path = f'{basic_path}/{subject_id}/eeg/{subject_id}_task-RPS_eeg.bdf'
    print(f"--- [step1_loading] Reading {bdf_file_path} ---")
    
    # Check if file exists to prevent hard crash
    if not os.path.exists(bdf_file_path):
        print(f"❌ Error: File not found at {bdf_file_path}")
        return None, None
    
    # Read raw data
    raw = mne.io.read_raw_bdf(bdf_file_path, preload=True, verbose=False)
    n_chans = len(raw.ch_names)
    print(f"   -> Total Channels Detected: {n_chans}")
    
    # --- DYNAMIC INDEX ASSIGNMENT ---
    # Player 1 is always the first 64 channels
    p1_indices = list(range(0, 64))
    
    # Determine Player 2 indices based on file size
    if n_chans >= 135:
        # Scenario A: Standard Layout (Gap between players)
        # Indices 71 to 134 (64 channels)
        print("   -> Layout detected: STANDARD (Gap between players).")
        p2_indices = list(range(71, 135))
        
    elif n_chans >= 128:
        # Scenario B: Compact Layout (No gap)
        # Indices 64 to 127 (64 channels)
        print("   -> Layout detected: COMPACT (No gap between players).")
        p2_indices = list(range(64, 128))
        
    else:
        # Scenario C: Invalid/Single Player
        print(f"❌ Critical Error: Not enough channels for 2 players ({n_chans} found).")
        return None, None

    # --- SPLIT DATA ---
    print("   -> Splitting data into Player 1 and Player 2...")
    
    try:
        raw_p1 = raw.copy().pick(p1_indices)
        raw_p2 = raw.copy().pick(p2_indices)
        
        print(f"   -> Player 1 Channels: {len(raw_p1.ch_names)}")
        print(f"   -> Player 2 Channels: {len(raw_p2.ch_names)}")
        
        return raw_p1, raw_p2
        
    except Exception as e:
        print(f"❌ Error during splitting: {e}")
        return None, None

def visualize_raw_data(raw, title="Raw Data"):
    # --- CHECK CHANNEL COUNT ---
    n_channels = len(raw.ch_names)
    print(f"Total Channels Detected: {n_channels}")
    print(f"Channel Names: {raw.ch_names[:5]} ... {raw.ch_names[-5:]}")

    # --- PLOT 1: Channels 0 - 64 ---
    print("Plotting Channels...")
    raw.plot(duration=5, n_channels=64, 
        scalings=dict(eeg=50e-6),
        remove_dc=True, title="Channels 0-64", show=True, 
        block=True)

    # --- PLOT 2: Full PSD ---
    """Plots raw PSD."""
    print(f"--- Visualizing {title} ---")
    # We exclude 'bads' and the Stim channel if present
    raw.compute_psd(fmax=60).plot(average=True, picks='all', exclude='bads')