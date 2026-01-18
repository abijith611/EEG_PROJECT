import mne
import os

def load_and_split_data(subject_id, basic_path):
    """
    Reads 143-channel BioSemi file and splits based on interleaved structure:
    - P1 EEG (0-63)    | P1 Physio (64-70)
    - P2 EEG (71-134)  | P2 Physio (135-141)
    - Status (142)
    """
    bdf_file_path = f'{basic_path}/{subject_id}/eeg/{subject_id}_task-RPS_eeg.bdf'
    print(f"--- [step1_loading] Reading {bdf_file_path} ---")
    
    # Read raw data
    raw = mne.io.read_raw_bdf(bdf_file_path, preload=True, verbose=False)
    
    n_chans = len(raw.ch_names)
    print(f"Total Channels Detected: {n_chans}")
    
    if n_chans < 143:
        print("⚠️ Warning: File has fewer than 143 channels. Check layout!")

    # --- DEFINE SPLITS (Python Slicing) ---
    # Player 1 EEG: 0 to 63
    p1_indices = list(range(0, 64))
    
    # Player 1 Physio: 64 to 70 (GSR, Resp, etc.)
    p1_physio_indices = list(range(64, 71))
    
    # Player 2 EEG: 71 to 134
    p2_indices = list(range(71, 135))
    
    # Player 2 Physio: 135 to 141
    p2_physio_indices = list(range(135, 142))
    
    # Status Channel: 142
    status_index = [142]

    # --- CREATE SEPARATE OBJECTS ---
    print("Splitting data into Player 1 and Player 2...")
    
    # Create Player 1 Object (EEG only for decoding)
    raw_p1 = raw.copy().pick(p1_indices)
    
    # Create Player 2 Object (EEG only for decoding)
    # Note: We must pick by index because channel names might be generic (e.g., A1, A2)
    raw_p2 = raw.copy().pick(p2_indices)
    
    # (Optional) If you ever need to analyze stress (GSR), you can extract these:
    # raw_p1_physio = raw.copy().pick(p1_physio_indices)
    # raw_p2_physio = raw.copy().pick(p2_physio_indices)
    
    print(f"Player 1 EEG Channels: {len(raw_p1.ch_names)}")
    print(f"Player 2 EEG Channels: {len(raw_p2.ch_names)}")
    
    return raw_p1, raw_p2

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