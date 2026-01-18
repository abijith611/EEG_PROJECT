import numpy as np

def create_pseudo_trials(epochs_tuple):
    """
    Takes the 3-phase tuple (Decision, Response, Feedback),
    bins them into 250ms chunks, and stitches them into a single 
    20-bin time course (0 to 5000ms).
    """
    print("--- [step4_pseudotrials] Binning and Stitching Phases ---")
    
    # 1. UNPACK THE TUPLE FROM STEP 3
    ep_dec, ep_resp, ep_feed = epochs_tuple
    
    # Helper function to bin a single epoch object
    def bin_data(epochs, tmax, bin_size=0.25):
        # Crop 0 to tmax (removing the negative baseline)
        # We assume the data is already baseline-corrected in Step 3
        data = epochs.copy().crop(tmin=0, tmax=tmax, include_tmax=False).get_data() 
        # Shape: (n_trials, n_channels, n_times)
        
        n_trials, n_ch, n_samples = data.shape
        sfreq = epochs.info['sfreq']
        samples_per_bin = int(bin_size * sfreq)
        n_bins = int(tmax / bin_size)
        
        # Truncate to exact number of bins (drop extra samples)
        limit = n_bins * samples_per_bin
        data = data[:, :, :limit]
        
        # Reshape to (Trials, Channels, Bins, SamplesPerBin) and Average
        # This creates the "Pseudo-trial"
        binned = data.reshape(n_trials, n_ch, n_bins, samples_per_bin).mean(axis=3)
        return binned

    # 2. BIN EACH PHASE SEPARATELY
    # Decision: 0-2.0s -> 8 Bins
    print("   -> Binning Decision Phase...")
    b_dec = bin_data(ep_dec, tmax=2.0)
    
    # Response: 0-2.0s -> 8 Bins
    print("   -> Binning Response Phase...")
    b_resp = bin_data(ep_resp, tmax=2.0)
    
    # Feedback: 0-1.0s -> 4 Bins
    print("   -> Binning Feedback Phase...")
    b_feed = bin_data(ep_feed, tmax=1.0)
    
    # 3. STITCH THEM TOGETHER
    # Concatenate along time axis (axis 2)
    # 8 + 8 + 4 = 20 Bins Total
    full_binned = np.concatenate([b_dec, b_resp, b_feed], axis=2)
    
    print(f"   -> Final Data Shape: {full_binned.shape} (Trials, Channels, 20 Bins)")
    return full_binned