
import numpy as np

def create_pseudo_trials_by_averaging(epochs_binned, labels, n_average=4, n_repeats=20, random_seed=42):
    """
    EXACTLY replicates MATLAB's: 
    ds_sel = cosmo_average_samples(ds_sel,'count',4,'repeats',20,'seed',1)
    
    Parameters:
    - epochs_binned: NumPy array (n_real_trials, n_channels, n_time_bins)
    - labels: Array (n_real_trials,) with values 1, 2, or 3
    - n_average: How many real trials to average (paper uses 4)
    - n_repeats: How many pseudo-trials to create per class (paper uses 20)
    - random_seed: For reproducibility (paper uses seed=1)
    
    Returns:
    - X_pseudo: Pseudo-trials (n_pseudo_trials, n_channels, n_time_bins)
    - y_pseudo: Labels (n_pseudo_trials,)
    """
    
    # Set random seed for reproducibility (like MATLAB's 'seed',1)
    np.random.seed(random_seed)
    
    print(f"  Creating pseudo-trials: average {n_average} trials, {n_repeats} repeats per class")
    print(f"  Original data: {epochs_binned.shape[0]} real trials")
    
    # Get unique classes (1=Rock, 2=Paper, 3=Scissors)
    unique_labels = np.unique(labels)
    print(f"  Classes found: {unique_labels}")
    
    # Store all pseudo-trials and labels
    all_pseudo_data = []
    all_pseudo_labels = []
    
    # For each class (Rock, Paper, Scissors)
    for class_label in unique_labels:
        # Find all real trials with this label
        class_indices = np.where(labels == class_label)[0]
        n_class_trials = len(class_indices)
        
        print(f"    Class {class_label}: {n_class_trials} real trials")
        
        # Check if we have enough trials
        if n_class_trials < n_average:
            print(f"    ⚠️ Warning: Not enough trials for class {class_label} "
                  f"({n_class_trials} < {n_average})")
            continue
        
        # Create n_repeats pseudo-trials for this class
        for repeat in range(n_repeats):
            # Randomly select n_average trials WITHOUT replacement
            # (This matches MATLAB's method)
            selected_indices = np.random.choice(class_indices, 
                                                size=n_average, 
                                                replace=False)
            
            # Average the selected trials
            # epochs_binned[selected_indices] shape: (n_average, channels, time_bins)
            # After mean(axis=0): (channels, time_bins)
            pseudo_trial = np.mean(epochs_binned[selected_indices], axis=0)
            
            all_pseudo_data.append(pseudo_trial)
            all_pseudo_labels.append(class_label)
    
    # Convert to NumPy arrays
    X_pseudo = np.array(all_pseudo_data)
    y_pseudo = np.array(all_pseudo_labels)
    
    print(f"  Created {X_pseudo.shape[0]} pseudo-trials")
    print(f"  Final distribution: Rock={sum(y_pseudo==1)}, "
          f"Paper={sum(y_pseudo==2)}, Scissors={sum(y_pseudo==3)}")
    
    return X_pseudo, y_pseudo

def bin_and_stitch_time_course(epochs_tuple):
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