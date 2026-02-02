import numpy as np

def bin_data(epochs, tmax=2.0, bin_size=0.25):
    """
    Downsamples the data by averaging time points into bins.
    Uses 250ms bins to match the paper's methodology.
    
    Parameters:
    - epochs: MNE Epochs object
    - tmax: End time of the phase (e.g., 2.0s for Decision/Response, 1.0s for Feedback)
    - bin_size: Size of each bin in seconds (0.25s = 250ms as in paper)
    
    Returns:
    - binned_data: Array of shape (n_trials, n_channels, n_bins)
    """
    # Crop to the relevant window
    data = epochs.copy().crop(tmin=0, tmax=tmax, include_tmax=False).get_data() 
    # Shape: (n_trials, n_channels, n_samples)
    
    n_trials, n_ch, n_samples = data.shape
    sfreq = epochs.info['sfreq']
    
    # Calculate samples per bin
    samples_per_bin = int(bin_size * sfreq)
    n_bins = int(tmax / bin_size)
    
    # Calculate exact number of samples needed
    limit = n_bins * samples_per_bin
    
    if limit > n_samples:
        print(f"Warning: Not enough samples for {n_bins} bins. Truncating.")
        limit = (n_samples // samples_per_bin) * samples_per_bin
        n_bins = limit // samples_per_bin
    
    # Truncate data to fit exact bins
    data = data[:, :, :limit]
    
    # Reshape and Average to Bin
    binned_data = data.reshape(n_trials, n_ch, n_bins, samples_per_bin).mean(axis=3)
    
    return binned_data