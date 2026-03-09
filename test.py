import mne
import h5py
import numpy as np

# --- 1. Define file paths ---
# Adjust these to point to your specific pair 01, player 1 files
python_file = 'project/ds006761/derivatives/pair-34_player-2_task-RPS-epo.fif'
matlab_file = 'ppmatlab34.mat' 

def calculate_rms_error(data1, data2):
    """Calculate the Root Mean Square error between two arrays."""
    return np.sqrt(np.mean((data1 - data2)**2))

print("Loading Python data...")
epochs = mne.read_epochs(python_file, preload=True, verbose=False)
py_data = epochs.get_data()  # Expected shape: (480, 64, 1331)
py_chans = epochs.ch_names
# Convert Python data from Volts to microVolts for an easier comparison
py_data_uv = py_data * 1e6 

print("Loading MATLAB data...")
with h5py.File(matlab_file, 'r') as f:
    # Get references to the trials in the cell array
    trial_refs = f['eeg_data']['trial'][:]
    
    mat_trials = []
    # Loop through references, extract, and transpose (HDF5 loads MATLAB arrays transposed)
    for ref in trial_refs.flatten():
        trial_data = f[ref][:].T  # Back to (64, 1332)
        mat_trials.append(trial_data)
        
    mat_data = np.stack(mat_trials) # Expected shape: (480, 64, 1332)

# Extract MATLAB labels to ensure we match the right indices
with h5py.File(matlab_file, 'r') as f:
    label_refs = f['eeg_data']['label'][:]
    mat_chans = [''.join(chr(c[0]) for c in f[ref][:]) for ref in label_refs.flatten()]

print(f"\nPython shape: {py_data_uv.shape}")
print(f"MATLAB shape: {mat_data.shape}")

# --- 2. Align and Compare ---
# Truncate MATLAB data to match Python's 1331 time points
min_samples = min(py_data_uv.shape[2], mat_data.shape[2])
py_data_uv = py_data_uv[:, :, :min_samples]
mat_data = mat_data[:, :, :min_samples]

print(f"\nTruncated to {min_samples} samples for comparison.")
print("-" * 40)

# Let's check a few specific channels
channels_to_check = ['Fp1', 'P6', 'Cz', 'Oz']

for chan in channels_to_check:
    if chan in py_chans and chan in mat_chans:
        py_idx = py_chans.index(chan)
        mat_idx = mat_chans.index(chan)
        
        py_chan_data = py_data_uv[:, py_idx, :]
        mat_chan_data = mat_data[:, mat_idx, :]
        
        rms_diff = calculate_rms_error(py_chan_data, mat_chan_data)
        
        print(f"Channel {chan}:")
        print(f"  Python mean: {np.mean(py_chan_data):.2f} µV")
        print(f"  MATLAB mean: {np.mean(mat_chan_data):.2f} µV")
        print(f"  RMS Difference: {rms_diff:.2f} µV\n")
    else:
        print(f"Channel {chan} not found in both datasets.\n")