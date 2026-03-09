import numpy as np
import pickle
import scipy.io as sio

# --- 1. Define file paths ---
python_file = 'project/ds006761/derivatives/pair-34_player-1_task-RPS_decoding.pkl' 
matlab_file = 'dmatlab34.mat' 

# --- 2. Load Python Data ---
with open(python_file, 'rb') as f:
    py_data = pickle.load(f)

py_decoding = py_data['decoding'] # This is a list of arrays

# --- 3. Load MATLAB Data ---
mat_data = sio.loadmat(matlab_file)
mat_acc_struct = mat_data['decoding_accuracy']

# --- 4. Compare Each Condition ---
print("=== DECODING ACCURACY COMPARISON ===")
print(f"Python contains {len(py_decoding)} conditions.")
print(f"MATLAB contains {mat_acc_struct.shape[1]} conditions.\n")

for i in range(mat_acc_struct.shape[1]):
    # Extract MATLAB 20-point array
    mat_item = mat_acc_struct[0, i]
    # In scipy.io, struct fields are accessed like this:
    mat_array = np.array(mat_item['samples'][0, 0]).flatten()
    
    # Extract matching Python array
    if i < len(py_decoding):
        py_array = np.array(py_decoding[i]).flatten()
        
        # Truncate to the minimum length just in case there's a 1-sample difference
        min_len = min(len(py_array), len(mat_array))
        py_arr_trunc = py_array[:min_len]
        mat_arr_trunc = mat_array[:min_len]
        
        # Calculate metrics
        mean_diff = np.mean(np.abs(py_arr_trunc - mat_arr_trunc))
        rms_diff = np.sqrt(np.mean((py_arr_trunc - mat_arr_trunc)**2))
        
        print(f"Condition {i+1}:")
        print(f"  Data Lengths: Python ({len(py_array)}), MATLAB ({len(mat_array)})")
        print(f"  Python Mean Acc: {np.mean(py_arr_trunc):.4f}")
        print(f"  MATLAB Mean Acc: {np.mean(mat_arr_trunc):.4f}")
        print(f"  Mean Difference: {mean_diff:.4f} (approx {mean_diff*100:.2f}%)")
        print(f"  RMS Difference:  {rms_diff:.4f}")
        print("-" * 40)
    else:
        print(f"Condition {i+1}: Found in MATLAB but missing in Python!")