import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import SlidingEstimator, cross_val_multiscore

def run_svm_decoding(data, custom_labels=None):
    """
    Runs SVM decoding.
    - data: Can be MNE Epochs object OR a NumPy array (Trials x Channels x Time)
    - custom_labels: Array of labels (1, 2, 3) matching the trials. 
                     REQUIRED if data is a NumPy array.
    """
    print("--- [step5_decoding] Running SVM Decoding ---")
    
    # 1. HANDLE INPUT TYPE
    # If it's an MNE Epochs object, extract data and labels automatically
    if hasattr(data, 'get_data'):
        X = data.get_data()
        if custom_labels is None:
            y = data.events[:, -1]
        else:
            y = custom_labels
        # Get time points for plotting
        times = data.times
        
    # If it's a NumPy Array (from Step 4), use it directly
    else:
        X = data # (Trials, Channels, Bins)
        if custom_labels is None:
            raise ValueError("❌ Error: When passing a NumPy array (from Step 4), you MUST provide 'custom_labels'.")
        y = custom_labels
        # Create a dummy time axis (0, 1, 2...) since real time is lost in binning
        times = np.arange(X.shape[2])

    # 2. SETUP SVM
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    
    # 3. RUN SLIDING ESTIMATOR
    # n_jobs=-1 uses all CPU cores for speed
    print(f"   Data Shape: {X.shape}, Labels Shape: {y.shape}")
    slider = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=False)
    
    # Run 5-fold Cross-Validation
    try:
        scores = cross_val_multiscore(slider, X, y, cv=5, n_jobs=-1)
        # Average across the 5 folds -> Result: (Time_Points,)
        mean_scores = np.mean(scores, axis=0) * 100
        return times, mean_scores
        
    except Exception as e:
        print(f"❌ SVM Failed: {e}")
        return None, None