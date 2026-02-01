import numpy as np
import random
import step4_pseudotrials  # Reuse the pseudo-trial creation logic
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import SlidingEstimator, cross_val_multiscore

# step5_decoding.py - UPDATE EXISTING FUNCTION

def run_svm_decoding_with_pseudotrials(X_pseudo, y_pseudo, n_folds=10):
    """
    UPDATED: Now includes pseudo-trials like MATLAB paper
    
    Parameters:
    - epochs_binned: (n_trials, n_channels, n_time_bins)
    - labels: (n_trials,) with values 1, 2, or 3
    - n_folds: 10-fold CV (paper uses 10)
    - n_average: Average 4 trials (paper uses 4)
    - n_repeats: 20 repeats per class (paper uses 20)
    """
    print("\n" + "="*60)
    print("   SVM DECODING WITH PSEUDO-TRIALS (Paper's Method)")
    print("="*60)
    
    # Step 1: Create pseudo-trials (MATLAB's method)
    
    
    # Step 2: Setup SVM (Your novelty - paper uses LDA)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=1.0, random_state=42)
    )
    
    # Step 3: YOUR NOVELTY - Stratified 10-fold CV (paper uses custom chunks)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results for each time bin
    n_time_bins = X_pseudo.shape[2]
    fold_accuracies = np.zeros((n_folds, n_time_bins))
    
    print(f"\n  Running {n_folds}-fold stratified cross-validation...")
    print(f"  Pseudo-trials shape: {X_pseudo.shape}")
    
    # Step 4: Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_pseudo, y_pseudo)):
        X_train, X_test = X_pseudo[train_idx], X_pseudo[test_idx]
        y_train, y_test = y_pseudo[train_idx], y_pseudo[test_idx]
        
        # Decode each time bin separately
        for time_bin in range(n_time_bins):
            svm_pipeline.fit(X_train[:, :, time_bin], y_train)
            accuracy = svm_pipeline.score(X_test[:, :, time_bin], y_test)
            fold_accuracies[fold_idx, time_bin] = accuracy
        
        if (fold_idx + 1) % 2 == 0:  # Print every 2 folds
            print(f"    Fold {fold_idx+1}/{n_folds} complete")
    
    # Calculate mean and standard error
    mean_accuracy = np.mean(fold_accuracies, axis=0) * 100  # Convert to percentage
    std_accuracy = np.std(fold_accuracies, axis=0) * 100
    times = np.arange(n_time_bins)  # Bin indices (0-19)
    
    # Print summary
    print(f"\n  Decoding complete!")
    print(f"  Mean accuracy: {np.mean(mean_accuracy):.2f}% ± {np.mean(std_accuracy):.2f}%")
    print(f"  Peak accuracy: {np.max(mean_accuracy):.2f}% at bin {np.argmax(mean_accuracy)}")
    
    return times, mean_accuracy, std_accuracy, fold_accuracies

# Keep your old function for comparison
def run_svm_decoding_original(data, custom_labels=None):
    """
    Your original function - keep for debugging
    """
    print("--- Running original SVM (no pseudo-trials) ---")
    
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