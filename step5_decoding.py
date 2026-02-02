import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# =================================================================
# HELPER: Generate Pseudo-Trials (Internal Use)
# =================================================================
def create_pseudo_trials_subset(X, y, n_average=4, n_repeats=20, random_seed=None):
    """
    Creates pseudo-trials from a specific SUBSET of data.
    """
    rng = np.random.RandomState(random_seed)
    unique_classes = np.unique(y)
    
    X_pseudo = []
    y_pseudo = []
    
    for cls in unique_classes:
        # Get indices of all available trials for this class in this subset
        cls_indices = np.where(y == cls)[0]
        
        # If we don't have enough trials to average, skip
        if len(cls_indices) < n_average:
            continue
            
        for _ in range(n_repeats):
            # Randomly select 'n_average' trials
            choices = rng.choice(cls_indices, size=n_average, replace=False)
            
            # Average them to create ONE pseudo-trial
            X_pseudo.append(np.mean(X[choices], axis=0))
            y_pseudo.append(cls)
            
    if len(X_pseudo) == 0:
        return np.array([]), np.array([])
        
    return np.array(X_pseudo), np.array(y_pseudo)

# =================================================================
# MAIN DECODING FUNCTION
# =================================================================
def run_svm_decoding_with_pseudotrials(X_raw, y_raw, n_folds=10, n_average=4, n_repeats=20):
    """
    Performs Stratified Cross-Validation with Pseudo-Trials generated INSIDE the loop.
    
    Parameters:
    - X_raw: The BINNED single trials (n_trials, n_channels, n_time_bins)
    - y_raw: The labels for single trials
    """
    # 1. Setup
    n_trials, n_channels, n_time_bins = X_raw.shape
    
    # Use Linear SVM with balanced class weights
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, class_weight='balanced'))
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = np.zeros((n_folds, n_time_bins))
    
    print(f"   Starting {n_folds}-fold CV on {n_trials} real trials...")
    
    # 2. Cross-Validation Loop
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_raw)):
        
        # A. Split the REAL trials
        X_train_real, y_train_real = X_raw[train_idx], y_raw[train_idx]
        X_test_real, y_test_real = X_raw[test_idx], y_raw[test_idx]
        
        # B. Generate Pseudo-Trials for TRAINING
        X_train_pseudo, y_train_pseudo = create_pseudo_trials_subset(
            X_train_real, y_train_real, 
            n_average=n_average, n_repeats=n_repeats, 
            random_seed=fold_idx # Different seed per fold
        )
        
        # C. Generate Pseudo-Trials for TESTING
        # Note: We apply the same SNR boost (averaging) to the test set
        X_test_pseudo, y_test_pseudo = create_pseudo_trials_subset(
            X_test_real, y_test_real, 
            n_average=n_average, n_repeats=n_repeats, 
            random_seed=fold_idx + 100 
        )
        
        # Safety check
        if len(X_train_pseudo) == 0 or len(X_test_pseudo) == 0:
            continue

        # D. Decode Time-by-Time
        for t in range(n_time_bins):
            # Extract data for this time bin (Trials x Channels)
            X_tr = X_train_pseudo[:, :, t]
            X_te = X_test_pseudo[:, :, t]
            
            # Train and Score
            clf.fit(X_tr, y_train_pseudo)
            fold_accuracies[fold_idx, t] = clf.score(X_te, y_test_pseudo)
            
    # 3. Aggregate Results
    mean_scores = np.mean(fold_accuracies, axis=0) * 100
    std_scores = np.std(fold_accuracies, axis=0) * 100
    times = np.arange(n_time_bins) # Dummy time axis
    
    print(f"   -> Average Accuracy: {np.mean(mean_scores):.2f}%")
    
    return times, mean_scores, std_scores, fold_accuracies