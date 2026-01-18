import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import step5_decoding  # Reuse the decoding logic

def run_comparative_analysis(epochs_binned, full_df, player_num):
    print(f"--- [step7_comparison] Analyzing Player {player_num} ---")
    
    # 1. Define Columns in the Events Table
    # If Player 1: Own=player1_resp, Opp=player2_resp
    p_own = f'player{player_num}_resp'
    p_opp = f'player{2 if player_num==1 else 1}_resp'
    
    # 2. Extract Label Arrays from DataFrame
    # These must match the number of epochs exactly (e.g., 80 trials)
    y_own_curr = full_df[p_own].values
    y_opp_curr = full_df[p_opp].values
    y_own_prev = full_df[p_own].shift(1).values
    y_opp_prev = full_df[p_opp].shift(1).values
    
    # 3. Create a Dictionary of Tasks
    tasks = {
        'Own Current': y_own_curr,
        'Opponent Current': y_opp_curr,
        'Own Previous': y_own_prev,
        'Opponent Previous': y_opp_prev
    }
    
    results = {}
    
    # 4. Run Decoding for Each Task
    for task_name, y_labels in tasks.items():
        # A. Create a Mask for Valid Trials
        # We must remove NaNs (from shift) and ensure labels are 1, 2, or 3
        # valid_mask is a list of True/False
        valid_mask = ~np.isnan(y_labels) & np.isin(y_labels, [1, 2, 3])
        
        # B. Subset the Data (NumPy Slicing)
        # epochs_binned shape is (Trials, Channels, Bins)
        # We select only the rows where valid_mask is True
        X_sub = epochs_binned[valid_mask, :, :]
        y_sub = y_labels[valid_mask]
        
        # C. Decode
        # We pass the subsetted data and labels to Step 5
        # times comes back as bin indices (0..19), we'll fix visual later
        times, scores = step5_decoding.run_svm_decoding(X_sub, custom_labels=y_sub)
        
        results[task_name] = scores # Store just scores

    return results

def plot_comparisons(results, player_num):
    """
    Plots the 4 comparison lines on a 0-5 second axis.
    """
    if not results:
        print("No results to plot.")
        return

    # Create Time Axis: 20 bins mapped to 0-5 seconds
    # We assume all results have the same length (20 bins)
    n_bins = len(next(iter(results.values())))
    time_axis = np.linspace(0, 5.0, n_bins)
    
    plt.figure(figsize=(12, 6))
    
    # Define Colors/Styles
    styles = {
        'Own Current':      ('blue', '-', 2.5),
        'Opponent Current': ('orange', '-', 2.0),
        'Own Previous':     ('green', '--', 1.5),
        'Opponent Previous':('red', '--', 1.5)
    }
    
    for name, scores in results.items():
        if scores is not None:
            c, ls, lw = styles.get(name, ('black', '-', 1))
            plt.plot(time_axis, scores, label=name, color=c, linestyle=ls, linewidth=lw)
    
    # Reference Lines & Text
    plt.axhline(33.33, color='k', linestyle=':', alpha=0.5)
    
    # Vertical Phase Lines
    plt.axvline(2.0, color='gray', linestyle='-', alpha=0.3)
    plt.text(1.0, 95, "DECISION", ha='center', fontsize=10, alpha=0.5)
    
    plt.axvline(4.0, color='gray', linestyle='-', alpha=0.3)
    plt.text(3.0, 95, "RESPONSE", ha='center', fontsize=10, alpha=0.5)
    
    plt.text(4.5, 95, "FEEDBACK", ha='center', fontsize=10, alpha=0.5)

    plt.title(f"Comparative Decoding: Player {player_num}")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding Accuracy (%)")
    plt.ylim(20, 100) # Zoom in a bit
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)

# --- ADD THIS TO THE BOTTOM OF step7_comparison.py ---

def plot_grand_average_comparison(group_data, player_num):
    """
    Plots the grand average of the comparative analysis (4 lines) with SEM shading.
    group_data: Dictionary containing lists of scores for each condition.
                e.g., {'Own Current': [sub1_arr, sub2_arr...], ...}
    """
    print(f"--- Plotting Grand Average Comparison for Player {player_num} ---")
    
    # Check if we have data (look at the first key's list)
    first_key = next(iter(group_data))
    if not group_data[first_key]:
        print("   -> No data to plot.")
        return

    # Create Time Axis (Assume 20 bins = 5.0s)
    n_bins = len(group_data[first_key][0])
    time_axis = np.linspace(0, 5.0, n_bins)
    
    plt.figure(figsize=(12, 6))
    
    # Define Colors & Styles
    styles = {
        'Own Current':      ('blue', '-'),
        'Opponent Current': ('orange', '-'),
        'Own Previous':     ('green', '--'),
        'Opponent Previous':('red', '--')
    }
    
    # Plot each condition
    for name, score_list in group_data.items():
        if not score_list: continue
        
        # Convert list of arrays to Matrix -> (N_Subjects, N_Bins)
        mat = np.array(score_list)
        n_subs = mat.shape[0]
        
        # Calculate Mean and Standard Error
        mean_scores = np.mean(mat, axis=0)
        sem_scores = np.std(mat, axis=0) / np.sqrt(n_subs)
        
        color, ls = styles.get(name, ('black', '-'))
        
        # Plot Mean Line
        plt.plot(time_axis, mean_scores, label=f"{name}", color=color, linestyle=ls, linewidth=2)
        
        # Plot Shaded Error Region
        plt.fill_between(time_axis, mean_scores - sem_scores, mean_scores + sem_scores, 
                         color=color, alpha=0.15)

    # Reference Lines
    plt.axhline(33.33, color='k', linestyle=':', alpha=0.5)
    plt.axvline(2.0, color='gray', alpha=0.3)
    plt.axvline(4.0, color='gray', alpha=0.3)
    
    plt.title(f"Grand Average Comparative Decoding: Player {player_num} (N={n_subs})")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(25, 45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)