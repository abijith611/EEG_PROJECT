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
        times, scores = step5_decoding.run_svm_decoding_original(X_sub, custom_labels=y_sub)
        
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
    
    # Chance Level
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    
    # --- PHASE SEPARATORS ---
    # Phase 1 (Decision) ends at 2.0s
    plt.axvline(2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.text(1.0, 39, "DECISION", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    
    # Phase 2 (Response) ends at 4.0s (2.0 + 2.0)
    plt.axvline(4.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.text(3.0, 39, "RESPONSE", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    
    # Phase 3 (Feedback) ends at 5.0s
    plt.text(4.5, 39, "FEEDBACK", ha='center', fontsize=12, fontweight='bold', alpha=0.5)

    plt.title(f"Comparative Decoding: Player {player_num}")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding Accuracy (%)")
    plt.ylim(30, 40) # Zoom in a bit
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show(block=False)


def plot_grand_average_comparison(group_results, group_name):
    """
    Plot comparative decoding results for winners vs losers
    Similar to Figure 3 in the paper
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{group_name} Group - Comparative Decoding Analysis', 
                 fontsize=16, fontweight='bold')
    
    keys = ['Own Current', 'Opponent Current', 'Own Previous', 'Opponent Previous']
    titles = ['A) Player Response', 'B) Opponent Response', 
              'C) Player Previous', 'D) Opponent Previous']
    
    time_axis = np.linspace(0, 5.0, 20)  # 20 time bins
    
    # Define phase boundaries
    phases = [('Decision', 0, 2, 'orange', 0.1),
              ('Response', 2, 4, 'red', 0.1),
              ('Feedback', 4, 5, 'purple', 0.1)]
    
    for idx, (ax, key, title) in enumerate(zip(axes.flatten(), keys, titles)):
        # Add phase shading
        for phase_name, start, end, color, alpha in phases:
            ax.axvspan(start, end, alpha=alpha, color=color)
        
        # Plot each condition if data exists
        if key in group_results and group_results[key]:
            all_scores = np.array(group_results[key])
            
            # Calculate mean and SEM
            mean_scores = np.mean(all_scores, axis=0)
            sem_scores = np.std(all_scores, axis=0) / np.sqrt(len(all_scores))
            
            # Plot with confidence interval
            ax.plot(time_axis, mean_scores, 'k-', linewidth=2, label=f'{key} (n={len(all_scores)})')
            ax.fill_between(time_axis, mean_scores - sem_scores, mean_scores + sem_scores, 
                           alpha=0.3, color='gray')
        
        # Add chance level and reference lines
        ax.axhline(y=33.33, color='blue', linestyle='--', linewidth=1.5, label='Chance')
        
        # Add phase boundary lines
        for boundary in [2, 4]:
            ax.axvline(x=boundary, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Set plot properties
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Decoding Accuracy (%)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.set_xlim([0, 5])
        ax.set_ylim([20, 60])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:  # Only show legend in first plot
            ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig