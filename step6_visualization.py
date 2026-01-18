import matplotlib.pyplot as plt
import numpy as np

def plot_paper_replication(times, scores, title):
    """
    Plots the stitched 20-bin time course (0-5000ms).
    Draws vertical lines to separate Decision, Response, and Feedback.
    """
    # Create a synthetic time axis for 20 bins (0 to 5 seconds)
    # 20 bins * 0.25s = 5.0s
    time_axis = np.linspace(0, 5.0, len(scores))
    
    plt.figure(figsize=(12, 6))
    
    # Plot Accuracy
    plt.plot(time_axis, scores, color='#2c3e50', linewidth=2, label='Decoding Accuracy')
    
    # Chance Level
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    
    # --- PHASE SEPARATORS ---
    # Phase 1 (Decision) ends at 2.0s
    plt.axvline(2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.text(1.0, 35, "DECISION", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    
    # Phase 2 (Response) ends at 4.0s (2.0 + 2.0)
    plt.axvline(4.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.text(3.0, 35, "RESPONSE", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    
    # Phase 3 (Feedback) ends at 5.0s
    plt.text(4.5, 35, "FEEDBACK", ha='center', fontsize=12, fontweight='bold', alpha=0.5)

    # Styling
    plt.title(f"{title} (Replicated 3-Phase Structure)")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100) # Keep 0-100 scale
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # Important: block=False keeps the code running!
    plt.show(block=False)

# --- ADD THIS TO step6_visualization.py ---

def plot_grand_average(times, all_subject_scores, player_num):
    """
    Plots the average decoding accuracy across multiple subjects.
    
    Parameters:
    - times: The time axis (0-5s array)
    - all_subject_scores: List of score arrays (e.g., [sub1_scores, sub2_scores...])
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert list to Matrix: (N_Subjects, N_TimePoints)
    score_matrix = np.array(all_subject_scores)
    n_subs = score_matrix.shape[0]
    
    if n_subs == 0:
        print(f"No data to plot for Player {player_num}")
        return

    # Calculate Mean and Standard Error (SEM)
    mean_scores = np.mean(score_matrix, axis=0)
    sem_scores = np.std(score_matrix, axis=0) / np.sqrt(n_subs)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Mean Line
    plt.plot(times, mean_scores, color='navy' if player_num == 1 else 'darkorange', 
             linewidth=2.5, label=f'Player {player_num} Mean (N={n_subs})')
    
    # Plot Shaded Error Region (Mean ± SEM)
    plt.fill_between(times, mean_scores - sem_scores, mean_scores + sem_scores, 
                     color='navy' if player_num == 1 else 'darkorange', alpha=0.2)
    
    # Reference Lines
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    plt.axvline(2.0, color='k', linestyle=':', alpha=0.5) # Decision End
    plt.axvline(4.0, color='k', linestyle=':', alpha=0.5) # Response End
    
    # Labels
    plt.text(1.0, 35, "DECISION", ha='center', fontsize=10, alpha=0.5)
    plt.text(3.0, 35, "RESPONSE", ha='center', fontsize=10, alpha=0.5)
    plt.text(4.5, 35, "FEEDBACK", ha='center', fontsize=10, alpha=0.5)
    
    plt.title(f"Grand Average Decoding Accuracy: Player {player_num}")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(20, 100) # Adjust based on your data range
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.show(block=False)