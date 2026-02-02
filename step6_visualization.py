import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_paper_comparison(time_axis, data_dict, title, chance_level=33.33):
    """
    Creates paper-style plots with phase shading.
    
    Parameters:
    - time_axis: Array of time points (seconds)
    - data_dict: Dictionary {'ConditionLabel': list_of_arrays}
    - title: Plot title
    - chance_level: Chance level for classification (default 33.33% for 3-class)
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Phase shading (paper style)
    phases = [
        ('Decision', 0, 2, '#FFA500', 0.15),  # Orange
        ('Response', 2, 4, '#2E8B57', 0.15),   # Green
        ('Feedback', 4, 5, '#9370DB', 0.15)    # Purple
    ]
    
    for phase_name, start, end, color, alpha in phases:
        ax.axvspan(start, end, alpha=alpha, color=color)
        ax.text((start+end)/2, 78, phase_name, ha='center', 
                fontsize=11, fontweight='bold', color='dimgray', alpha=0.8)
    
    # Colors for different conditions
    colors = {
        'Winner': '#1E88E5',      # Blue
        'Loser': '#D81B60',       # Pink/Red
        'Player 1': '#1E88E5',    # Blue
        'Player 2': '#D81B60',    # Pink/Red
        'Own Current': '#004D40', # Teal
        'Own Previous': '#5E35B1' # Purple
    }
    
    # Plot each condition
    for label, matrix_list in data_dict.items():
        if len(matrix_list) == 0:
            continue
        
        # Convert to array
        matrix = np.array(matrix_list)
        
        # Calculate mean and SEM
        mean_scores = np.mean(matrix, axis=0)
        sem_scores = stats.sem(matrix, axis=0)
        
        # Get color
        color = colors.get(label, '#000000')  # Default black
        
        # Plot
        ax.plot(time_axis, mean_scores, color=color, 
                linewidth=2.5, label=f"{label} (N={len(matrix)})")
        ax.fill_between(time_axis, 
                        mean_scores - sem_scores,
                        mean_scores + sem_scores,
                        color=color, alpha=0.2)
    
    # Chance level
    ax.axhline(y=chance_level, color='black', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Chance')
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time from Decision Onset (s)", fontsize=12)
    ax.set_ylabel("Decoding Accuracy (%)", fontsize=12)
    ax.set_ylim(25, 80)
    ax.set_xlim(-0.2, 5.2)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    return fig, ax

def plot_single_decoding(time_axis, mean_scores, sem_scores, title):
    """
    Simple plot for single decoding results.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Phase shading
    phases = [
        ('Decision', 0, 2, '#FFA500', 0.15),
        ('Response', 2, 4, '#2E8B57', 0.15),
        ('Feedback', 4, 5, '#9370DB', 0.15)
    ]
    
    for phase_name, start, end, color, alpha in phases:
        ax.axvspan(start, end, alpha=alpha, color=color)
    
    # Plot
    ax.plot(time_axis, mean_scores, color='#1E88E5', linewidth=2)
    ax.fill_between(time_axis, mean_scores - sem_scores, 
                    mean_scores + sem_scores, color='#1E88E5', alpha=0.3)
    
    # Chance level
    ax.axhline(y=33.33, color='black', linestyle='--', alpha=0.7)
    
    # Labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Decoding Accuracy (%)", fontsize=12)
    ax.set_ylim(25, 80)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax