import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def plot_decoding_with_confidence(time_axis, mean_scores, std_scores, title, chance_level=33.33):
    """
    Plot decoding accuracy over time with confidence intervals and phase shading
    Similar to Figure 2/3 in the paper
    
    Parameters:
    time_axis: array of time points
    mean_scores: mean decoding accuracy per time bin
    std_scores: standard deviation per time bin
    title: plot title
    chance_level: chance level for decoding (33.33% for 3 classes)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define phase boundaries and colors (as in paper)
    phases = [
        ('Decision', 0, 2, 'orange', 0.1),
        ('Response', 2, 4, 'red', 0.1),
        ('Feedback', 4, 5, 'purple', 0.1)
    ]
    
    # Add phase background shading
    for phase_name, start, end, color, alpha in phases:
        ax.axvspan(start, end, alpha=alpha, color=color, label=phase_name if phase_name == 'Decision' else "")
    
    # Plot mean decoding accuracy with confidence interval
    ax.plot(time_axis, mean_scores, 'k-', linewidth=2, label='Decoding Accuracy')
    ax.fill_between(time_axis, 
                    mean_scores - std_scores, 
                    mean_scores + std_scores, 
                    alpha=0.3, color='gray')
    
    # Add chance level line
    ax.axhline(y=chance_level, color='blue', linestyle='--', linewidth=1.5, 
               label=f'Chance ({chance_level:.1f}%)')
    
    # Add significance markers (simplified - paper uses Bayes Factors)
    # For bins where accuracy > chance_level + 2*std, add asterisk
    sig_threshold = chance_level + 2 * np.mean(std_scores)
    sig_indices = np.where(mean_scores > sig_threshold)[0]
    if len(sig_indices) > 0:
        ax.plot(time_axis[sig_indices], mean_scores[sig_indices], 'r*', 
                markersize=8, label='Above Chance')
    
    # Set plot properties
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Decoding Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 5])
    ax.set_ylim([25, 75])  # Reasonable range for RPS decoding
    
    # Add phase labels at the top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([1, 3, 4.5])  # Middle of each phase
    ax2.set_xticklabels(['Decision', 'Response', 'Feedback'], fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('none')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig, ax

def plot_fold_variability(fold_accuracies, time_axis, title):
    """
    Plot variability across cross-validation folds
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each fold as semi-transparent line
    for i, fold_acc in enumerate(fold_accuracies):
        ax.plot(time_axis, fold_acc, alpha=0.3, linewidth=0.5)
    
    # Plot mean across folds
    mean_acc = np.mean(fold_accuracies, axis=0)
    std_acc = np.std(fold_accuracies, axis=0)
    
    ax.plot(time_axis, mean_acc, 'k-', linewidth=2, label='Mean')
    ax.fill_between(time_axis, mean_acc - std_acc, mean_acc + std_acc, 
                    alpha=0.3, color='gray', label='±1 SD')
    
    ax.axhline(y=33.33, color='blue', linestyle='--', linewidth=1.5, label='Chance')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{title} - Fold Variability', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 5])
    
    plt.tight_layout()
    return fig

def plot_grand_average_with_stats(group_results, player_num):
    """
    Plot grand average across subjects with statistical information
    Similar to paper's group-level analysis
    """
    if not group_results[player_num]:
        return None
    
    all_means = []
    for result in group_results[player_num]:
        if 'mean_scores' in result:
            all_means.append(result['mean_scores'])
    
    if not all_means:
        return None
    
    mean_matrix = np.array(all_means)
    n_subjects = mean_matrix.shape[0]
    
    # Calculate grand mean and SEM
    grand_mean = np.mean(mean_matrix, axis=0)
    sem = np.std(mean_matrix, axis=0) / np.sqrt(n_subjects)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Grand average with SEM
    time_axis = np.linspace(0, 5.0, len(grand_mean))
    
    # Add phase shading
    phases = [('Decision', 0, 2, 'orange', 0.1),
              ('Response', 2, 4, 'red', 0.1),
              ('Feedback', 4, 5, 'purple', 0.1)]
    
    for phase_name, start, end, color, alpha in phases:
        ax1.axvspan(start, end, alpha=alpha, color=color)
    
    ax1.plot(time_axis, grand_mean, 'k-', linewidth=2, label=f'Grand Mean (n={n_subjects})')
    ax1.fill_between(time_axis, grand_mean - sem, grand_mean + sem, 
                     alpha=0.3, color='gray', label='SEM')
    
    ax1.axhline(y=33.33, color='blue', linestyle='--', linewidth=1.5, label='Chance')
    ax1.set_ylabel('Decoding Accuracy (%)', fontsize=12)
    ax1.set_title(f'Player {player_num} - Grand Average Decoding', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 5])
    ax1.set_ylim([25, 70])
    
    # Plot 2: Subject variability heatmap
    im = ax2.imshow(mean_matrix, aspect='auto', cmap='viridis',
                   extent=[0, 5, 0, n_subjects], interpolation='nearest')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Subject Index', fontsize=12)
    ax2.set_title('Individual Subject Decoding Patterns', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Accuracy (%)')
    
    # Add phase dividers
    for phase_boundary in [2, 4]:
        ax2.axvline(x=phase_boundary, color='white', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    return fig