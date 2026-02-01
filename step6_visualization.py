import numpy as np
import matplotlib.pyplot as plt

def plot_decoding_with_confidence(times, mean_scores, std_scores, title="Decoding with Confidence"):
    """
    Plot decoding accuracy with confidence intervals from standard deviation.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot mean accuracy
    plt.plot(times, mean_scores, color='#2c3e50', linewidth=2, label='Mean Accuracy')
    
    # Plot confidence interval (mean ± std)
    plt.fill_between(times, mean_scores - std_scores, mean_scores + std_scores,
                     color='#2c3e50', alpha=0.2, label='± Std Dev')
    
    # Chance level
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    
    # Phase separators
    plt.axvline(2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.axvline(4.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Labels
    plt.text(1.0, 35, "DECISION", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    plt.text(3.0, 35, "RESPONSE", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    plt.text(4.5, 35, "FEEDBACK", ha='center', fontsize=12, fontweight='bold', alpha=0.5)
    
    plt.title(f"{title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding Accuracy (%)")
    plt.ylim(25, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.show(block=False)

def plot_fold_variability(fold_accuracies, times, title="Cross-Validation Fold Variability"):
    """
    Plot individual fold accuracies to show variability across folds.
    """
    plt.figure(figsize=(12, 6))
    
    n_folds = fold_accuracies.shape[0]
    
    # Plot each fold
    for fold in range(n_folds):
        plt.plot(times, fold_accuracies[fold, :] * 100, 
                alpha=0.3, linewidth=0.8, color='gray')
    
    # Plot mean across folds
    mean_across_folds = np.mean(fold_accuracies, axis=0) * 100
    std_across_folds = np.std(fold_accuracies, axis=0) * 100
    
    plt.plot(times, mean_across_folds, color='navy', linewidth=2, 
             label=f'Mean (N={n_folds} folds)')
    
    # Plot chance level
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    
    # Phase separators
    plt.axvline(2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.axvline(4.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.title(f"{title} - Individual Fold Performance")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(20, 100)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show(block=False)
    
    # Print statistics
    print(f"Fold Statistics:")
    print(f"  Mean accuracy across all folds: {np.mean(mean_across_folds):.2f}%")
    print(f"  Std across folds (temporal mean): {np.mean(std_across_folds):.2f}%")
    print(f"  Max variability at bin {np.argmax(std_across_folds)}: {np.max(std_across_folds):.2f}%")

def plot_grand_average_with_stats(group_results, player_num):
    """
    Enhanced grand average plot with statistics across subjects.
    """
    if not group_results[player_num]:
        print(f"No data to plot for Player {player_num}")
        return
    
    # Extract data
    all_mean_scores = []
    all_std_scores = []
    
    for result in group_results[player_num]:
        all_mean_scores.append(result['mean_scores'])
        all_std_scores.append(result['std_scores'])
    
    # Convert to arrays
    mean_matrix = np.array(all_mean_scores)  # (N_subjects, 20_bins)
    std_matrix = np.array(all_std_scores)    # (N_subjects, 20_bins)
    
    n_subjects = mean_matrix.shape[0]
    
    # Calculate grand statistics
    grand_mean = np.mean(mean_matrix, axis=0)
    grand_std = np.std(mean_matrix, axis=0)  # Between-subject variability
    sem = grand_std / np.sqrt(n_subjects)    # Standard error of the mean
    
    # Create time axis
    time_axis = np.linspace(0, 5.0, len(grand_mean))
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot individual subjects (thin lines)
    for sub_idx in range(n_subjects):
        plt.plot(time_axis, mean_matrix[sub_idx, :], 
                alpha=0.2, linewidth=0.5, color='gray')
    
    # Plot grand mean
    plt.plot(time_axis, grand_mean, color='navy', linewidth=3, 
             label=f'Grand Mean (N={n_subjects})')
    
    # Plot confidence interval (SEM)
    plt.fill_between(time_axis, grand_mean - sem, grand_mean + sem,
                     color='navy', alpha=0.2, label='± SEM')
    
    # Chance level
    plt.axhline(33.33, color='r', linestyle='--', label='Chance (33%)')
    
    # Phase separators
    plt.axvline(2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.axvline(4.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.title(f"Grand Average Decoding Accuracy: Player {player_num}")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(20, 100)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.show(block=False)
    
    # Print statistics
    print(f"\nGrand Average Statistics - Player {player_num}:")
    print(f"  N subjects: {n_subjects}")
    print(f"  Overall mean accuracy: {np.mean(grand_mean):.2f}% ± {np.mean(sem):.2f}%")
    print(f"  Peak accuracy: {np.max(grand_mean):.2f}% at {time_axis[np.argmax(grand_mean)]:.2f}s")
    
    # Statistical test (simple t-test against chance)
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(mean_matrix, 33.33, axis=0)
    significant_bins = np.where(p_value < 0.05)[0]
    print(f"  Significant above chance at {len(significant_bins)} bins (p<0.05)")
    if len(significant_bins) > 0:
        print(f"    Bins: {significant_bins}")
        print(f"    Times: {time_axis[significant_bins]}s")