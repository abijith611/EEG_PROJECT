import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_behavior(markov_results):
    """Recreates Figure 1F: Markov predictability[cite: 181]."""
    plt.figure(figsize=(8, 5))
    windows = range(5, 101)
    avg = markov_results.mean(axis=1) * 100
    plt.plot(windows, avg, color='dodgerblue', lw=3)
    plt.axhline(33.3, color='black', ls='--')
    plt.title("Markov Chain Predictability")
    plt.xlabel("N Previous Games")
    plt.ylabel("Accuracy (%)")
    plt.show()

def plot_decoding(results, bfs):
    """Recreates Figure 2: Decoding Accuracy & Bayes Factors[cite: 217]."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    data = np.array(list(results.values())) * 100
    mean_acc = np.mean(data, axis=0)
    
    ax1.plot(mean_acc, marker='o', color='red')
    ax1.axhline(33.3, color='black', ls='--')
    ax1.set_ylabel("Decoding Accuracy (%)")
    
    ax2.bar(range(len(bfs)), bfs, color='purple')
    ax2.axhline(0, color='black')
    ax2.set_ylabel("Log10 Bayes Factor")
    plt.show()