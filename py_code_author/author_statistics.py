import numpy as np
import pingouin as pg

def calculate_bfs(decoding_results, chance=1/3):
    """Calculates Bayes Factors for above-chance decoding[cite: 585]."""
    data_matrix = np.array(list(decoding_results.values()))
    num_bins = data_matrix.shape[1]
    log_bfs = []
    
    for b in range(num_bins):
        # Bayesian t-test against chance level [cite: 581]
        bf = pg.bayesfactor_ttest(data_matrix[:, b], mu=chance, alternative='greater')
        log_bfs.append(np.log10(bf))
    return np.array(log_bfs)