# step2b_markovchain.py
"""
RPS Markov Chain
Predict the response of the player based on N previous trials.
Window sizes from 5 to 100.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
from config import PATH_TO_DATA, DERIV_DIR, PAIR_IDS, NUM_TRIALS, get_logger, setup_root_logger

logger = get_logger(__name__)

num_windows = 100


def run_markov(max_pairs: Optional[int] = None) -> None:
    """
    Run Markov chain analysis for all pairs.

    Mathematical Motivation:
    This function computes a 1st-order Markov chain transition matrix for each player's 
    game history. We evaluate how well a player's *next* move can be predicted strictly 
    based on their *previous* move across sliding windows (N=5 to 100).

    Args:
        max_pairs: If given, process only the first `max_pairs` pairs.
    """
    pairs_to_run = PAIR_IDS[:max_pairs] if max_pairs is not None else PAIR_IDS
    num_pairs_run = len(pairs_to_run)
    logger.info(f"Running Markov chain analysis for {num_pairs_run} pairs.")

    Mean_Accuracy = np.zeros((num_pairs_run, 2, num_windows + 1))
    M_pred = np.zeros((num_pairs_run, 2, num_windows + 1, NUM_TRIALS, 4))

    for p_idx, pair in enumerate(pairs_to_run):
        logger.info(f'Loading pair {p_idx + 1} of {num_pairs_run} (ID: {pair})')
        sub_str = f'sub-{pair:02d}'

        events_file = os.path.join(PATH_TO_DATA, sub_str, 'eeg', f'{sub_str}_task-RPS_events.tsv')
        if not os.path.exists(events_file):
            logger.error(f"Events file not found for pair {pair}. Skipping.")
            continue

        events = pd.read_csv(events_file, sep='\t')

        for ppt in [1, 2]:
            ppt_col = 'player1_resp' if ppt == 1 else 'player2_resp'
            resp = events[ppt_col].values

            # =========================================================================
            # SANITY CHECK & DISCUSSION: Input Validation
            # =========================================================================
            # Sanity check: responses should be 1,2,3 (Rock, Paper, Scissors)
            if not np.all(np.isin(resp[resp > 0], [1, 2, 3])):
                logger.warning(f"Invalid response values for pair {pair} player {ppt}.")
            else:
                logger.info(f"Sanity check passed: Pair {pair} Player {ppt} inputs are valid. "
                            f"This seems correct, because the RPS paradigm strictly permits "
                            f"Rock (1), Paper (2), or Scissors (3). Any 0s (missed trials) are safely ignored.")

            # =========================================================================
            # ALGORITHMIC DOCUMENTATION: Transition Matrix & Laplace Smoothing
            # =========================================================================
            # prob_data tracks the cumulative frequencies of 1st-order transitions.
            # It uses a flat 1D array of 13 elements per trial:
            #   Index 0:    Trial number
            #   Index 1-4:  [Total 'Rock', Rock->Rock, Rock->Paper, Rock->Scissors]
            #   Index 5-8:  [Total 'Paper', Paper->Rock, Paper->Paper, Paper->Scissors]
            #   Index 9-12: [Total 'Scissors', Scissors->Rock, Scissors->Paper, Scissors->Scissors]

            prob_data = np.full((NUM_TRIALS, 13), np.nan)
            
            # Why initialize with [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1]?
            # This implements Laplace Smoothing (a Dirichlet prior of alpha=1).
            # We artificially inject 1 "phantom" observation for every possible transition 
            # (e.g., Rock->Rock=1, Rock->Paper=1, Rock->Scissors=1, meaning Total Rock=3).
            # This is mathematically necessary to prevent division-by-zero errors early in 
            # the sequence when a player hasn't executed a specific transition yet.
            prob_data[0, :] = [1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1]

            for i in range(1, NUM_TRIALS):
                prob_data[i, :] = prob_data[i-1, :]
                prob_data[i, 0] = i + 1

                r_prev = resp[i-1]
                r_curr = resp[i]

                if r_prev == 1:
                    prob_data[i, 1] += 1
                    if r_curr == 1:
                        prob_data[i, 2] += 1
                    elif r_curr == 2:
                        prob_data[i, 3] += 1
                    else:
                        prob_data[i, 4] += 1
                elif r_prev == 2:
                    prob_data[i, 5] += 1
                    if r_curr == 1:
                        prob_data[i, 6] += 1
                    elif r_curr == 2:
                        prob_data[i, 7] += 1
                    else:
                        prob_data[i, 8] += 1
                elif r_prev == 3:
                    prob_data[i, 9] += 1
                    if r_curr == 1:
                        prob_data[i, 10] += 1
                    elif r_curr == 2:
                        prob_data[i, 11] += 1
                    else:
                        prob_data[i, 12] += 1

            prob_res = np.full((NUM_TRIALS, 4), np.nan)
            m_Prob = np.full((3, 3), 1/3)
            inter_prob_data = np.full((NUM_TRIALS, 13), np.nan)

            for window_size in range(5, 101):
                for i in range(2, NUM_TRIALS):
                    if i < window_size:
                        inter_prob_data[i, :] = prob_data[i-1, :]
                    else:
                        inter_prob_data[i, :] = prob_data[i-1, :] - prob_data[i-window_size, :]

                    inter_prob_data[i, 0] = i + 1

                    if inter_prob_data[i, 1] > 0:
                        m_Prob[0, :] = inter_prob_data[i, 2:5] / inter_prob_data[i, 1]
                    else:
                        m_Prob[0, :] = [1/3, 1/3, 1/3]

                    if inter_prob_data[i, 5] > 0:
                        m_Prob[1, :] = inter_prob_data[i, 6:9] / inter_prob_data[i, 5]
                    else:
                        m_Prob[1, :] = [1/3, 1/3, 1/3]

                    if inter_prob_data[i, 9] > 0:
                        m_Prob[2, :] = inter_prob_data[i, 10:13] / inter_prob_data[i, 9]
                    else:
                        m_Prob[2, :] = [1/3, 1/3, 1/3]

                    prob_res[i, 0] = resp[i]

                    idx = i
                    if resp[i-1] > 0:
                        idx = i
                    elif i > 1 and resp[i-2] > 0:
                        idx = i-1
                    else:
                        idx = i-2

                    if idx > 0:
                        last_r = resp[idx-1] - 1
                        if last_r >= 0 and last_r <= 2:
                            best_pred = np.argmax(m_Prob[int(last_r), :]) + 1
                            prob_res[i, 1] = best_pred
                            prob_res[i, 2] = np.max(m_Prob[int(last_r), :])

                    if np.isnan(prob_res[i, 2]):
                        prob_res[i, 3] = np.nan
                    elif prob_res[i, 0] == prob_res[i, 1]:
                        prob_res[i, 3] = 1
                    else:
                        prob_res[i, 3] = 0

                data_mean = prob_res[2:480, 3]
                data_mean = data_mean[np.isfinite(data_mean)]
                if len(data_mean) > 0:
                    Mean_Accuracy[p_idx, ppt-1, window_size] = np.mean(data_mean)
                M_pred[p_idx, ppt-1, window_size, :, :] = prob_res

    os.makedirs(DERIV_DIR, exist_ok=True)
    out_file = os.path.join(DERIV_DIR, 'markov_chain_pred.npy')
    np.save(out_file, {'M_pred': M_pred, 'Mean_Accuracy': Mean_Accuracy})
    logger.info(f"Markov chain results saved to {out_file}")


if __name__ == '__main__':
    import argparse
    setup_root_logger(log_to_file=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pairs', type=int, default=None,
                        help='Number of pairs to process (for testing)')
    args = parser.parse_args()
    run_markov(max_pairs=args.test_pairs)