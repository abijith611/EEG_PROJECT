"""
Master script to run the entire EEG pipeline sequentially.
Usage:
  python run_all.py                   -> Runs on all 31 pairs
  python run_all.py --test_pairs 4    -> Runs on just the first 4 pairs
  python step3b_plot_Fig2_Fig3.py --test_pairs 4 -> Just plots decoding for first 4 pairs (useful for testing)
"""

import argparse
import sys

import step1_preprocessing
import step2a_decoding
import step2b_markovchain
import step3a_plot_Fig1
import step3b_plot_Fig2_Fig3
import debug_decoding

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full EEG pipeline")
    parser.add_argument('--test_pairs', type=int, default=None, 
                        help="Limit the number of pairs to process for testing (e.g., 4)")
    args = parser.parse_args()
    
    print("==================================================")
    print(f"Starting pipeline... {'(TESTING ON ' + str(args.test_pairs) + ' PAIRS)' if args.test_pairs else '(FULL RUN)'}")
    print("==================================================\n")
    
    print("\n>>> STEP 1: PREPROCESSING <<<")
    step1_preprocessing.run_preprocessing(max_pairs=args.test_pairs)
    
    print("\n>>> STEP 2a: DECODING <<<")
    step2a_decoding.run_decoding(max_pairs=args.test_pairs)
    
    print("\n>>> STEP 2b: MARKOV CHAIN PREDICTABILITY <<<")
    step2b_markovchain.run_markov(max_pairs=args.test_pairs)
    
    print("\n>>> STEP 3a: PLOTTING FIGURE 1 (BEHAVIOR) <<<")
    step3a_plot_Fig1.plot_behavior(max_pairs=args.test_pairs)
    
    print("\n>>> STEP 3b: PLOTTING FIGURES 2 & 3 (DECODING & BFS) <<<")
    step3b_plot_Fig2_Fig3.plot_decoding(max_pairs=args.test_pairs)
    
    print("\n>>> DEBUG: DECODING STATISTICS <<<")
    debug_decoding.run_debug()
    
    print("\n==================================================")
    print("Pipeline Complete! Check 'results/plots' for results.")
    print("==================================================")