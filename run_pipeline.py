"""
Master script to run the entire EEG pipeline sequentially.
Usage:
  python run_all.py [--test_pairs N] [--classifiers LIST] [--skip_searchlight]
"""

import argparse
import sys

import step1_preprocessing
import step2a_decoding
import step2b_markovchain
import step3a_plot_Fig1
import step3b_plot_Fig2_Fig3
import debug_decoding
import bayes_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full EEG pipeline")
    parser.add_argument('--test_pairs', type=int, default=None,
                        help="Limit the number of pairs to process for testing (e.g., 4)")
    parser.add_argument('--classifiers', nargs='+', default=['svm', 'lda', 'logistic', 'ridge'],
                        choices=['svm', 'lda', 'logistic', 'ridge', 'rf', 'gb', 'knn', 'nb', 'mlp', 'elastic', 'qda', 'rbf_svm'],
                        help="List of classifiers to run in decoding step")
    parser.add_argument('--skip_searchlight', action='store_true',
                        help="Skip searchlight computation (faster)")
    args = parser.parse_args()

    print("==================================================")
    print(f"Starting pipeline... {'(TESTING ON ' + str(args.test_pairs) + ' PAIRS)' if args.test_pairs else '(FULL RUN)'}")
    print(f"Classifiers: {', '.join(args.classifiers)}")
    print("==================================================\n")

    print("\n>>> STEP 1: PREPROCESSING <<<")
    step1_preprocessing.run_preprocessing(max_pairs=args.test_pairs)

    print("\n>>> STEP 2a: DECODING <<<")
    step2a_decoding.run_decoding(max_pairs=args.test_pairs, classifiers=args.classifiers)

    print("\n>>> STEP 2b: MARKOV CHAIN PREDICTABILITY <<<")
    step2b_markovchain.run_markov(max_pairs=args.test_pairs)

    print("\n>>> STEP 3a: PLOTTING FIGURE 1 (BEHAVIOR) <<<")
    step3a_plot_Fig1.plot_behavior(max_pairs=args.test_pairs)

    print("\n>>> STEP 3b: PLOTTING FIGURES 2 & 3 (DECODING & BFS) <<<")
    for clf in args.classifiers:
        print(f"\n--- Plotting for classifier: {clf} ---")
        step3b_plot_Fig2_Fig3.plot_decoding(max_pairs=args.test_pairs, classifier=clf)

    print("\n>>> DEBUG: DECODING STATISTICS <<<")
    debug_decoding.run_debug()

    print("\n>>> BAYES FACTOR OUTPUT FOR REPORT <<<")
    bayes_output.extract_winner_loser_stats()

    print("\n==================================================")
    print("Pipeline Complete! Check 'results/plots' for plots.")
    print("==================================================")