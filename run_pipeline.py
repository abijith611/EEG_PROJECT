# run_pipeline.py
"""
Master script to run the entire EEG pipeline sequentially.
Usage:
  python run_pipeline.py [--test_pairs N] [--classifiers LIST] [--skip_searchlight]
"""

import argparse
import sys
import time
import logging
from config import setup_root_logger, get_logger, LOG_DIR

# IMPORTANT: Configure root logger BEFORE any other imports that might create loggers.
setup_root_logger(level=logging.INFO, log_to_file=True)

# import pipeline steps
import step1_preprocessing
import step2a_decoding
import step2b_markovchain
import step3a_plot_Fig1
import step3b_plot_Fig2_Fig3
import debug_decoding
import bayes_output
import generate_report   

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full EEG pipeline")
    parser.add_argument('--test_pairs', type=int, default=None,
                        help="Limit the number of pairs to process for testing (e.g., 4)")
    parser.add_argument('--classifiers', nargs='+', default=['svm', 'lda', 'logistic', 'ridge'],
                        choices=['svm', 'lda', 'logistic', 'ridge'],
                        help="List of classifiers to run in decoding step")
    parser.add_argument('--skip_searchlight', action='store_true',
                        help="Skip searchlight computation (faster)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info(f"Starting pipeline... {'(TESTING ON ' + str(args.test_pairs) + ' PAIRS)' if args.test_pairs else '(FULL RUN)'}")
    logger.info(f"Classifiers: {', '.join(args.classifiers)}")
    logger.info(f"Skip searchlight: {args.skip_searchlight}")
    logger.info("=" * 50)

    start_time = time.time()

    # Step 1: Preprocessing
    logger.info("\n>>> STEP 1: PREPROCESSING <<<")
    step1_preprocessing.run_preprocessing(max_pairs=args.test_pairs)

    # Step 2a: Decoding
    logger.info("\n>>> STEP 2a: DECODING <<<")
    step2a_decoding.run_decoding(max_pairs=args.test_pairs,
                                 classifiers=args.classifiers,
                                 skip_searchlight=args.skip_searchlight)

    # Step 2b: Markov chain predictability
    logger.info("\n>>> STEP 2b: MARKOV CHAIN PREDICTABILITY <<<")
    step2b_markovchain.run_markov(max_pairs=args.test_pairs)

    # Step 3a: Plot Figure 1 (behaviour)
    logger.info("\n>>> STEP 3a: PLOTTING FIGURE 1 (BEHAVIOR) <<<")
    step3a_plot_Fig1.plot_behavior(max_pairs=args.test_pairs)

    # Step 3b: Plot Figures 2 & 3 (decoding & Bayes factors)
    logger.info("\n>>> STEP 3b: PLOTTING FIGURES 2 & 3 (DECODING & BFS) <<<")
    for clf in args.classifiers:
        logger.info(f"\n--- Plotting for classifier: {clf} ---")
        step3b_plot_Fig2_Fig3.plot_decoding(max_pairs=args.test_pairs, classifier=clf)

    # Debug: decoding statistics
    logger.info("\n>>> DEBUG: DECODING STATISTICS <<<")
    debug_decoding.run_debug()

    # Bayes factor output for report
    logger.info("\n>>> BAYES FACTOR OUTPUT FOR REPORT <<<")
    bayes_output.extract_winner_loser_stats()

    # Generate final HTML report
    logger.info("\n>>> GENERATING HTML REPORT <<<")
    generate_report.run_report()   

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 50)
    logger.info(f"Pipeline Complete! Total time: {elapsed:.2f} seconds")
    logger.info("Check 'results/plots' for plots and 'results/logs' for log files.")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()