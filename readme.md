# EEG Project - Decoding Rock-Paper-Scissors Game

This repository contains a complete Python pipeline for analyzing EEG data from a competitive Rock-Paper-Scissors (RPS) game. The project replicates and extends the original MATLAB analysis, adding support for multiple classifiers, Bayesian statistics, and publication-ready figures. The goal is to decode players' own and opponent's responses from EEG signals, compare winners and losers, and evaluate the predictability of moves using Markov chains.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
  - [Command‑Line Arguments](#command-line-arguments)
  - [Examples](#examples)
  - [Performance Optimization](#performance-optimization)
- [Outputs](#outputs)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Repository Structure

```t
.  
├── bayes_output.py          # Computes directional Bayes Factors for winners vs. losers  
├── biosemi64.mat            # Channel coordinates (used for topoplots)  
├── config.py                # Central configuration file (paths, constants, classifier lists) 
├── debug_decoding.py        # Diagnostic script that prints decoding stats & plots averages
├── project/ds006761/        # The dataset directory (created during setup)
├── README.md                # This documentation file 
├── setup_pipeline.py        # Automated cross-platform setup script  
├── run_pipeline.py          # Master script that executes the entire pipeline  
├── step1_preprocessing.py   # Raw data import, bad channel repair, epoching, downsampling  
├── step2a_decoding.py       # Time-resolved and searchlight decoding (multiple classifiers)  
├── step2b_markovchain.py    # Markov chain analysis of response predictability  
├── step3a_plot_Fig1.py      # Generates Figure 1 (behavioural results)  
├── step3b_plot_Fig2_Fig3.py # Generates Figures 2 & 3 (decoding accuracy and Bayes factors)  
└── results/plots/           # Output directory for figures (created automatically)  

```

## Requirements

### Software

- **Python 3.11+** (The setup script will attempt to install this if missing)
- **R** (Required for the `BayesFactor` package; fallback available via `pingouin` if R fails)
- **Git**
### Python Packages

The pipeline requires the following packages (handled automatically by setup):

```t
mne  
numpy  
pandas  
scipy  
scikit-learn  
pingouin  
matplotlib  
rpy2  
tqdm  
joblib  
datalad  
```

### R Packages

- **BayesFactor**  – Installed automatically by the setup script via `rpy2` / `Rscript`.

## Dataset

The full EEG dataset is **78 GB** and must be downloaded from OpenNeuro.

**DOI:** 
- [10.18112/openneuro.ds006761.v1.0.0](<https://openneuro.org/datasets/ds006761>)

The automated setup script handles this via DataLad. If you already have the dataset downloaded, simply place it in `project/ds006761/` before running the setup script, and the download step will be skipped automatically.


## Setup Instructions

### Automated Setup (Recommended)

We provide a cross-platform setup script (`setup_pipeline.py`) that checks your Python version, installs Git/R if needed, creates a virtual environment (`venv_eeg`), installs all dependencies, and downloads the 78 GB dataset via DataLad. It supports Windows, macOS, Ubuntu, and Linux Mint.

- **1. Clone the repository**  
```
git clone <your-repo-url>
cd <repo-folder>
```

- **2. Run the automated setup script** _(recommended)_  

```
python setup_pipeline.py
```
- _Note: The dataset download can take a long time and requires a stable internet connection. If you already have the dataset, place it in `project/ds006761/` before running the script – it will detect it and skip the download._

- **3. Activate the virtual environment:** After the script completes, activate the newly created environment:

- Windows: `venv_eeg\Scripts\activate`
- macOS/Linux: `source venv_eeg/bin/activate`

### Manual Setup
If you prefer to set up the environment manually (or are using an unsupported OS):

- **1. Create and activate a virtual environment:**

```t
python3 -m venv venv_eeg
source venv_eeg/bin/activate  # Or venv_eeg\Scripts\activate on Windows
```
- **2. Install Python dependencies:** 
```
pip install mne numpy pandas scipy scikit-learn pingouin matplotlib rpy2 tqdm joblib datalad
```

- **3. Install R and the BayesFactor package:** Ensure R is installed on your system path, then run:
```
Rscript -e "install.packages('BayesFactor')"
```
- **4. Download the dataset:** 
```
mkdir -p project/ds006761
datalad install https://github.com/OpenNeuroDatasets/ds006761.git project/ds006761
cd project/ds006761
datalad get .
cd ../..

```
## Usage

The main entry point is `run_all.py`. It sequentially executes all preprocessing, decoding, Markov chain, and plotting steps.

### Command‑Line Arguments

| Argument            | Description                                                      | Default                    |
|---------------------|------------------------------------------------------------------|----------------------------|
| `--test_pairs N`    | Process only the first `N` pairs (for quick testing)            | `None` (all pairs)         |
| `--classifiers LIST`| Space‑separated list of classifiers to run                       | `svm lda logistic ridge`   |
| `--skip_searchlight`| Skip the computationally expensive searchlight analysis          | `False`                    |

**Available classifiers:** svm, lda, logistic, ridge

### Examples

**Full analysis** _(may take several hours / days depending on hardware):_
```
python run_all.py  
```
**Quick test** _(only 4 pairs, only SVM and LDA, skip searchlight):_
```
python run_all.py --test_pairs 4 --classifiers svm lda --skip_searchlight  
```
**Run only a specific classifier** _(e.g., logistic regression):_
```
python run_all.py --classifiers logistic  
```
### Performance Optimization

The searchlight decoding step (step2a_decoding.py) can be computationally heavy because it evaluates every channel‑time pair. To speed it up, the script uses parallel processing via joblib. You can control the number of parallel jobs by setting the variable N_JOBS_SEARCHLIGHT at the top of the file:
```t
N_JOBS_SEARCHLIGHT = -1 # Use all available CPU cores  
\# Set to 1 to disable parallelism (useful for debugging)  
\# Set to e.g., 4 to use exactly 4 cores  
```
- Using -1 will utilize all CPU cores, dramatically reducing runtime on multi‑core machines.
- If you run out of memory, reduce the number of jobs (e.g., to 2 or 4).
- The default value is -1 to maximize speed.
- For even faster testing, use the --skip_searchlight flag to bypass this step entirely.

## Outputs

All generated files are stored in:

`project/ds006761/derivatives/` – decoding results (`.pkl` files)

`results/plots/` – figures (`Figure1.png`, `Figure2_*.png`, `Figure3_*.png`)

Refer to the original README for detailed descriptions of each output file.

### Decoding Results (per classifier)

For each subject and classifier, a pickle file named pair-XX_player-Y_task-RPS_decoding_&lt;clf&gt;.pkl contains:

- 'decoding': list of 4 arrays (20 time bins) - time‑resolved accuracy for each condition.
- 'searchlight': list of 4 arrays (64 channels × 20 time bins) - searchlight accuracy maps.
- 'ch_names': list of channel names.
- 'classifier': classifier name.

### Markov Chain Results

- markov_chain_pred.npy - contains prediction accuracy for window sizes 5-100.

### Figures

- Figure1.png - behavioural results (outcome frequencies, response distribution, switch rates, Markov chain predictability).
- Figure2_&lt;clf&gt;.png - decoding accuracy over time (overall) with Bayes factors and topoplots (if searchlight was run).
- Figure3_&lt;clf&gt;.png - winners vs. losers decoding accuracy with Bayes factors.

### Console Output

- bayes_output.py prints a table of peak accuracies and Bayes factors for winners and losers, formatted for easy inclusion in a report.

## Detailed File Descriptions

### step1_preprocessing.py

- Reads raw .bdf files and event files.
- Splits data by player, renames channels to standard 10‑20 labels.
- Epochs from -0.2 s to 5.0 s relative to decision onset.
- Repairs bad channels using inverse‑distance weighting (threshold 5 cm) based on participants.tsv.
- Downsamples to 256 Hz and saves as -epo.fif files.

### step2a_decoding.py

- Loads epoched data and bins it into 20 time windows (250 ms each).
- Creates pseudo‑trials (averages of 4 trials) to increase SNR.
- For each of the four decoding targets (own response, opponent's response, own previous response, opponent's previous response):
  - Runs time‑resolved decoding using 10‑fold stratified group cross‑validation.
  - Runs searchlight decoding (only for SVM and LDA) using parallel processing.
- Saves results in a pickle file named with the classifier suffix.

### step2b_markovchain.py

- Constructs a first‑order Markov chain from the response sequences.
- For window sizes from 5 to 100, computes the probability of the next move based on the previous move.
- Stores prediction accuracy for each participant.

### step3a_plot_Fig1.py

- Generates Figure 1 from the paper:
  - Raincloud plots (half‑violin + boxplot + scatter) for outcome percentages, response frequencies, and game‑to‑game switch rates.
  - Line plot of Markov chain predictability with confidence intervals.
  - Pie charts showing the distribution of most‑played moves.
- Uses custom raincloud implementation for precise control over appearance.

### step3b_plot_Fig2_Fig3.py

- **Figure 2:** Decoding accuracy over time (overall) with Bayes factor dots and topographical maps (if searchlight data exists).
- **Figure 3:** Winners vs. losers decoding accuracy, with three rows of Bayes factors (winners, losers, difference).
- Bayes factors are computed using R's BayesFactor package (falls back to pingouin if R is unavailable).
- Topoplots use channel coordinates from biosemi64.mat and a custom hot colormap. 

### debug_decoding.py

- Quick diagnostic script that loads all decoding files for each classifier and prints mean accuracies per condition, plus winner/loser comparisons.
- Also generates a simple plot for visual inspection.

### bayes_output.py

- Specifically extracts winners' and losers' data, computes peak accuracy and maximum Bayes factor for each phase (decision, response, feedback), and prints a nicely formatted table.
- Uses R's ttestBF with directional null interval (0.5, Inf) to get evidence for above‑chance decoding.

### config.py

Central configuration file that holds all shared constants and settings used by the other scripts.  
- Defines file paths (`PATH_TO_DATA`, `DERIV_DIR`, `PLOT_DIR`).  
- Lists participant IDs to analyse (`PAIR_IDS`).  
- Sets numerical parameters: number of trials (`NUM_TRIALS`), number of decoding targets (`NUM_TESTS`), number of time bins (`NUM_TIME_BINS`), sampling frequency (`SFREQ`), and number of EEG channels (`NUM_CHAN`).  
- Provides the standard 10‑20 channel mapping (`MATLAB_LAYOUT_LABELS`) used for renaming.  
- Controls which classifiers run searchlight (`SEARCHLIGHT_CLASSIFIERS`) and the default list of classifiers (`DEFAULT_CLASSIFIERS`).  
- Includes a helper function `get_pos_dict()` that loads electrode coordinates from `biosemi64.mat` (used for neighbour lists in searchlight and for topoplots).

### run_all.py

- Master script that calls all the above in order.
- Parses command‑line arguments and passes them to each step.

## Reproducibility Notes

- **Random seeds** are set in `step2a_decoding.py` (using pair and player IDs) to ensure consistent pseudo‑trial creation and cross‑validation splits.
- **Python version** is pinned to 3.11.9 – the setup script will attempt to install it if not present.
- **R and BayesFactor** are installed automatically by the script, guaranteeing a consistent environment.
For further details, refer to the original documentation or contact the authors.

## Troubleshooting

- **`setup_pipeline.py` fails on Windows:** Ensure you have administrative privileges – the script may need to install Git, R, or Python via Chocolatey or winget. If automatic installation fails, install the missing components manually and rerun the script.
- **R or BayesFactor not found:** After installation, make sure Rscript is in your system PATH. On Windows you may need to restart your terminal.
- **Memory errors during searchlight:** Use `--skip_searchlight` for testing, or reduce the number of parallel jobs in `step2a_decoding.py` (change `N_JOBS_SEARCHLIGHT`).
- **Dataset download is slow or interrupted:** You can download the dataset manually using DataLad or directly from OpenNeuro, then place it in `project/ds006761/` before running `setup_pipeline.py`.
## References

- **Original dataset:** OpenNeuro [ds006761](https://openneuro.org/datasets/ds006761)
- **Journal Article:** <https://academic.oup.com/scan/article/20/1/nsaf101/8269262>
- **BayesFactor R package:** <https://cran.r-project.org/package=BayesFactor>
- **MNE‑Python:** <https://mne.tools/>

For any questions, please contact [Shriram](<mailto:st194304@stud.uni-stuttgart.de>), [Abijith](<mailto:st194438@stud.uni-stuttgart.de>),  [Tejesh](<mailto:st194770@stud.uni-stuttgart.de>).