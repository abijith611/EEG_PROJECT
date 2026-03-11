# EEG Project - Decoding Rock-Paper-Scissors Game

This repository contains a complete Python pipeline for analyzing EEG data from a competitive Rock-Paper-Scissors (RPS) game. The project replicates and extends the original MATLAB analysis, adding support for multiple classifiers, Bayesian statistics, and publication‑ready figures. The goal is to decode players' own and opponent's responses from EEG signals, compare winners and losers, and evaluate the predictability of moves using Markov chains.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
  - [Software](#software)
  - [Python Packages](#python-packages)
  - [R Package](#r-package)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
  - [Option 1: Manual Installation](#option-1-manual-installation)
  - [Option 2: Docker (Recommended)](#option-2-docker-recommended)
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
├── bayes_output.py # Computes directional Bayes Factors for winners vs. losers  
├── biosemi64.mat # Channel coordinates (used for topoplots)  
├── debug_decoding.py # Diagnostic script that prints decoding statistics  
├── ds006761-1.0.0.sh # (Not used) Expired download script from OpenNeuro  
├── Dockerfile # Docker image definition (Python 3.11.9 + R + BayesFactor)  
├── docker-compose.yml # Docker Compose configuration (mounts data volumes)  
├── download_data.sh # Entrypoint script that downloads dataset via DataLad  
├── README.md # This file  
├── run_all.py # Master script that runs the entire pipeline  
├── step1_preprocessing.py # Raw data import, bad channel repair, epoching, downsampling  
├── step2a_decoding.py # Time‑resolved and searchlight decoding (multiple classifiers)  
├── step2b_markovchain.py # Markov chain analysis of response predictability  
├── step3a_plot_Fig1.py # Generates Figure 1 (behavioural results)  
├── step3b_plot_Fig2_Fig3.py # Generates Figures 2 & 3 (decoding accuracy and Bayes factors)  
└── results/plots/ # Output directory for figures (created automatically)  

```

## Requirements

### Software

- **Python 3.11.9** (the exact version used for development)
- **R** (with the BayesFactor package)
- **Docker** (optional, but strongly recommended)

### Python Packages

The following packages are required (install via pip):

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
datalad # for automated dataset download (optional if you already have the data)  

_All packages are listed in the Dockerfile and will be installed automatically when using Docker._

### R Package

- **BayesFactor** - install inside R with:

```R
install.packages("BayesFactor", repos = "https://cloud.r-project.org/")
```

## Dataset

The full EEG dataset is **78 GB** and must be downloaded from OpenNeuro.

**DOI:** 
- [10.18112/openneuro.ds006761.v1.0.0](<https://openneuro.org/datasets/ds006761>)

If you use the Docker method, the dataset will be downloaded automatically via datalad.

If you install manually, you can download it using datalad:
```
pip install datalad
datalad install https://github.com/OpenNeuroDatasets/ds006761.git project/ds006761
cd project/ds006761
datalad get .

```

## Setup Instructions

### Option 1: Manual Installation

- **Clone the repository**  
```
git clone <your-repo-url>
cd <repo-folder>
```

- **Create and activate a Python virtual environment** _(recommended)_  
```
python3.11 -m venv venv  
source venv/bin/activate # On Windows: venv\\Scripts\\activate  
```

- **Install Python packages**  
```
pip install mne numpy pandas scipy scikit-learn pingouin matplotlib rpy2 tqdm joblib datalad  
```

- **Install R and BayesFactor**
  - Install R from <https://www.r-project.org/>
  - In R, run:  

```R
install.packages("BayesFactor", repos = "https://cloud.r-project.org/")
```

- **Download the dataset** _(if not already present)_  
```
mkdir -p project/ds006761
datalad install https://github.com/OpenNeuroDatasets/ds006761.git project/ds006761
cd project/ds006761
datalad get .
cd ../..
```

- **Run the pipeline**  
```
python run_all.py  
```
_(Add optional arguments as described in the_ [_Usage_](#usage) _section.)_

### Option 2: Docker (Recommended)

This method guarantees an identical environment and handles the dataset download automatically.

- **Install Docker and Docker Compose** (Docker Desktop includes both).
- **Clone the repository**  
```
git clone <your-repo-url>
cd <repo-folder>
```

- **Run the pipeline**  
```
docker-compose up  
```
  - **First run:** The Docker image will be built (Python 3.11.9, R, BayesFactor, all Python packages). Then the dataset will be downloaded via datalad into project/ds006761/ (on your host). This may take a while (78 GB). Subsequent runs will skip the download.
  - After the download, the full pipeline executes with default settings (all pairs, all classifiers).

To run a smaller test (e.g., first 4 pairs with only SVM and LDA), use:
```
docker-compose run eeg-pipeline python run_all.py --test_pairs 4 --classifiers svm lda
```  

Results are saved in project/ds006761/derivatives/ and results/plots/ on your host machine.

## Usage

The main entry point is run_all.py. It sequentially executes all preprocessing, decoding, Markov chain, and plotting steps.

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

All generated files are stored in project/ds006761/derivatives/ (for decoding results) and results/plots/ (for figures).

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

### run_all.py

- Master script that calls all the above in order.
- Parses command‑line arguments and passes them to each step.

### Dockerfile & docker-compose.yml

- Define the containerised environment.
- download_data.sh is the entrypoint: it checks for the dataset, downloads it if missing, then runs run_all.py.

## Reproducibility Notes

- **Random seeds** are set in step2a_decoding.py (using pair and player IDs) to ensure consistent pseudo‑trial creation and cross‑validation splits.
- **Python version** is pinned to 3.11.9 in the Docker image.
- **R version** is the latest from Debian's repositories, but BayesFactor is installed from CRAN, ensuring a known version.
- **Docker** guarantees that the exact same software stack runs on any machine.

## Troubleshooting

- **Docker Build Fails:** Ensure Docker Desktop is running and you have a stable internet connection. If git-annex fails to install, comment out that line (it is only needed for datalad; you can also download the dataset manually and place it in project/ds006761).
- **Dataset Download Takes Too Long:** The dataset is 78 GB; a fast internet connection is essential. You can manually download the dataset using datalad outside Docker and place it in project/ds006761 before running docker-compose up. The script will detect it and skip the download.
- **R or BayesFactor Not Found:** In manual installation, ensure R is in your PATH and that BayesFactor is installed correctly. In Docker, the installation is automated; if issues persist, check the Docker build logs.
- **Memory Errors:** The searchlight step is memory‑intensive. Use --skip_searchlight for testing. If you run out of RAM, reduce the number of parallel jobs by setting N_JOBS_SEARCHLIGHT in step2a_decoding.py to a lower number (e.g., 2 or 4) instead of -1.

## References

- **Original dataset:** OpenNeuro [ds006761](https://openneuro.org/datasets/ds006761)
- **BayesFactor R package:** <https://cran.r-project.org/package=BayesFactor>
- **MNE‑Python:** <https://mne.tools/>

For any questions, please contact [Shriram](<mailto:st194304@stud.uni-stuttgart.de>), [Abijith],  [Tejesh].