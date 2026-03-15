# EEG Project - Decoding Rock-Paper-Scissors Game

This repository contains a complete Python pipeline for analyzing EEG data from a competitive Rock-Paper-Scissors (RPS) game. The project replicates and extends the original MATLAB analysis, adding support for multiple classifiers, Bayesian statistics, automated HTML reporting, and comprehensive logging. The goal is to decode players' own and opponent's responses from EEG signals, compare winners and losers, and evaluate the predictability of moves using Markov chains.

**Scientific Context:** The pipeline breaks down each trial into three critical cognitive phases for analysis: **Decision (0–2s)**, **Response (2–4s)**, and **Feedback (4–5s)**. By tracking neural activity across these windows, the project identifies when and how players represent their own strategies versus their opponent's actions. *For a comprehensive neuroscientific interpretation of the results and detailed algorithmic justification, please refer to the included `Project_Report.pdf`.*

## Table of Contents

- [Repository Structure](#repository-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Software & Packages](#software--packages)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
  - [Automated Setup (Recommended)](#automated-setup-recommended)
  - [Manual Setup](#manual-setup)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Examples](#examples)
  - [Performance Optimization](#performance-optimization)
- [Outputs](#outputs)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Extensibility: Adding Custom Classifiers](#extensibility-adding-custom-classifiers)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Repository Structure

```t
.  
├── bayes_output.py                   # Computes directional Bayes Factors for winners vs. losers  
├── biosemi64.mat                     # Channel coordinates (used for topoplots)  
├── config.py                         # Central configuration & logging setup 
├── debug_decoding.py                 # Diagnostic script (prints decoding stats & plots averages)
├── generate_report.py                # Compiles all results into a comprehensive HTML report
├── Project_Report.pdf                # Detailed scientific report & parameter justifications
├── project/ds006761/                 # The dataset directory (created during setup)
├── readme.md                         # This documentation file 
├── requirements.txt                  # Exact Python package versions
├── setup_pipeline.py                 # Automated cross-platform setup script  
├── run_pipeline.py                   # Master script that executes the entire pipeline  
├── step1_preprocessing.py            # Raw data import, bad channel repair, epoching, downsampling  
├── step2a_decoding.py                # Time-resolved and searchlight decoding (multiple classifiers)  
├── step2b_markovchain.py             # Markov chain analysis of response predictability  
├── step3a_plot_Fig1.py               # Generates Figure 1 (behavioural results)  
├── step3b_plot_Fig2_Fig3.py          # Generates Figures 2 & 3 (decoding accuracy and Bayes factors)  
├── test_pipeline.py                  # Formal unit test suite for data engineering and statistical validation
└── EEG-PROJECT/results/              # Output directory
    ├── logs/                         # Timestamped execution logs (.md)
    └── plots/                        # Generated figures (.png) and HTML reports (.html)
```

## Pipeline Architecture

```text
[Raw BDF & TSV] (ds006761)
      │
      ▼
[Step 1: Preprocessing] repair bad channels, epoch (-0.2s to 5.0s), downsample (256Hz)
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
[Step 2a: Decoding]                       [Step 2b: Markov Chain]
Time-resolved & Searchlight               Predict response based on sequence history
(SVM, LDA, Logistic, Ridge)                     │
      │                                         │
      ▼                                         ▼
[Step 3b: Plot Figs 2 & 3]                [Step 3a: Plot Fig 1]
Accuracy over time & Bayes Factors        Behavioral distributions & Markov predictability
      │                                         │
      └───────────────────┬─────────────────────┘
                          ▼
                  [generate_report.py]
             (Builds comprehensive HTML summary)
```

## Requirements

### Hardware Requirements

- **RAM:** 16GB+ recommended for standard time-resolved decoding. 32GB+ is strongly recommended if running the parallelized searchlight analysis across all channels (alternatively, reduce `N_JOBS_SEARCHLIGHT` in `config.py`).
- **Storage:** ~100 GB of free disk space is required (78 GB for the raw OpenNeuro dataset + ~20 GB for generated epoched data and decoding results).

### Software & Packages

- **Python 3.11+** (The setup script will attempt to install this if missing)
- **R** (Required for the `BayesFactor` package; fallback available via `pingouin` if R fails)
- **Git**

The pipeline requires the following Python packages (handled automatically by setup):

```text
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

- **R Packages:** `BayesFactor` (Installed automatically by the setup script via `rpy2` / `Rscript`).

## Dataset

The full EEG dataset is **78 GB** and must be downloaded from OpenNeuro.

**DOI:** - [10.18112/openneuro.ds006761.v1.0.0](https://openneuro.org/datasets/ds006761)

The automated setup script handles this via DataLad. If you already have the dataset downloaded, simply place it in `project/ds006761/` before running the setup script, and the download step will be skipped automatically.

## Setup Instructions

### Automated Setup (Recommended)

We provide a cross-platform setup script (`setup_pipeline.py`) that checks your Python version, installs Git/R if needed, creates a virtual environment (`venv_eeg`), installs all dependencies, and downloads the 78 GB dataset via DataLad. It supports Windows, macOS, Ubuntu, and Linux Mint.

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-folder>
```

**2. Run the automated setup script** _(recommended)_  
```bash
python setup_pipeline.py
```

*Note: The dataset download can take a long time and requires a stable internet connection. If you already have the dataset, place it in `project/ds006761/` before running the script – it will detect it and skip the download.*

**3. Activate the virtual environment:** After the script completes, activate the newly created environment:

- Windows: `venv_eeg\Scripts\activate`
- macOS/Linux: `source venv_eeg/bin/activate`

### Manual Setup

If you prefer to set up the environment manually (or are using an unsupported OS):

**1. Create and activate a virtual environment:**

```bash
python3 -m venv venv_eeg
source venv_eeg/bin/activate  # Or venv_eeg\Scripts\activate on Windows
```

**2. Install Python dependencies:** ```bash
pip install mne numpy pandas scipy scikit-learn pingouin matplotlib rpy2 tqdm joblib datalad
```

**3. Install R and the BayesFactor package:** Ensure R is installed on your system path, then run:
```r
Rscript -e "install.packages('BayesFactor')"
```

**4. Download the dataset:**
```bash
mkdir -p project/ds006761
datalad install https://github.com/OpenNeuroDatasets/ds006761.git project/ds006761
cd project/ds006761
datalad get .
cd ../..
```

## Usage

The main entry point is `run_pipeline.py`. It sequentially executes all preprocessing, decoding, Markov chain, plotting steps, and finally generates a comprehensive HTML report. Ensure your virtual environment is active before running.

### Command Line Arguments

| Argument            | Description                                                      | Default                    |
|---------------------|------------------------------------------------------------------|----------------------------|
| `--test_pairs`      | Process only the first `N` pairs (for quick testing)             | `None` (all pairs)         |
| `--classifiers`     | Space‑separated list of classifiers to run                       | `svm lda logistic ridge`   |
| `--skip_searchlight`| Skip the computationally expensive searchlight analysis          | `False`                    |

### Examples

**Run Unit Tests / Sanity Checks** *(Validates data engineering, channel mapping, and Bayes math.):*
```bash
python -m unittest test_pipeline.py -v
```

**Full analysis** _(may take several hours / days depending on hardware):_
```bash
python run_pipeline.py  
```

**Quick test** *(only 4 pairs, only SVM and LDA, skip searchlight computation):*
```bash
python run_pipeline.py --test_pairs 4 --classifiers svm lda --skip_searchlight  
```

**Run only a specific classifier** *(e.g., logistic regression):*
```bash
python run_pipeline.py --classifiers logistic  
```

### Performance Optimization

The searchlight decoding step (`step2a_decoding.py`) evaluates every channel-time pair and is computationally heavy.

- **Parallel Processing**: By default, the script uses all available CPU cores via `joblib` (`N_JOBS_SEARCHLIGHT = -1` in `config.py`).
- **Memory Limits**: If you run out of RAM, edit `config.py` and reduce `N_JOBS_SEARCHLIGHT` (e.g., to `4` or `2`).
- **Bypass**: For fast testing, always use the `--skip_searchlight` flag.

## Outputs

All generated files are stored in structured directories:

- **`project/ds006761/derivatives/`**:
  - Epoched EEG data (`-epo.fif`)
  - Decoding results (`.pkl` files for each subject and classifier)
  - Markov chain predictions (`markov_chain_pred.npy`)

- **`EEG-PROJECT/results/plots/`**:
  - `Figure1.png` (behavioural results)
  - `Figure2_<clf>.png` (overall decoding accuracy over time & topoplots)
  - `Figure3_<clf>.png` (winners vs. losers decoding accuracy)
  - `debug_decoding_<clf>.png` (diagnostic plots)
  - **`report_<timestamp>.html`** (A self-contained HTML report summarizing all stats and embedding all figures)

- **`EEG-PROJECT/results/logs/`**:
  - **`run_<timestamp>.md`** (Detailed timestamped logs of the pipeline execution)

## Detailed File Descriptions

### `Project_Report.pdf` (LaTeX Output)
The primary scientific document accompanying this repository. It provides detailed parameter justifications (e.g., channel repair thresholds, SVM alphas) and in-depth neuroscientific interpretation of the generated Bayes Factors and temporal decoding plots.

### `test_pipeline.py`
Automated unit test suite utilizing Python's `unittest` framework. Contains formal sanity checks that validate the mathematical shape of generated pseudo-trials, prove cross-validation integrity, verify the strict 1:1 mapping of 10-20 system channel coordinates, and assert the statistical logic behind directional Bayes Factor computation.

### `step1_preprocessing.py`
Reads raw .bdf files and event files. Epochs from -0.2 s to 5.0 s relative to decision onset. Repairs bad channels using inverse‑distance weighting (threshold 5 cm). Downsamples to 256 Hz and saves as `-epo.fif` files.

### `step2a_decoding.py`
Loads epoched data and bins it into 20 time windows (250 ms each). Creates pseudo‑trials to increase SNR. Runs time‑resolved and searchlight decoding using 10‑fold stratified group cross‑validation. Saves results as `.pkl` files.

### `step2b_markovchain.py`
Constructs a first‑order Markov chain from the response sequences (window sizes 5 to 100) and computes prediction accuracy for participant behaviors. Implements Laplace smoothing for Dirichlet priors.

### `step3a_plot_Fig1.py`
Generates Figure 1 from the paper using custom raincloud plots (half‑violin + boxplot + scatter) for behavioral data and Markov chain predictability.

### `step3b_plot_Fig2_Fig3.py`
Generates Figure 2 (Decoding accuracy over time with topoplots) and Figure 3 (Winners vs. losers decoding accuracy). Uses R's BayesFactor package for statistics and a custom hot colormap for topographies.

### `debug_decoding.py` & `bayes_output.py`
Diagnostic scripts that calculate overall mean accuracies, winner/loser differences, and specific Bayes Factors across all classifiers, conditions, and phases.

### `generate_report.py`
Dynamically generates a styled, comprehensive HTML report at the end of the pipeline. It automatically embeds generated figures (using base64 encoding) and formats complex tables displaying Bayes factors and decoding accuracies.

### `config.py`
Central configuration file holding paths, standard 10-20 channel mappings, and hyper-parameters. Now includes a centralized logging setup (`setup_root_logger`) to write outputs to both the console and Markdown files.

## Extensibility: Adding Custom Classifiers

You can easily extend the pipeline to use your own scikit-learn compatible classifiers:

1. Define a short string name for your classifier in `config.py` under the `DEFAULT_CLASSIFIERS` list (e.g., `'rf'` for Random Forest).
2. Open `step2a_decoding.py`, locate the `get_classifier()` function, and add your model to the `if/elif` block. Ensure you wrap your model in a standard pipeline: `return make_pipeline(StandardScaler(), YourClassifier())`.

## Reproducibility Notes

- **Random seeds** are strictly set in `step2a_decoding.py` using subject and pair IDs to ensure pseudo-trial generation and K-Fold splits are perfectly reproducible.
- **Environment Isolation:** R packages and Python requirements are pinned to ensure the Bayes calculations do not drift with external package updates.

## Troubleshooting

- **`setup_pipeline.py` fails on Windows:** Ensure you are running your terminal with Administrative privileges, as the script may need to install Git or Python via Windows package managers.
- **"R/BayesFactor not available" Warning:** If R installation failed or isn't on your PATH, the pipeline will seamlessly fall back to `pingouin` for approximated Bayes Factors. Ensure `Rscript` is accessible in your terminal.
- **Memory errors during searchlight:** Use `--skip_searchlight` or manually lower `N_JOBS_SEARCHLIGHT` inside `config.py` if parallel cross-validation exhausts your system memory.
- **Dataset download interrupted:** DataLad depends on Git-Annex. If the download interrupts, you can safely CD into `project/ds006761` and run `datalad get .` again to resume.

## References

- **Original dataset:** OpenNeuro [ds006761](https://openneuro.org/datasets/ds006761)
- **Journal Article:** [SCAN Article](https://academic.oup.com/scan/article/20/1/nsaf101/8269262)
- **BayesFactor R package:** [CRAN BayesFactor](https://cran.r-project.org/package=BayesFactor)
- **MNE‑Python:** [MNE Tools](https://mne.tools/)

For any questions, please contact: **Team SAT** [Shriram](mailto:st194304@stud.uni-stuttgart.de), [Abijith](mailto:st194438@stud.uni-stuttgart.de), [Tejesh](mailto:st194770@stud.uni-stuttgart.de).