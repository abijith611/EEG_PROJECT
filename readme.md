# EEG Project – Version 2 (Python)

Recreation of the original EEG analysis pipeline in Python, with support for multiple classifiers.

## Requirements

Install the following packages (all mandatory):

```bash
pip install mne numpy pandas scipy scikit-learn pingouin matplotlib seaborn ptitprince rpy2 tqdm joblib

#Additionally, you need R installed and the BayesFactor R package:

install.packages("BayesFactor")