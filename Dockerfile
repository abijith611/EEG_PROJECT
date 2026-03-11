FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system build tools (required for compiling rpy2)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create Conda environment with Python 3.11.9, R, BayesFactor, and rpy2
RUN conda create -n eeg python=3.11.9 r-base r-bayesfactor rpy2 -c conda-forge -y && \
    conda clean -afy

# Activate the environment by adding its bin directory to PATH
ENV PATH /opt/conda/envs/eeg/bin:$PATH

# Install remaining Python packages via pip (rpy2 is removed from here)
RUN pip install --no-cache-dir \
    mne \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    pingouin \
    matplotlib \
    seaborn \
    ptitprince \
    tqdm \
    joblib \
    datalad

# Copy project files
COPY . /app

# Make entrypoint executable
RUN chmod +x /app/download_data.sh

ENTRYPOINT ["conda", "run", "-n", "eeg", "/app/download_data.sh"]
