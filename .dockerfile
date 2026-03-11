# Use the exact Python version you used for development
FROM python:3.11.9-slim

# Install system dependencies: R, git-annex (for datalad), and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    git-annex \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install BayesFactor R package
RUN Rscript -e "install.packages('BayesFactor', repos='https://cloud.r-project.org/')"

# Set working directory
WORKDIR /app

# Install Python packages (list from your readme)
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
    rpy2 \
    tqdm \
    joblib \
    datalad

# Copy the entire project into the container
COPY . /app

# Make the download script executable
RUN chmod +x /app/download_data.sh

# Set the entrypoint to run the download script and then the pipeline
ENTRYPOINT ["/app/download_data.sh"]