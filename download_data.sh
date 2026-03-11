#!/bin/bash
set -e

DATASET_URL="https://github.com/OpenNeuroDatasets/ds006761.git"   # Git clone URL for datalad
TARGET_DIR="/app/project/ds006761"

# Check if dataset is already present and non-empty
if [ -d "$TARGET_DIR" ] && [ "$(ls -A $TARGET_DIR)" ]; then
    echo "Dataset already exists at $TARGET_DIR. Skipping download."
else
    echo "Dataset not found. Downloading from OpenNeuro using DataLad..."
    mkdir -p /app/project
    # Clone the dataset (metadata only first)
    datalad install $DATASET_URL $TARGET_DIR
    cd $TARGET_DIR
    # Actually retrieve all file contents (this will download the full 78GB)
    datalad get .
    echo "Dataset downloaded successfully."
fi

# Now execute the main pipeline script, passing any arguments
exec python /app/run_all.py "$@"