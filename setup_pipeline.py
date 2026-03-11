#!/usr/bin/env python3
"""
Cross-platform setup script for EEG pipeline.
Supports Windows, macOS, Ubuntu, and Linux Mint.
For other Linux distributions, provides manual instructions.
"""

import os
import sys
import platform
import subprocess
import shutil
import venv
from pathlib import Path

# Configuration
REQUIRED_PYTHON_VERSION = (3, 11)
VENV_DIR = "venv_eeg"
DATASET_DIR = Path("project") / "ds006761"
DATASET_URL = "https://github.com/OpenNeuroDatasets/ds006761.git"
R_PACKAGES = ["BayesFactor"]
PIP_PACKAGES = ["mne", "numpy", "pandas", "scipy", "scikit-learn", "pingouin", "matplotlib", "rpy2", "tqdm", "joblib", "datalad"]

def print_header(msg):
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)

def print_step(msg):
    print(f">>> {msg}")

def check_command(cmd):
    return shutil.which(cmd) is not None

def run_cmd(cmd, check=True, capture=False):
    """Run a shell command, optionally capturing output."""
    print(f"Running: {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Error: command failed: {cmd}")
        sys.exit(1)
    return result

def get_os_info():
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        # Check for Ubuntu or Mint
        try:
            with open("/etc/os-release") as f:
                os_release = f.read().lower()
            if "ubuntu" in os_release or "mint" in os_release:
                return "ubuntu"
            else:
                return "linux_other"
        except:
            return "linux_other"
    else:
        return "unknown"

def check_python_version():
    """Check if Python version is at least required."""
    py_version = sys.version_info[:2]
    if py_version >= REQUIRED_PYTHON_VERSION:
        return True
    else:
        print(f"Error: Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} or higher is required, but you have {py_version[0]}.{py_version[1]}")
        return False

def install_packages_apt(packages):
    run_cmd("sudo apt update")
    run_cmd(f"sudo apt install -y {' '.join(packages)}")

def install_packages_brew(packages):
    if not check_command("brew"):
        print("Error: Homebrew is not installed. Please install from https://brew.sh/")
        sys.exit(1)
    run_cmd(f"brew install {' '.join(packages)}")

def install_packages_choco(packages):
    if not check_command("choco"):
        print("Error: Chocolatey is not installed. Please install from https://chocolatey.org/")
        sys.exit(1)
    run_cmd(f"choco install -y {' '.join(packages)}")

def install_packages_winget(packages):
    for pkg in packages:
        run_cmd(f"winget install -e --id {pkg}")

def ensure_git():
    if check_command("git"):
        print("   git already installed.")
        return
    print("   git not found. Attempting to install...")
    os_info = get_os_info()
    if os_info == "ubuntu":
        install_packages_apt(["git"])
    elif os_info == "macos":
        install_packages_brew(["git"])
    elif os_info == "windows":
        if check_command("choco"):
            install_packages_choco(["git"])
        elif check_command("winget"):
            install_packages_winget(["Git.Git"])
        else:
            print("Error: Please install git manually from https://git-scm.com/")
            sys.exit(1)
    else:
        print(f"Unsupported OS: {os_info}. Please install git manually from https://git-scm.com/")
        sys.exit(1)
    if not check_command("git"):
        print("Error: git installation failed.")
        sys.exit(1)
    print("   git installed successfully.")

def ensure_python():
    # Check if python3.11 or python3 is available with correct version
    python_cmd = None
    for cmd in ["python3.11", "python3"]:
        if check_command(cmd):
            # Check version
            try:
                output = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT, text=True)
                version_str = output.strip().split()[1]
                version = tuple(map(int, version_str.split('.')))
                if version >= REQUIRED_PYTHON_VERSION:
                    python_cmd = cmd
                    break
            except:
                continue
    if python_cmd:
        print(f"   Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+ already installed (using {python_cmd}).")
        return python_cmd
    # Not found, attempt to install
    print(f"   Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+ not found. Attempting to install...")
    os_info = get_os_info()
    if os_info == "ubuntu":
        install_packages_apt(["python3.11", "python3.11-venv", "python3.11-dev"])
        python_cmd = "python3.11"
    elif os_info == "macos":
        install_packages_brew(["python@3.11"])
        python_cmd = "python3.11"
    elif os_info == "windows":
        if check_command("choco"):
            install_packages_choco(["python311"])
            python_cmd = "python"
        elif check_command("winget"):
            install_packages_winget(["Python.Python.3.11"])
            python_cmd = "python"
        else:
            print("Error: Please install Python 3.11 manually from https://www.python.org/downloads/")
            sys.exit(1)
    else:
        print(f"Unsupported OS: {os_info}. Please install Python 3.11 manually from https://www.python.org/downloads/")
        sys.exit(1)
    if not check_command(python_cmd):
        print(f"Error: Python installation failed (command '{python_cmd}' not found).")
        sys.exit(1)
    print(f"   Python installed successfully (using {python_cmd}).")
    return python_cmd

def ensure_r():
    if check_command("Rscript"):
        print("   R already installed.")
        return
    print("   R not found. Attempting to install...")
    os_info = get_os_info()
    if os_info == "ubuntu":
        # Add CRAN repository for latest R
        run_cmd("sudo apt update")
        run_cmd("sudo apt install -y software-properties-common dirmngr")
        run_cmd("sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7'")
        run_cmd('sudo add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"')
        run_cmd("sudo apt update")
        install_packages_apt(["r-base"])
    elif os_info == "macos":
        install_packages_brew(["r"])
    elif os_info == "windows":
        if check_command("choco"):
            install_packages_choco(["r.project"])
        elif check_command("winget"):
            install_packages_winget(["RProject.R"])
        else:
            print("Error: Please install R manually from https://www.r-project.org/")
            sys.exit(1)
    else:
        print(f"Unsupported OS: {os_info}. Please install R manually from https://www.r-project.org/")
        sys.exit(1)
    if not check_command("Rscript"):
        print("Error: R installation failed.")
        sys.exit(1)
    print("   R installed successfully.")

def install_r_packages(packages):
    print(f">>> Installing R packages: {', '.join(packages)}")
    for pkg in packages:
        cmd = f'Rscript -e "if (!require(\'{pkg}\', quietly = TRUE)) install.packages(\'{pkg}\', repos = \'https://cloud.r-project.org/\')"'
        run_cmd(cmd)

def create_virtualenv(python_cmd, venv_dir):
    print(f">>> Creating virtual environment in '{venv_dir}'...")
    if os.path.exists(venv_dir):
        print("   Virtual environment already exists.")
    else:
        venv.create(venv_dir, with_pip=True)
        print("   Virtual environment created.")
    
    # Upgrade pip using python -m pip (more reliable on Windows)
    if os.name == "nt":
        py_path = os.path.join(venv_dir, "Scripts", "python")
    else:
        py_path = os.path.join(venv_dir, "bin", "python")
    print("   Upgrading pip...")
    run_cmd(f"{py_path} -m pip install --upgrade pip")

def install_python_packages(venv_dir, packages):
    print(">>> Installing Python packages...")
    pip_cmd = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip")
    run_cmd(f"{pip_cmd} install {' '.join(packages)}")

def download_dataset(venv_dir):
    print(">>> Checking dataset...")
    dataset_path = Path(DATASET_DIR)
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"   Dataset already exists in {dataset_path}. Skipping download.")
        return
    print(f"   Downloading dataset from OpenNeuro using datalad (this may take a while, ~78 GB)...")
    if os.name == "nt":
        python_cmd = os.path.join(venv_dir, "Scripts", "python")
    else:
        python_cmd = os.path.join(venv_dir, "bin", "python")
    # Use python -m datalad
    run_cmd(f"{python_cmd} -m datalad install {DATASET_URL} {dataset_path}")
    run_cmd(f"cd {dataset_path} && {python_cmd} -m datalad get .")
    print("   Dataset download complete.")

def main():
    print_header("EEG Pipeline – Automated Setup (no run)")

    # Step 1: Check prerequisites
    print_step("Checking prerequisites...")
    if not check_python_version():
        sys.exit(1)
    python_cmd = ensure_python()
    ensure_git()
    ensure_r()

    # Step 2: Create virtual environment
    create_virtualenv(python_cmd, VENV_DIR)

    # Step 3: Install Python packages
    install_python_packages(VENV_DIR, PIP_PACKAGES)

    # Step 4: Install R packages
    install_r_packages(R_PACKAGES)

    # Step 5: Ensure dataset directory exists
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Step 6: Download dataset if missing
    download_dataset(VENV_DIR)

    print_header("Setup complete!")
    print("\nYou can now run the EEG pipeline manually:")
    print(f"  1. Activate the virtual environment:")
    if os.name == "nt":
        print(f"       {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"       source {VENV_DIR}/bin/activate")
    print(f"  2. Run the pipeline with desired options:")
    print(f"       python run_all.py [--test_pairs N] [--classifiers LIST] [--skip_searchlight]")
    print(f"\nFor example:")
    print(f"       python run_all.py --test_pairs 4 --classifiers svm lda --skip_searchlight")
    print(f"\nAll outputs will be saved in 'results/plots/' and 'project/ds006761/derivatives/'.\n")

if __name__ == "__main__":
    main()