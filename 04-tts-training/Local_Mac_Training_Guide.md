# Local Mac Piper Training Guide (Python 3.10)

This guide will help you set up **Python 3.10** on your Mac to train Piper locally. This **solves the `piper-phonemize` issue** because the official pre-built wheels work perfectly on Python 3.10.

> [!WARNING]
> **Performance Warning:** Training on a Mac CPU/Metal is **significantly slower** than Google Colab. A 1-hour training run on Colab might take **24-48 hours** on your Mac.

---

## Step 1: Install System Requirements
Open your **Terminal** and run these commands:

1.  **Install Homebrew** (if you don't have it):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2.  **Install Espeak-NG** (Critical for Piper):
    ```bash
    brew install espeak-ng
    ```

3.  **Install Miniconda** (To manage Python 3.10):
    *   Download the installer: [Miniconda for Mac (M1/M2/M3 Apple Silicon)](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh)
    *   Or for Intel Mac: [Miniconda for Mac (Intel)](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh)
    *   Install it by running the downloaded script in terminal or following the on-screen prompts.

---

## Step 2: Create Python 3.10 Environment
Once Miniconda is installed, run this in your terminal:

```bash
# 1. Create a clean environment named 'piper_local' with Python 3.10
conda create -n piper_local python=3.10 -y

# 2. Activate the environment
conda activate piper_local
```

---

## Step 3: Install Piper Dependencies
Now that you are in the `piper_local` environment (it should show in your terminal prompt), run:

```bash
# Install PyTorch (Mac optimized)
pip install torch torchvision torchaudio

# Install Piper Phonemize (Works instantly on Py3.10!)
pip install piper-phonemize

# Install PyTorch Lightning (Compatible version)
pip install "pytorch-lightning==1.9.5" "torchmetrics>=0.7.0"

# Install other tools
pip install onnx onnxruntime
```

---

## Step 4: Clone Piper & Install
```bash
# Go to your desktop or working folder
cd ~/Desktop

# Clone Piper
git clone https://github.com/rhasspy/piper.git
cd piper/src/python

# Install Piper requirements
pip install -r requirements.txt
pip install -e .

# Build the alignment tool (Required)
./build_monotonic_align.sh
```

---

## Step 5: Start Training
Assuming you have your dataset ready (metadata.csv and wavs folder):

1.  **Preprocess:**
    ```bash
    python -m piper_train.preprocess \
      --language ta \
      --input-dir /path/to/your/dataset \
      --output-dir ~/Desktop/piper_model \
      --dataset-name tamil_local \
      --dataset-format ljspeech \
      --sample-rate 22050
    ```

2.  **Train:**
    ```bash
    python -m piper_train \
        --dataset-dir ~/Desktop/piper_model \
        --accelerator cpu \
        --devices 1 \
        --batch-size 4 \
        --max_epochs 100 \
        --quality medium
    ```

*(Note: We use `--accelerator cpu` because Piper's training code often has issues with Mac's `mps` GPU accelerator. CPU is safer but slower.)*
