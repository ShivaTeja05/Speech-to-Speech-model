# Layer 04 — TTS Training (custom Tamil voice)

The pipeline that trained our **own Tamil TTS voice**
(`ta_IN-iitm-female-s1-medium.onnx`), which layers 01 and 02 load as the Tamil
entry in their per-language TTS set. Built from the SPRINGLab **IndicTTS_Tamil**
dataset.

> Full context and where this fits: see the root [ARCHITECTURE.md](../ARCHITECTURE.md#layer-04--tts-training-the-tamil-voice).

## Contents

```
04-tts-training/
├── notebooks/          # Piper + Coqui-VITS training notebooks (Colab/GPU)
├── data_scripts/       # Download / extract / prepare the IndicTTS_Tamil dataset
├── voice_assets/       # Trained HTS voice asset
└── Local_Mac_Training_Guide.md
```

### `data_scripts/`
| Script | Purpose |
|--------|---------|
| `download_dataset.py`, `retry_downloads.py`, `fix_corrupt_file.py` | Fetch IndicTTS_Tamil parquet shards from Hugging Face. |
| `inspect_parquet.py`, `check_gender.py` | Inspect shards / confirm speaker. |
| `extract_dataset.py`, `extract_fixed_part.py` | Extract WAV + text from parquet. |
| `prepare_tamil_data.py` | Build the Piper dataset (`metadata.csv` + `wav_22050/`). |
| `manual_link_dataset.py` | Link the dataset into the training dojo. |
| `download_checkpoint.py` | Fetch a pretrained Piper checkpoint to fine-tune from. |
| `train_piper.sh` | Piper training launcher. |

### `notebooks/`
- `Piper_Tamil_Training.ipynb`, `Tamil_Piper_Official_Fix.ipynb`, `Piper_Colab_Conda_Py310.ipynb`, `piper_multilingual_training_notebook.ipynb` — Piper training (Colab/GPU, with dependency fixes).
- `Coqui_VITS_Tamil.ipynb` — alternative Coqui VITS training track.

## Configurable paths

The data scripts default to **relative paths** and accept environment overrides
so they run on any machine:

```bash
export TAMIL_DATASET_DIR="path/to/DATASET FOR TAMIL"   # input parquet/wav
export TAMIL_OUTPUT_DIR="path/to/output_dataset"       # prepared dataset
export PYTHON_EXEC="$(command -v python3)"              # for train_piper.sh
```

## Outputs
- `voice_assets/iitm_unified_tamil_female.htsvoice` — HTS voice asset.
- `../02-voice-model-ai/ta_IN-iitm-female-s1-medium.onnx` (+ `.json`) — the trained Piper voice used at runtime.

> Third-party tool clones used during training (`piper/`, `piper-phonemize/`,
> `TextyMcSpeechy/`) are **not** vendored here — they are upstream dependencies.
