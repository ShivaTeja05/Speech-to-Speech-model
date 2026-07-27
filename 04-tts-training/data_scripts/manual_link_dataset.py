import os

# Configuration
dojo_dir = "piper tamil model /tts_dojo/tamil_model_dojo"
dataset_name = "tamil_voice"
datasets_dir = "piper tamil model /tts_dojo/DATASETS"
quality = "M" # Medium
sampling_rate = 22050
max_workers = 4 

# Paths
scripts_dir = os.path.join(dojo_dir, "scripts")
target_dataset_dir = os.path.join(dojo_dir, "target_voice_dataset")
pretrained_checkpoint_dir = os.path.join(dojo_dir, "pretrained_tts_checkpoint")

dataset_source = os.path.join(datasets_dir, dataset_name)
wav_source = os.path.join(dataset_source, "wav_22050")
metadata_source = os.path.join(dataset_source, "metadata.csv")
conf_source = os.path.join(dataset_source, "dataset.conf")

# Default Checkpoint Source (Generic Medium)
# Note: User confirmed use of "Generic" model earlier.
# Check where we downloaded it: "piper tamil model /tts_dojo/PRETRAINED_CHECKPOINTS/default/M_voice/medium/epoch=2164-step=1355540.ckpt"
checkpoint_source_dir = "piper tamil model /tts_dojo/PRETRAINED_CHECKPOINTS/default/M_voice/medium"
# Find actual file
checkpoint_file = None
if os.path.exists(checkpoint_source_dir):
    for f in os.listdir(checkpoint_source_dir):
        if f.endswith(".ckpt"):
            checkpoint_file = os.path.join(checkpoint_source_dir, f)
            break

print(f"Configuring Dojo in {dojo_dir}...")

# 1. Create hidden config files in scripts/
os.makedirs(scripts_dir, exist_ok=True)
with open(os.path.join(scripts_dir, ".QUALITY"), "w") as f: f.write(quality) # Actually wait, link_dataset puts .QUALITY in target_voice_dataset? No, scripts/link_dataset.sh says: VARFILE_QUALITY="../target_voice_dataset/.QUALITY"
with open(os.path.join(scripts_dir, ".SAMPLING_RATE"), "w") as f: f.write(str(sampling_rate))
with open(os.path.join(scripts_dir, ".MAX_WORKERS"), "w") as f: f.write(str(max_workers))

# 2. Config files in target_voice_dataset
os.makedirs(target_dataset_dir, exist_ok=True)
with open(os.path.join(target_dataset_dir, ".QUALITY"), "w") as f: f.write(quality)
with open(os.path.join(target_dataset_dir, ".SCRATCH"), "w") as f: f.write("false") # Use pretrained

# 3. Create Symlinks
# wav
link_wav = os.path.join(target_dataset_dir, "wav")
if os.path.exists(link_wav): os.remove(link_wav)
# We need absolute paths or correct relative paths. Python's os.symlink(src, dst)
# src should be absolute to be safe.
os.symlink(os.path.abspath(wav_source), os.path.abspath(link_wav))

# metadata
link_meta = os.path.join(target_dataset_dir, "metadata.csv")
if os.path.exists(link_meta): os.remove(link_meta)
os.symlink(os.path.abspath(metadata_source), os.path.abspath(link_meta))

# dataset.conf
link_conf = os.path.join(target_dataset_dir, "dataset.conf")
if os.path.exists(link_conf): os.remove(link_conf)
os.symlink(os.path.abspath(conf_source), os.path.abspath(link_conf))

# 4. Link Checkpoint
if checkpoint_file:
    print(f"Linking checkpoint: {checkpoint_file}")
    # Remove existing in pretrained_tts_checkpoint
    for f in os.listdir(pretrained_checkpoint_dir):
        os.remove(os.path.join(pretrained_checkpoint_dir, f))
    
    link_ckpt = os.path.join(pretrained_checkpoint_dir, os.path.basename(checkpoint_file))
    os.symlink(os.path.abspath(checkpoint_file), os.path.abspath(link_ckpt))
else:
    print("WARNING: No checkpoint found!")

print("Manual configuration complete.")
