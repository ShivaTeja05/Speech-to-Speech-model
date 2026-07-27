import pandas as pd
import os

# Configuration
input_file = "DATASET FOR TAMIL/train-00003-of-00017.parquet"
output_dir = "DATASET FOR TAMIL/extracted_wavs"
metadata_file = "DATASET FOR TAMIL/metadata.csv"

# Global index offset - need to find the last index used
# Ideally we should read the existing metadata to find the max index, 
# or just append with a unique prefix to be safe.
# Let's use a high offset to avoid collisions or parse metadata.
# Actually, let's parse metadata to be safe.

existing_ids = []
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) > 0:
                existing_ids.append(parts[0])

# Just use a prefix for this batch
prefix = "new_batch_"
global_idx = 0

print(f"Extracting {input_file} to {output_dir}...")
new_metadata = []

try:
    df = pd.read_parquet(input_file)
    for index, row in df.iterrows():
        audio_data = row['audio']
        text = row['text']
        
        if audio_data and 'bytes' in audio_data:
            filename = f"{prefix}audio_{global_idx:06d}"
            wav_path = os.path.join(output_dir, f"{filename}.wav")
            
            with open(wav_path, "wb") as f:
                f.write(audio_data['bytes'])
            
            new_metadata.append(f"{filename}|{text}")
            global_idx += 1
            
    print(f"Extracted {len(new_metadata)} new files.")
    
    # Append to metadata
    with open(metadata_file, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(new_metadata))
        
except Exception as e:
    print(f"Error: {e}")
