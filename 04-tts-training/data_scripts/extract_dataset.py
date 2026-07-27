import pandas as pd
import os
import wave
import io

# Configuration
input_dir = "DATASET FOR TAMIL"
output_dir = "DATASET FOR TAMIL/extracted_wavs"
metadata_file = "DATASET FOR TAMIL/metadata.csv"

# Speaker mapping based on gender
# 0 -> Male (Speaker 0), 1 -> Female (Speaker 1)
SPEAKER_MAP = {
    0: 0,
    1: 1
}

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Starting extraction from {input_dir} to {output_dir}...")

# Initialize metadata list
metadata_entries = []

# List all parquet files
parquet_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
total_files = len(parquet_files)

global_idx = 0

for i, p_file in enumerate(parquet_files):
    p_path = os.path.join(input_dir, p_file)
    print(f"Processing file {i+1}/{total_files}: {p_file}...")
    
    try:
        df = pd.read_parquet(p_path)
        
        for index, row in df.iterrows():
            audio_data = row['audio'] # Dictionary with 'bytes'
            text = row['text']
            gender = row.get('gender', 0) # Default to 0 if missing
            speaker_id = SPEAKER_MAP.get(gender, 0)
            
            if audio_data and 'bytes' in audio_data:
                # Generate a unique filename
                filename = f"audio_{global_idx:06d}"
                wav_path = os.path.join(output_dir, f"{filename}.wav")
                
                # Write WAV file
                with open(wav_path, "wb") as f:
                    f.write(audio_data['bytes'])
                
                # Add to metadata (Piper expects: filename|speaker|text)
                metadata_entries.append(f"{filename}|{speaker_id}|{text}")
                
                global_idx += 1
                
    except Exception as e:
        print(f"Error processing {p_file}: {e}")

# Write metadata.csv
print(f"Writing {len(metadata_entries)} entries to {metadata_file}...")
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write("\n".join(metadata_entries))

print("Extraction complete!")
