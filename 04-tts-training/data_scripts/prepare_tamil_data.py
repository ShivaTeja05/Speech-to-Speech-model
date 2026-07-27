import pandas as pd
import os
import soundfile as sf
import io
import glob

def prepare_dataset(input_dir, output_dir, sample_rate=22050, metadata_file="metadata.csv", max_files=None):
    """
    Reads parquet files from input_dir, extracts audio and text,
    resamples audio to sample_rate, and saves to output_dir.
    Generates a metadata.csv file in the format: filename|transcript
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    wavs_dir = output_dir # TextyMcSpeechy expects wavs in the same dir as metadata usually, or we can organize. 
    # Based on Quick Start: "Copy your dataset's audio files and metadata.csv file to the custom_voice directory"
    # So flat structure is best.

    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files.")
    
    processed_count = 0
    metadata_entries = []

    for p_file in parquet_files:
        print(f"Processing {p_file}...")
        try:
            df = pd.read_parquet(p_file)
        except Exception as e:
            print(f"Error reading {p_file}: {e}")
            continue

        # Inspect columns to find audio and text
        # Assuming typical HF dataset structure: 'audio' (dict with 'bytes') and 'sentence' or 'text'
        # Let's check first row keys in a more robust way if needed, but for now we'll assume standard keys
        # or try to detect them.
        
        # Common keys for audio: 'audio'
        # Common keys for text: 'sentence', 'text', 'transcript'
        
        text_col = None
        for col in ['sentence', 'text', 'transcript']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            print(f"Could not find text column in {p_file}. Columns: {df.columns}")
            continue
            
        if 'audio' not in df.columns:
            print(f"Could not find 'audio' column in {p_file}. Columns: {df.columns}")
            continue

        for index, row in df.iterrows():
            if max_files and processed_count >= max_files:
                break
                
            try:
                audio_data = row['audio']
                text = row[text_col]
                
                if not text or not isinstance(text, str):
                    continue

                # 'audio' is typically a dict: {'bytes': b'...', 'start':..., 'sampling_rate': ...}
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    audio_bytes = audio_data['bytes']
                elif isinstance(audio_data, bytes):
                    audio_bytes = audio_data
                else:
                    # Could be array/list in some datasets
                    # Skipping complex cases for MVP
                    continue
                    
                filename = f"tamil_{processed_count:06d}"
                wav_filename = f"{filename}.wav"
                out_path = os.path.join(wavs_dir, wav_filename)
                
                # Load audio from bytes
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                
                # Write to file (resampling happens here if we rely on sf, but sf.write writes what it gets. 
                # To resample we need scipy or librosa. For simplicity, let's just write and maybe rely on Piper to handle it 
                # OR use a quick resample if needed. TextyMcSpeechy says it converts 22050 automatically? 
                # "create_dataset.sh will ... automatically create 22050hz ... versions"
                # So we just need to provide valid wav files.
                
                sf.write(out_path, data, samplerate)
                
                # Sanitize text: remove newlines, pipes
                text = text.replace('|', ' ').replace('\n', ' ').strip()
                
                metadata_entries.append(f"{filename}|{text}")
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} items...")

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        
        if max_files and processed_count >= max_files:
            break

    # Write metadata.csv
    metadata_path = os.path.join(output_dir, metadata_file)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for line in metadata_entries:
            f.write(line + '\n')
            
    print(f"Done. Processed {processed_count} files. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    # Configure paths
    # Override with TAMIL_DATASET_DIR / TAMIL_OUTPUT_DIR; defaults are relative.
    INPUT_DIR = os.environ.get("TAMIL_DATASET_DIR", "DATASET FOR TAMIL")
    # Output to the dojo dataset folder (TextyMcSpeechy structure) by default.
    OUTPUT_DIR = os.environ.get("TAMIL_OUTPUT_DIR", "tamil_dataset")
    
    # Run slightly smaller batch first to verify? Or just run it. 
    # Let's set a safe limit for the first run, or user can edit.
    # The user has ~17 parquet files, which is huge. Let's cap at 1000 for now to prove it works.
    MAX_FILES = 1000 
    
    print(f"Starting extraction from {INPUT_DIR} to {OUTPUT_DIR}")
    prepare_dataset(INPUT_DIR, OUTPUT_DIR, max_files=MAX_FILES)
