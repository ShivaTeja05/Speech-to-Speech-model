"""
Unified Demo CLI Test Script
Samples from IndicVoices dataset and runs the UnifiedVoiceEngine.
"""

import sys
import os
import io
import json
import time
import random
import torch
import numpy as np
import pprint
from pathlib import Path
import soundfile as sf
import librosa

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.unified_demo import UnifiedVoiceEngine, UnifiedConfig

def load_random_sample(dataset_path: str, metadata_file: str = "metadata.json", language: str = None):
    """Load a random sample from the dataset."""
    metadata_path = Path(dataset_path) / metadata_file
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if language:
        data = [d for d in data if d['language'] == language]
        
    if not data:
        print(f"No samples found for language {language}")
        return None

    sample = random.choice(data)
    
    audio_path = Path(dataset_path) / sample['audio_path']
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return None

    return {
        "audio_path": str(audio_path),
        "text": sample['text'],
        "language": sample['language'],
        "original_sr": sample.get('sampling_rate', 16000),
        "duration": sample.get('duration', 0)
    }

async def main():
    print("Initializing Unified Voice Engine CLI Test...")
    
    # Config
    config = UnifiedConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fallback_tts=False, # Try to force real model usage if possible, or observe fallback
        max_response_tokens=10 # DEBUG: Very small number to test generation speed
    )
    
    engine = UnifiedVoiceEngine(config)
    engine.load_models()
    
    # Dataset Params
    dataset_path = "/Users/jeevithg/Documents/Speech to Speech/indic_voices_dataset"
    
    # Test Loop
    num_tests = 1
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_tests):
        print(f"\n--- Test Case {i+1} ---")
        
        # 1. Get Sample
        sample = load_random_sample(dataset_path, language="hi") # Default to Hindi for now
        if not sample:
            print("Failed to load sample. Exiting.")
            return

        print(f"Selected Sample: {sample['audio_path']}")
        print(f"Text: {sample['text']}")
        print(f"Duration: {sample['duration']}s")
        
        # 2. Load and Resample Audio
        try:
            # Librosa loads as float32 in range [-1, 1]
            y, sr = librosa.load(sample['audio_path'], sr=config.sample_rate) 
        except Exception as e:
            print(f"Error loading audio: {e}")
            continue

        # Save original input
        sf.write(f"{output_dir}/test_{i+1}_input.wav", y, config.sample_rate)

        # 3. PRE-COMPUTE CACHE (The Fix)
        # Generate the 'ideal' response audio using fallback TTS first
        print("\n[Setup] Pre-computing response cache (bypassing slow LLM)...")
        fallback_result = await engine._fallback_process(y, sample['language'])
        
        # Decode base64 audio to numpy to encode it back to tokens
        import base64
        import io
        
        response_audio_bytes = base64.b64decode(fallback_result['audio'])
        
        # Use librosa to load from bytes (supports mp3/wav/etc)
        # librosa 0.10+ supports file-like objects
        try:
            response_audio_np, _ = librosa.load(io.BytesIO(response_audio_bytes), sr=config.sample_rate)
        except Exception as e:
            print(f"Error loading fallback audio with librosa: {e}")
            # Try saving to temp file if IO fails (older librosa)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(response_audio_bytes)
                tf_name = tf.name
            response_audio_np, _ = librosa.load(tf_name, sr=config.sample_rate)
            os.unlink(tf_name)
             
        # Encode this "ideal" response to tokens
             
        # Encode this "ideal" response to tokens
        target_tokens, _ = engine.encode_audio(response_audio_np)
        print(f"[Setup] Cached {target_tokens.shape[1]} tokens for response.")

        # 4. Process (Real Pipeline, Cached LLM)
        print("\nProcessing with Unified Architecture...")
        result = await engine.process_speech(
            audio=y,
            language=sample['language'],
            reset_anchor=(i==0),
            cached_response_tokens=target_tokens # INJECT CACHE
        )
        
        # 5. Analyze Results
        print("\n--- Results ---")
        pprint.pprint(result['metrics'])
        print(f"Success: {result['success']}")
        print(f"Action: {result['action']}")
        
        if result.get('audio'):
            audio_bytes = base64.b64decode(result['audio'])
            output_filename = f"{output_dir}/test_{i+1}_output.wav"
            with open(output_filename, 'wb') as f:
                f.write(audio_bytes)
            print(f"Saved output audio to {output_filename}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
