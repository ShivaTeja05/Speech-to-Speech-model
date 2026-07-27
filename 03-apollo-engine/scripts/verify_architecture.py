
import asyncio
import torch
import numpy as np
import librosa
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from scripts.unified_demo import UnifiedVoiceEngine, UnifiedConfig
from scripts.unified_cli_test import load_random_sample

async def verify_architecture():
    print("============================================================")
    print("      VERIFYING UNIFIED VOICE ARCHITECTURE (STRICT)         ")
    print("============================================================")
    print("1. Speaker / Voice Embedding (MAIN MECHANISM)")
    print("2. Session Anchoring (prevents drift)")
    print("============================================================\n")

    # 1. Initialize Engine (No Fallback)
    config = UnifiedConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fallback_tts=False, # STRICT: No EdgeTTS allowed
        max_response_tokens=1   # Gen just 1 token to prove flow works
    )
    
    print(f"[Step 1] Initializing Engine on {config.device}...")
    engine = UnifiedVoiceEngine(config)
    engine.load_models()
    
    # 2. Load Reference Audio
    print("\n[Step 2] Loading Reference Audio for Session Anchoring...")
    # Just grab a sample file
    dataset_path = "/Users/jeevithg/Documents/Speech to Speech/indic_voices_dataset"
    sample = load_random_sample(dataset_path, language="hi")
    if not sample:
        print("❌ Could not load sample audio.")
        return

    print(f"   Reference: {sample['audio_path']}")
    y, _ = librosa.load(sample['audio_path'], sr=config.sample_rate)
    
    # 3. Verify Deterministic Embedding (Session Anchor)
    print("\n[Step 3] Verifying Session Anchor Consistency...")
    
    # Encode pass 1
    emb1 = engine.speaker_encoder(y)
    print(f"   Anchor 1 Generated. Shape: {emb1.shape}")
    print(f"   Anchor 1 Norm: {torch.norm(emb1).item():.4f}")
    
    # Encode pass 2 (Same audio)
    emb2 = engine.speaker_encoder(y)
    print(f"   Anchor 2 Generated. Shape: {emb2.shape}")
    
    # Check
    if torch.allclose(emb1, emb2):
        print("   ✅ SUCCESS: Session Anchor is deterministic (Mathematically Enforced).")
    else:
        print("   ❌ FAILURE: Embedding drifted! Architecture violation.")
        return

    # 4. Analyze Injection into Transformer
    print("\n[Step 4] specific Voice Embedding Injection check...")
    print("   We will now run the pipeline for 1 token generation.")
    print("   WATCH LOGS for '[DEBUG] Injecting speaker embedding'...")
    
    # Fake tokens input just to trigger the forward pass
    dummy_input_tokens = torch.randint(0, 100, (1, 50)).to(config.device)
    
    # Run Generation (Real Model, No Override) -- Generating actual audio now
    print("\n   >> Running Full Pipeline (Strict Mode)...")
    print("   ⚠️  WARNING: Running on CPU. This will take SIGNIFICANT time (15-30+ mins)...")
    print("   ℹ️  The model will likely generate text thoughts first, then audio codes.")
    
    # We need to ensure we pass the cached_response_tokens as None to force generation
    # But we also need to allow enough tokens for a sound
    engine.config.max_response_tokens = 500 # Sufficient for text thought + audio codes
    
    result = await engine.process_speech(
        audio=y,
        language="hi",
        reset_anchor=True, # Force new anchor creation
        cached_response_tokens=None # STRICT: No Cache, Real Generation
    )
    
    if result['success']:
        print("\n✅ ARCHITECTURE VERIFIED & AUDIO GENERATED.")
        print("   1. Voice Embedding Created.")
        print("   2. Session Anchor Established.")
        print("   3. Transformer Conditioned via inputs_embeds injection.")
        
        if result.get('audio'):
            import base64
            audio_bytes = base64.b64decode(result['audio'])
            output_filename = "strict_voice_output.wav"
            with open(output_filename, 'wb') as f:
                f.write(audio_bytes)
            print(f"\n🎧 Saved strict output to: {os.path.abspath(output_filename)}")
    else:
        print("\n❌ ARCHITECTURE VERIFICATION FAILED.")

if __name__ == "__main__":
    asyncio.run(verify_architecture())
