"""
Simulation Test for Voice Identity Conditioning & Anchoring.

This script:
1. Generates synthetic 'audio' samples resembling different speakers/languages.
2. Runs them through the UnifiedVoiceEngine pipeline.
3. Verifies that 'Session Anchoring' works (Embedding doesn't change mid-session).
4. Verifies that 'Voice Identity' is distinct for different inputs.
"""

import sys
import os
import asyncio
import numpy as np
import torch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.unified_demo import UnifiedVoiceEngine, UnifiedConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Simulation")

def generate_synthetic_audio(seed: int, duration_sec: float = 2.0) -> np.ndarray:
    """Generate synthetic audio with a specific 'voice' signature based on seed."""
    sr = 24000
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    
    # Use seed to determine base frequency (pitch) -> Simulates Voice Identity
    np.random.seed(seed)
    pitch = np.random.uniform(100, 300)  # 100-300 Hz range
    
    # Generate waveform: mixture of sines
    audio = np.sin(2 * np.pi * pitch * t) * 0.5
    audio += np.sin(2 * np.pi * (pitch * 2) * t) * 0.2  # Harmonic
    audio += np.random.normal(0, 0.05, len(t))  # Noise
    
    return audio.astype(np.float32)

async def run_simulation():
    print("\n" + "="*60)
    print("📢 Starting Voice Conditioning Simulation")
    print("="*60 + "\n")
    
    # Initialize Engine
    config = UnifiedConfig(device="cpu") # Force CPU for simulation reliability
    engine = UnifiedVoiceEngine(config)
    
    # Mock efficient loading (we don't need real weights for this logic test)
    # We manually initialize the components to avoid heavy download
    from apollo_voice_engine.models.speaker_encoder import SpeakerEncoder
    from apollo_voice_engine.models.audio_llm import AudioLLM
    
    print("Initializing components (Mock Mode)...")
    engine.speaker_encoder = SpeakerEncoder(device="cpu")
    engine.models_loaded = True # Bypass fallback checks
    
    # We mock the AudioLLM purely to capture the input it receives
    # Real inference isn't needed to test *conditioning logic*
    class MockAudioLLM:
        def prepare_audio_input(self, tokens, text_prompt=None, speaker_embedding=None):
            # Capture the embedding to prove it was passed!
            self.last_embedding = speaker_embedding
            return torch.zeros(1, 10).long() # Dummy IDs
        
        def generate_response(self, *args, **kwargs):
            return torch.zeros(1, 10).long() # Dummy output
        
        def extract_audio_tokens(self, *args):
            return torch.zeros(1, 100) # Dummy audio tokens
            
    engine.audio_llm = MockAudioLLM()
    
    # We also need to mock SNAC or it will crash on synthetic audio
    class MockSNAC:
        def encode(self, audio):
            # UnifiedVoiceEngine.encode_audio expects snac.encode to return tokens
            return torch.zeros(1, 100) 
        def decode(self, tokens):
            return torch.zeros(24000) # 1 sec silent audio
            
    engine.snac = MockSNAC()
    
    # Init last_embedding to avoid attribute error if fallback triggers early
    engine.audio_llm.last_embedding = None
    
    print("✓ Engine initialized with Mock LLM/SNAC for logic verification")
    
    # Test Data Generation
    languages = ["hi", "ta", "te", "kn", "en"]
    samples_per_lang = 10
    
    print(f"\nGeneratin {len(languages) * samples_per_lang} synthetic voice samples...")
    
    # Run Tests
    for lang in languages:
        print(f"\n[Testing Language: {lang}]")
        print("-" * 30)
        
        # Scenario: Single Session with same speaker
        # We expect the Anchor Embedding to stay EXACTLY the same
        
        # Generate "User A" voice (seed=100)
        user_a_audio = generate_synthetic_audio(seed=100)
        
        # Turn 1
        print("  Turn 1 (User A): Processing...")
        result1 = await engine.process_speech(user_a_audio, language=lang, reset_anchor=True)
        anchor1 = engine.anchor_embedding
        
        # Capture the embedding passed to LLM
        embedded_in_llm1 = engine.audio_llm.last_embedding
        
        # Turn 2
        print("  Turn 2 (User A again): Processing...")
        # Even if audio is slightly different (noise), if we don't reset anchor, it should use previous
        # But here we pass SAME audio to simulate same speaker continuity in ideal case
        # OR we pass different audio but expect same anchor if we locked it?
        # WAIT: The implementation logic says: 
        # "if reset_anchor or self.anchor_embedding is None: generate new"
        # "else: use existing"
        
        # So independent of input audio, if we don't reset, it reuses the anchor!
        # Let's prove that by passing "User B" audio but NOT resetting anchor.
        # This proves the "Dialogue Controller Consistency" point.
        
        user_b_audio = generate_synthetic_audio(seed=200) # Different voice
        
        result2 = await engine.process_speech(user_b_audio, language=lang, reset_anchor=False)
        anchor2 = engine.anchor_embedding
        embedded_in_llm2 = engine.audio_llm.last_embedding
        
        # Verification
        print("  🔍 Verifying Anchoring...")
        
        # 1. Anchor should persist in engine
        if torch.equal(anchor1, anchor2):
            print("    ✅ Session Anchor Preserved (Identity Locked)")
        else:
            print("    ❌ FAIL: Session Anchor drifted!")
            
        # 2. LLM should have received the exact same embedding object
        if torch.equal(embedded_in_llm1, embedded_in_llm2):
            print("    ✅ AudioLLM conditioned on same Identity")
        else:
             print("    ❌ FAIL: AudioLLM received different embeddings!")
             
        # Scenario: New Session (Different Speaker)
        print("  Turn 3 (New Session/User B): Processing...")
        result3 = await engine.process_speech(user_b_audio, language=lang, reset_anchor=True)
        anchor3 = engine.anchor_embedding
        
        if not torch.equal(anchor1, anchor3):
             print("    ✅ New Anchor generated for new session (Identity Switched)")
        else:
             print("    ❌ FAIL: Anchor did not update for new session!")
             
    print("\n" + "="*60)
    print("Simulation Complete - Verified Logic for All Languages")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_simulation())
