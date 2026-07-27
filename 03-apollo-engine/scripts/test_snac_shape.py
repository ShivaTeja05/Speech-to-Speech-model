
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apollo_voice_engine.models.snac_wrapper import SNACWrapper

def test_shape_error():
    print("Initializing SNAC Wrapper (Mocking load for speed test)...")
    # We rely on the real installed 'snac' library, so we must load it.
    wrapper = SNACWrapper(device="cpu")
    wrapper.load_model()
    
    # 1. Simulate the problematic shape: (1, 88064) which is just (Batch, Time)
    # The error suggests this is being interpreted as (Batch, Channels, Time) or something
    # that makes channels = 88064.
    
    # User had: tensor of size 88064
    T = 88064 # Corresponds to ~3.6s at 24kHz
    
    print(f"\nTest 1: Encoding (1, {T}) - implicit [B, T]")
    try:
        audio = torch.randn(1, T)
        # unified_demo.py does: if dim==1: unsqueeze(0). BUT if dim==2 (1, T), it leaves it.
        # So we pass (1, T) to wrapper.encode
        wrapper.encode(audio)
        print("✅ Test 1 Passed (Surprising!)")
    except RuntimeError as e:
        print(f"❌ Test 1 Failed as expected: {e}")
        
    print(f"\nTest 2: Encoding (1, 1, {T}) - explicit [B, C, T]")
    try:
        audio = torch.randn(1, 1, T)
        wrapper.encode(audio)
        print("✅ Test 2 Passed")
    except RuntimeError as e:
        print(f"❌ Test 2 Failed: {e}")

if __name__ == "__main__":
    test_shape_error()
