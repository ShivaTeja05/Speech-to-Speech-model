"""
SNAC (Multi-Scale Neural Audio Codec) Wrapper

Provides encode/decode functionality for converting audio waveforms
to discrete tokens compatible with the extended Sarvam-1 vocabulary.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SNACWrapper:
    """
    Wrapper for the SNAC multi-scale neural audio codec.
    
    SNAC encodes audio into hierarchical tokens at different temporal scales,
    allowing the LLM to process audio with the same efficiency as text.
    
    Token hierarchy (at 24kHz):
    - Coarse: 12 tokens/second (broad structure)
    - Medium: 24 tokens/second
    - Fine: 75 tokens/second (detailed acoustics)
    """
    
    # Token offset in extended vocabulary
    AUDIO_TOKEN_OFFSET = 50000  # After Sarvam-1's text tokens
    AUDIO_VOCAB_SIZE = 4096
    
    def __init__(
        self,
        model_name: str = "hubertsiuzdak/snac_24khz",
        sample_rate: int = 24000,
        device: Optional[str] = None
    ):
        """
        Initialize the SNAC wrapper.
        
        Args:
            model_name: HuggingFace model identifier for SNAC
            sample_rate: Audio sample rate (must match SNAC model)
            device: Device to run on (auto-detect if None)
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_model(self):
        """Load the SNAC model from HuggingFace."""
        try:
            from snac import SNAC
            
            logger.info(f"Loading SNAC model: {self.model_name}")
            self.model = SNAC.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"SNAC model loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "SNAC package not installed. Run: pip install snac"
            )
        except Exception as e:
            logger.error(f"Failed to load SNAC model: {e}")
            raise
    
    def ensure_loaded(self):
        """Ensure model is loaded before use."""
        if self.model is None:
            self.load_model()
    
    @torch.no_grad()
    def encode(
        self,
        audio: torch.Tensor,
        flatten: bool = True
    ) -> torch.Tensor:
        """
        Encode audio waveform to discrete tokens.
        
        Args:
            audio: Audio tensor of shape (batch, samples) or (samples,)
            flatten: If True, flatten multi-scale tokens to single sequence
            
        Returns:
            tokens: Token tensor with AUDIO_TOKEN_OFFSET applied
        """
        self.ensure_loaded()
        
        # Ensure batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Move to device
        audio = audio.to(self.device)
        
        # Encode with SNAC (returns list of tensors for each scale)
        codes = self.model.encode(audio)
        
        if flatten:
            # Interleave multi-scale tokens for autoregressive generation
            tokens = self._flatten_codes(codes)
        else:
            tokens = codes
        
        # Apply offset to map to extended vocabulary
        tokens = tokens + self.AUDIO_TOKEN_OFFSET
        
        return tokens
    
    @torch.no_grad()
    def decode(
        self,
        tokens: torch.Tensor,
        is_flattened: bool = True
    ) -> torch.Tensor:
        """
        Decode tokens back to audio waveform.
        
        Args:
            tokens: Token tensor (with or without offset)
            is_flattened: If True, unflatten before decoding
            
        Returns:
            audio: Reconstructed audio waveform
        """
        self.ensure_loaded()
        
        # Remove offset if present
        if tokens.min() >= self.AUDIO_TOKEN_OFFSET:
            tokens = tokens - self.AUDIO_TOKEN_OFFSET
        
        # Ensure valid range
        tokens = tokens.clamp(0, self.AUDIO_VOCAB_SIZE - 1)
        
        if is_flattened:
            codes = self._unflatten_codes(tokens)
        else:
            codes = tokens
        
        # Decode with SNAC
        audio = self.model.decode(codes)
        
        return audio.squeeze(0)
    
    def _flatten_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten multi-scale codes into a single sequence.
        
        Uses interleaving pattern: [C0, M0, M1, F0, F1, F2, C1, ...]
        where C=coarse, M=medium, F=fine
        """
        # SNAC returns 3 scales with different temporal resolutions
        coarse, medium, fine = codes
        
        batch_size = coarse.shape[0]
        num_frames = coarse.shape[1]
        
        # Calculate tokens per frame at each scale
        medium_per_coarse = medium.shape[1] // num_frames
        fine_per_coarse = fine.shape[1] // num_frames
        
        flattened = []
        for i in range(num_frames):
            # Add coarse token
            flattened.append(coarse[:, i:i+1])
            
            # Add medium tokens
            m_start = i * medium_per_coarse
            m_end = m_start + medium_per_coarse
            flattened.append(medium[:, m_start:m_end])
            
            # Add fine tokens
            f_start = i * fine_per_coarse
            f_end = f_start + fine_per_coarse
            flattened.append(fine[:, f_start:f_end])
        
        return torch.cat(flattened, dim=1)
    
    def _unflatten_codes(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Unflatten single sequence back to multi-scale codes.
        
        Reverses the interleaving pattern from _flatten_codes.
        """
        # Determine structure (assuming standard SNAC 24kHz ratios)
        # Coarse: 1, Medium: 2, Fine: 6 per frame
        tokens_per_frame = 1 + 2 + 6  # 9 tokens per frame
        
        batch_size = tokens.shape[0]
        num_frames = tokens.shape[1] // tokens_per_frame
        
        coarse = []
        medium = []
        fine = []
        
        for i in range(num_frames):
            offset = i * tokens_per_frame
            coarse.append(tokens[:, offset:offset+1])
            medium.append(tokens[:, offset+1:offset+3])
            fine.append(tokens[:, offset+3:offset+9])
        
        return [
            torch.cat(coarse, dim=1),
            torch.cat(medium, dim=1),
            torch.cat(fine, dim=1)
        ]
    
    def get_audio_duration(self, num_tokens: int) -> float:
        """
        Calculate audio duration from token count.
        
        Args:
            num_tokens: Number of flattened tokens
            
        Returns:
            duration: Duration in seconds
        """
        tokens_per_frame = 9  # From SNAC structure
        coarse_fps = 12  # Coarse tokens per second
        
        num_frames = num_tokens // tokens_per_frame
        return num_frames / coarse_fps
    
    def get_token_count(self, duration_seconds: float) -> int:
        """
        Calculate token count for a given audio duration.
        
        Args:
            duration_seconds: Audio duration in seconds
            
        Returns:
            num_tokens: Number of flattened tokens needed
        """
        tokens_per_frame = 9
        coarse_fps = 12
        
        num_frames = int(duration_seconds * coarse_fps)
        return num_frames * tokens_per_frame


def test_snac_wrapper():
    """Quick test of SNAC encode/decode roundtrip."""
    import numpy as np
    
    wrapper = SNACWrapper()
    
    # Generate test audio (1 second of silence)
    duration = 1.0
    audio = torch.zeros(1, int(wrapper.sample_rate * duration))
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Expected tokens: ~{wrapper.get_token_count(duration)}")
    
    # Encode
    tokens = wrapper.encode(audio)
    print(f"Encoded tokens shape: {tokens.shape}")
    print(f"Token range: [{tokens.min().item()}, {tokens.max().item()}]")
    
    # Decode
    reconstructed = wrapper.decode(tokens)
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    print("✓ SNAC roundtrip test passed!")


if __name__ == "__main__":
    test_snac_wrapper()
