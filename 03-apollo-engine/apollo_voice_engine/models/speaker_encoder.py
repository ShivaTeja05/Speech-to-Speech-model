"""
Speaker Encoder for Voice Identity Conditioning.

This module provides the mechanism to encode voice identity into a fixed-size embedding.
This embedding is used to condition the AudioLLM generation, ensuring voice consistency
and enabling "Session Anchoring".
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class SpeakerEncoder(nn.Module):
    """
    Encodes audio into a speaker embedding vector.
    
    In a production system, this would wrap a model like Resemblyzer, WavLM, or ECAPA-TDNN.
    For this implementation, we provide the interface and a learnable/random fallback 
    to validate the pipeline architecture.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 512, 
        device: str = "cpu",
        model_name: str = "mock_encoder"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.model_name = model_name
        
        # In a real scenario, load the model here
        # self.model = load_pretrained(model_name)
        
        logger.info(f"Initialized SpeakerEncoder (dim={embedding_dim}, device={device})")

    def forward(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Encode audio samples into a speaker embedding.
        
        Args:
            audio: Audio samples (shape: [Batch, Time] or [Time])
            
        Returns:
            Speaker embedding tensor (shape: [1, embedding_dim])
        """
        # Ensure input is a tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
            
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # MOCK IMPLEMENTATION:
        # Generate a deterministic embedding based on the audio content stats
        # This ensures that the same audio input produces the same "voice" embedding
        # but prevents us from needing a heavy external dependency for this demo phase.
        
        with torch.no_grad():
            # precise, deterministic "hashing" of audio features for the mock
            # In real usage: embedding = self.model(audio)
            
            # Simple feature extraction for mock: mean energy and zero-crossing rate concepts
            # mapped to the embedding dimension
            
            # Use a random generator seeded by the audio sum to get a consistent vector
            seed = int(torch.sum(torch.abs(audio)).item() * 1000) % 2**32
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            embedding = torch.randn(1, self.embedding_dim, generator=generator, device=self.device)
            
            # Normalize to unit length (common for speaker embeddings)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        return embedding

    def get_random_embedding(self) -> torch.Tensor:
        """Generate a random speaker embedding for testing."""
        embedding = torch.randn(1, self.embedding_dim, device=self.device)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)
