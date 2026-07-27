"""
Tests for SNAC Wrapper
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch


class TestSNACWrapper:
    """Test suite for SNACWrapper."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        wrapper = SNACWrapper()
        
        assert wrapper.model_name == "hubertsiuzdak/snac_24khz"
        assert wrapper.sample_rate == 24000
        assert wrapper.model is None  # Not loaded yet
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        wrapper = SNACWrapper(
            model_name="custom/model",
            sample_rate=16000,
            device="cpu"
        )
        
        assert wrapper.model_name == "custom/model"
        assert wrapper.sample_rate == 16000
        assert wrapper.device == "cpu"
    
    def test_audio_token_constants(self):
        """Test audio token constants are correct."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        wrapper = SNACWrapper()
        
        assert wrapper.AUDIO_TOKEN_OFFSET == 50000
        assert wrapper.AUDIO_VOCAB_SIZE == 4096
    
    def test_get_audio_duration(self):
        """Test duration calculation from token count."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        wrapper = SNACWrapper()
        
        # 9 tokens per frame, 12 frames per second
        # 108 tokens = 12 frames = 1 second
        duration = wrapper.get_audio_duration(108)
        assert duration == 1.0
        
        # 216 tokens = 24 frames = 2 seconds
        duration = wrapper.get_audio_duration(216)
        assert duration == 2.0
    
    def test_get_token_count(self):
        """Test token count calculation from duration."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        wrapper = SNACWrapper()
        
        # 1 second = 12 frames * 9 tokens = 108 tokens
        tokens = wrapper.get_token_count(1.0)
        assert tokens == 108
        
        # 5 seconds = 60 frames * 9 tokens = 540 tokens
        tokens = wrapper.get_token_count(5.0)
        assert tokens == 540


class TestSNACWrapperMocked:
    """Tests requiring mocked SNAC model."""
    
    @patch('apollo_voice_engine.models.snac_wrapper.SNAC')
    def test_load_model(self, mock_snac_class):
        """Test model loading with mock."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        # Setup mock
        mock_model = Mock()
        mock_snac_class.from_pretrained.return_value = mock_model
        
        wrapper = SNACWrapper(device="cpu")
        wrapper.load_model()
        
        mock_snac_class.from_pretrained.assert_called_once_with(
            "hubertsiuzdak/snac_24khz"
        )
        assert wrapper.model is not None
    
    @patch('apollo_voice_engine.models.snac_wrapper.SNAC')
    def test_encode_with_offset(self, mock_snac_class):
        """Test that encoding applies token offset."""
        from apollo_voice_engine.models.snac_wrapper import SNACWrapper
        
        # Setup mock to return fake codes
        mock_model = Mock()
        mock_model.encode.return_value = [
            torch.tensor([[0, 1, 2]]),  # coarse
            torch.tensor([[0, 1, 0, 1, 0, 1]]),  # medium
            torch.tensor([[0, 1, 2, 3, 4, 5] * 3]),  # fine
        ]
        mock_snac_class.from_pretrained.return_value = mock_model
        
        wrapper = SNACWrapper(device="cpu")
        wrapper.load_model()
        
        audio = torch.zeros(1, 24000)  # 1 second
        tokens = wrapper.encode(audio)
        
        # All tokens should have offset applied
        assert tokens.min().item() >= wrapper.AUDIO_TOKEN_OFFSET


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
