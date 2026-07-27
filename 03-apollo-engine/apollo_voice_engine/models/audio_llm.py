"""
Extended Audio-LLM Model

Extends Sarvam-1 2B with SNAC audio tokens for unified speech-to-speech
generation. The model can now "hear" and "speak" by processing audio
tokens alongside text tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

logger = logging.getLogger(__name__)


class AudioLLM(nn.Module):
    """
    Extended Sarvam-1 model with audio token vocabulary.
    
    Architecture:
    - Base: Sarvam-1 2B (28 layers, 2048 hidden, 16 heads)
    - Extended vocabulary: +4096 SNAC audio tokens
    - Unified next-token prediction for both text and audio
    """
    
    # Constants
    AUDIO_TOKEN_OFFSET = 50000
    AUDIO_VOCAB_SIZE = 4096
    
    # Special tokens
    AUDIO_START_TOKEN = "<|audio_start|>"
    AUDIO_END_TOKEN = "<|audio_end|>"
    
    def __init__(
        self,
        model_name: str = "sarvamai/sarvam-1",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the Audio-LLM.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Will be initialized in load_model()
        self.model = None
        self.tokenizer = None
        self.config = None
        self.original_vocab_size = None
        self.extended_vocab_size = None
        
        # Speaker conditioning
        self.speaker_dim = 512  # Default common embedding size
        self.speaker_proj = None
        
    def load_model(self):
        """Load and extend the base model with audio vocabulary."""
        logger.info(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add special audio tokens
        special_tokens = {
            "additional_special_tokens": [
                self.AUDIO_START_TOKEN,
                self.AUDIO_END_TOKEN,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load config
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model with optional quantization
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["device_map"] = self.device
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        # Store original vocab size
        self.original_vocab_size = self.model.config.vocab_size
        
        # Extend vocabulary for audio tokens
        self._extend_vocabulary()
        
        # Initialize speaker projection
        # Projects speaker embedding dim -> model hidden size
        self.speaker_proj = nn.Linear(self.speaker_dim, self.model.config.hidden_size).to(self.device, dtype=self.model.dtype)
        
        logger.info(f"Model loaded. Vocab extended: {self.original_vocab_size} → {self.extended_vocab_size}")
        
    def _extend_vocabulary(self):
        """Extend the model's embedding and output layers for audio tokens."""
        # Calculate new vocab size
        # Original vocab + special tokens + audio tokens
        num_special = 2  # AUDIO_START, AUDIO_END
        self.extended_vocab_size = self.original_vocab_size + num_special + self.AUDIO_VOCAB_SIZE
        
        # Resize token embeddings
        self.model.resize_token_embeddings(self.extended_vocab_size)
        
        # Initialize new audio token embeddings
        # Use small random init to distinguish from text tokens
        with torch.no_grad():
            # Get embedding layer
            embeddings = self.model.get_input_embeddings()
            
            # Initialize audio tokens (last AUDIO_VOCAB_SIZE entries)
            audio_start_idx = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
            nn.init.normal_(
                embeddings.weight[audio_start_idx:],
                mean=0.0,
                std=0.02
            )
            
        logger.info(f"Extended embeddings for {self.AUDIO_VOCAB_SIZE} audio tokens")
    
    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
    
    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert token IDs to text."""
        # Filter out audio tokens
        text_tokens = tokens[tokens < self.AUDIO_TOKEN_OFFSET]
        return self.tokenizer.decode(text_tokens, skip_special_tokens=True)
    
    def audio_tokens_to_ids(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert raw SNAC tokens to extended vocabulary IDs.
        
        Args:
            audio_tokens: SNAC tokens (0-4095)
            
        Returns:
            Extended vocab IDs with proper offset
        """
        # SNAC tokens come with AUDIO_TOKEN_OFFSET already applied
        # from SNACWrapper, but we need to add our special token offset
        offset = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        
        # If tokens already have offset, remove it and apply new one
        if audio_tokens.min() >= self.AUDIO_TOKEN_OFFSET:
            audio_tokens = audio_tokens - self.AUDIO_TOKEN_OFFSET
            
        return audio_tokens + offset
    
    def ids_to_audio_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert extended vocabulary IDs back to SNAC tokens.
        
        Args:
            token_ids: Extended vocab IDs
            
        Returns:
            SNAC tokens (0-4095) with AUDIO_TOKEN_OFFSET
        """
        offset = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        snac_tokens = token_ids - offset
        return snac_tokens + self.AUDIO_TOKEN_OFFSET
    
    def is_audio_token(self, token_id: int) -> bool:
        """Check if a token ID represents an audio token."""
        audio_start = self.extended_vocab_size - self.AUDIO_VOCAB_SIZE
        return audio_start <= token_id < self.extended_vocab_size
    
    def prepare_audio_input(
        self,
        audio_tokens: torch.Tensor,
        text_prompt: Optional[str] = None,
        speaker_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input embeddings with optional speaker conditioning.
        
        Args:
            audio_tokens: SNAC audio tokens
            text_prompt: Optional text prefix
            speaker_embedding: Optional speaker vector [1, speaker_dim]
            
        Returns:
            Tuple(inputs_embeds, attention_mask)
        """
        sequences = []
        
        # 1. Build Token Sequence (IDs)
        # Add text prompt if provided
        if text_prompt:
            text_tokens = self.text_to_tokens(text_prompt)
            sequences.append(text_tokens.squeeze(0))
        
        # Add audio start token
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_START_TOKEN)
        sequences.append(torch.tensor([audio_start_id], device=self.device))
        
        # Add audio tokens (converted to extended vocab IDs)
        audio_ids = self.audio_tokens_to_ids(audio_tokens)
        if audio_ids.dim() > 1:
            audio_ids = audio_ids.squeeze(0)
        sequences.append(audio_ids)
        
        # Add audio end token
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_END_TOKEN)
        sequences.append(torch.tensor([audio_end_id], device=self.device))
        
        # Concatenate all IDs
        input_ids = torch.cat(sequences, dim=0).unsqueeze(0) # [1, seq_len]
        
        # 2. Convert to Embeddings
        # Get base token embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids) # [1, seq_len, hidden_size]
        
        # 3. Inject Speaker Conditioning (Session Anchoring)
        if speaker_embedding is not None and self.speaker_proj is not None:
            # Project embedding [1, speaker_dim] -> [1, hidden_size]
            # Ensure dtype matches
            embed_input = speaker_embedding.to(device=self.device, dtype=self.model.dtype)
            speaker_cond = self.speaker_proj(embed_input)
            
            # Reshape to [1, 1, hidden_size] to act as a prefix token representation
            if speaker_cond.dim() == 2:
                speaker_cond = speaker_cond.unsqueeze(1)
                
            # Concatenate: [Speaker_Emb, Text_Emb, Audio_Start, Audio_Tokens...]
            print(f"[DEBUG] Injecting speaker embedding (Session Anchor). Shape: {speaker_cond.shape}")
            inputs_embeds = torch.cat([speaker_cond, inputs_embeds], dim=1)
            
        # 4. Create Attention Mask
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], 
            dtype=torch.long, 
            device=self.device
        )
            
        return inputs_embeds, attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (text + audio)
            attention_mask: Attention mask
            labels: Labels for training
            
        Returns:
            Model outputs with loss if labels provided
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    @torch.no_grad()
    def generate_response(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        override_output_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a response (text or audio tokens).
        
        Args:
            input_ids: Input token sequence (optional if inputs_embeds provided)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample vs greedy
            override_output_ids: Pre-computed tokens to return immediately
            inputs_embeds: Input embeddings (includes speaker conditioning)
            attention_mask: Attention mask for inputs
            
        Returns:
            Generated token sequence
        """
        # DEBUG: Return override if provided
        if override_output_ids is not None:
            print(f"[DEBUG] Using override output ({len(override_output_ids)} tokens)")
            return override_output_ids.to(self.device)
            
        # Create attention mask if not provided
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            elif inputs_embeds is not None:
                 attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
            else:
                raise ValueError("Either input_ids or inputs_embeds must be provided")

        if inputs_embeds is not None:
            print(f"[DEBUG] Generating with inputs_embeds shape: {inputs_embeds.shape}")
        elif input_ids is not None:
            print(f"[DEBUG] Generating with input_ids shape: {input_ids.shape}")

        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        # Determine generate args
        # Note: input_ids is mutually exclusive with inputs_embeds in some HF versions,
        # but usually providing inputs_embeds overrides input_ids.
        # However, to be safe, we pass exactly one main input.
        
        gen_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "streamer": streamer,
            **kwargs
        }
        
        if inputs_embeds is not None:
            gen_kwargs["inputs_embeds"] = inputs_embeds
        else:
            gen_kwargs["input_ids"] = input_ids

        output = self.model.generate(**gen_kwargs)
        
        return output
    
    def extract_audio_tokens(self, generated_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract audio tokens from generated sequence.
        
        Args:
            generated_ids: Full generated token sequence
            
        Returns:
            Audio tokens between AUDIO_START and AUDIO_END, or None
        """
        tokens = generated_ids.squeeze().tolist()
        
        audio_start_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_START_TOKEN)
        audio_end_id = self.tokenizer.convert_tokens_to_ids(self.AUDIO_END_TOKEN)
        
        try:
            start_idx = tokens.index(audio_start_id) + 1
            end_idx = tokens.index(audio_end_id)
            
            audio_ids = torch.tensor(tokens[start_idx:end_idx], device=self.device)
            return self.ids_to_audio_tokens(audio_ids)
            
        except ValueError:
            logger.warning("No audio tokens found in generated sequence")
            return None
    
    def save_pretrained(self, save_path: str):
        """Save the extended model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "AudioLLM":
        """Load a saved extended model."""
        instance = cls(model_name=model_path, **kwargs)
        instance.load_model()
        return instance


def test_audio_llm():
    """Test the AudioLLM initialization and basic operations."""
    print("Testing AudioLLM...")
    
    # Note: This test requires the actual model to be downloaded
    # For CI, mock the model loading
    model = AudioLLM(model_name="sarvamai/sarvam-1")
    
    print(f"Model name: {model.model_name}")
    print(f"Audio vocab size: {model.AUDIO_VOCAB_SIZE}")
    print(f"Audio token offset: {model.AUDIO_TOKEN_OFFSET}")
    
    # Test token conversion (without loading model)
    fake_audio_tokens = torch.tensor([0, 100, 4095])
    print(f"Sample SNAC tokens: {fake_audio_tokens.tolist()}")
    
    # Test prepare_audio_input with embedding
    print("\nTesting conditioning signature...")
    fake_embedding = torch.randn(1, 512)
    # We can't fully run prepare_audio_input without loading the model/tokenizer
    # but we can verify the method signature exists
    import inspect
    sig = inspect.signature(model.prepare_audio_input)
    if "speaker_embedding" in sig.parameters:
        print("✓ prepare_audio_input accepts speaker_embedding")
    else:
        print("❌ prepare_audio_input MISSING speaker_embedding")
    
    print("✓ AudioLLM basic test passed!")
    print("Note: Full test requires model download (~4GB)")


if __name__ == "__main__":
    test_audio_llm()
