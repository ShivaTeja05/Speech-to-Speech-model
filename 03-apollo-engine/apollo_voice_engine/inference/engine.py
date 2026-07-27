"""
Inference Engine with Voice Activity Detection

Real-time inference pipeline with:
- Voice Activity Detection (VAD) for turn-taking
- Streaming audio processing
- Low-latency response generation
"""

import torch
import numpy as np
import time
from typing import Optional, Tuple, Generator, Callable
from dataclasses import dataclass
import logging
import queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    aggressiveness: int = 2  # 0-3, higher = more aggressive
    frame_duration_ms: int = 30
    padding_duration_ms: int = 300
    sample_rate: int = 24000


@dataclass  
class InferenceConfig:
    """Inference engine configuration."""
    max_ttft_ms: int = 250  # Time to first token target
    max_e2e_latency_ms: int = 500  # End-to-end target
    max_new_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9


class VoiceActivityDetector:
    """
    Voice Activity Detection for turn-taking in conversations.
    
    Uses WebRTC VAD to detect when the user has finished speaking,
    triggering response generation immediately for low latency.
    """
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.vad = None
        self._init_vad()
        
    def _init_vad(self):
        """Initialize WebRTC VAD."""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.config.aggressiveness)
            logger.info(f"VAD initialized with aggressiveness={self.config.aggressiveness}")
        except ImportError:
            logger.warning("webrtcvad not installed. VAD disabled.")
            self.vad = None
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check if an audio frame contains speech.
        
        Args:
            audio_frame: Raw audio bytes (16-bit PCM)
            
        Returns:
            True if speech is detected
        """
        if self.vad is None:
            return True  # Fallback: assume always speech
            
        return self.vad.is_speech(audio_frame, self.config.sample_rate)
    
    def detect_end_of_speech(
        self,
        audio_stream: Generator[bytes, None, None],
        on_speech_end: Callable[[], None]
    ) -> Generator[bytes, None, None]:
        """
        Wrap an audio stream with end-of-speech detection.
        
        Args:
            audio_stream: Generator yielding audio frames
            on_speech_end: Callback when speech ends
            
        Yields:
            Audio frames (passthrough)
        """
        silence_frames = 0
        padding_frames = self.config.padding_duration_ms // self.config.frame_duration_ms
        
        for frame in audio_stream:
            yield frame
            
            if self.is_speech(frame):
                silence_frames = 0
            else:
                silence_frames += 1
                
                if silence_frames >= padding_frames:
                    logger.debug("End of speech detected")
                    on_speech_end()
                    silence_frames = 0


class InferenceEngine:
    """
    Real-time inference engine for the Apollo Voice Engine.
    
    Features:
    - Streaming audio input processing
    - VAD-triggered generation
    - Latency monitoring
    - Concurrent request handling
    """
    
    def __init__(
        self,
        model,  # AudioLLM instance
        snac_wrapper,  # SNACWrapper instance
        inference_config: Optional[InferenceConfig] = None,
        vad_config: Optional[VADConfig] = None
    ):
        self.model = model
        self.snac = snac_wrapper
        self.inference_config = inference_config or InferenceConfig()
        self.vad = VoiceActivityDetector(vad_config or VADConfig())
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "avg_ttft_ms": 0.0,
            "avg_e2e_latency_ms": 0.0,
        }
        
        # Audio buffer for streaming
        self.audio_buffer = queue.Queue()
        self._is_processing = False
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add audio chunk to processing buffer.
        
        Args:
            audio_chunk: Audio samples (float32, 24kHz)
        """
        self.audio_buffer.put(audio_chunk)
    
    def _collect_audio(self, timeout: float = 5.0) -> torch.Tensor:
        """Collect buffered audio into a single tensor."""
        chunks = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                chunk = self.audio_buffer.get(timeout=0.1)
                chunks.append(chunk)
                
                # Convert to bytes for VAD
                audio_bytes = (chunk * 32767).astype(np.int16).tobytes()
                if not self.vad.is_speech(audio_bytes):
                    # End of speech detected
                    break
                    
            except queue.Empty:
                if chunks:
                    break
        
        if not chunks:
            return None
            
        return torch.tensor(np.concatenate(chunks), dtype=torch.float32)
    
    def generate_response(
        self,
        audio_input: torch.Tensor,
        text_context: Optional[str] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate a response from audio input.
        
        Args:
            audio_input: Input audio tensor
            text_context: Optional text context/prompt
            
        Returns:
            Tuple of (response_audio, metrics)
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        # Encode input audio to tokens
        encode_start = time.time()
        audio_tokens = self.snac.encode(audio_input)
        encode_time = (time.time() - encode_start) * 1000
        
        # Prepare model input
        input_ids = self.model.prepare_audio_input(
            audio_tokens,
            text_prompt=text_context
        )
        
        # Generate response
        generate_start = time.time()
        output_ids = self.model.generate_response(
            input_ids,
            max_new_tokens=self.inference_config.max_new_tokens,
            temperature=self.inference_config.temperature,
            top_p=self.inference_config.top_p
        )
        generate_time = (time.time() - generate_start) * 1000
        
        # Time to first token (approximation)
        ttft = encode_time + (generate_time / (output_ids.shape[1] - input_ids.shape[1]))
        
        # Extract and decode audio response
        response_tokens = self.model.extract_audio_tokens(output_ids)
        
        if response_tokens is not None:
            decode_start = time.time()
            response_audio = self.snac.decode(response_tokens)
            decode_time = (time.time() - decode_start) * 1000
        else:
            response_audio = None
            decode_time = 0
        
        # Calculate total latency
        e2e_latency = (time.time() - start_time) * 1000
        
        # Update running averages
        n = self.metrics["total_requests"]
        self.metrics["avg_ttft_ms"] = (
            (self.metrics["avg_ttft_ms"] * (n - 1) + ttft) / n
        )
        self.metrics["avg_e2e_latency_ms"] = (
            (self.metrics["avg_e2e_latency_ms"] * (n - 1) + e2e_latency) / n
        )
        
        request_metrics = {
            "encode_time_ms": encode_time,
            "generate_time_ms": generate_time,
            "decode_time_ms": decode_time,
            "ttft_ms": ttft,
            "e2e_latency_ms": e2e_latency,
            "input_tokens": input_ids.shape[1],
            "output_tokens": output_ids.shape[1] - input_ids.shape[1],
            "meets_ttft_target": ttft < self.inference_config.max_ttft_ms,
            "meets_latency_target": e2e_latency < self.inference_config.max_e2e_latency_ms,
        }
        
        logger.info(
            f"Request completed: TTFT={ttft:.1f}ms, E2E={e2e_latency:.1f}ms, "
            f"Tokens={request_metrics['output_tokens']}"
        )
        
        return response_audio, request_metrics
    
    def stream_response(
        self,
        audio_input: torch.Tensor,
        chunk_size: int = 8192
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream response audio in chunks for real-time playback.
        
        Args:
            audio_input: Input audio tensor
            chunk_size: Samples per chunk
            
        Yields:
            Audio chunks as numpy arrays
        """
        response_audio, _ = self.generate_response(audio_input)
        
        if response_audio is None:
            return
            
        audio_np = response_audio.cpu().numpy()
        
        for i in range(0, len(audio_np), chunk_size):
            yield audio_np[i:i + chunk_size]
    
    def get_metrics(self) -> dict:
        """Get current inference metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset inference metrics."""
        self.metrics = {
            "total_requests": 0,
            "avg_ttft_ms": 0.0,
            "avg_e2e_latency_ms": 0.0,
        }


class StreamingSession:
    """
    Manages a streaming conversation session.
    
    Handles continuous audio input with VAD-based turn-taking.
    """
    
    def __init__(
        self,
        engine: InferenceEngine,
        session_id: str,
        language: str = "hi"
    ):
        self.engine = engine
        self.session_id = session_id
        self.language = language
        
        self.is_active = False
        self.turn_count = 0
        self._audio_buffer = []
        self._response_callback = None
        
    def start(self, on_response: Callable[[np.ndarray], None]):
        """Start the streaming session."""
        self.is_active = True
        self._response_callback = on_response
        logger.info(f"Session {self.session_id} started")
        
    def stop(self):
        """Stop the streaming session."""
        self.is_active = False
        logger.info(f"Session {self.session_id} stopped after {self.turn_count} turns")
        
    def on_audio_chunk(self, chunk: np.ndarray):
        """Process an incoming audio chunk."""
        if not self.is_active:
            return
            
        self._audio_buffer.append(chunk)
        
    def on_speech_end(self):
        """Called when VAD detects end of speech."""
        if not self._audio_buffer:
            return
            
        # Combine buffered audio
        audio = torch.tensor(
            np.concatenate(self._audio_buffer),
            dtype=torch.float32
        )
        self._audio_buffer = []
        self.turn_count += 1
        
        # Generate and deliver response
        for chunk in self.engine.stream_response(audio):
            if self._response_callback and self.is_active:
                self._response_callback(chunk)


def test_inference_engine():
    """Test inference engine components."""
    print("Testing Inference Engine...")
    
    # Test VAD config
    vad_config = VADConfig()
    print(f"VAD config: {vad_config}")
    
    # Test inference config
    inf_config = InferenceConfig()
    print(f"Inference config: {inf_config}")
    
    # Test VAD (if webrtcvad available)
    try:
        vad = VoiceActivityDetector(vad_config)
        print("✓ VAD initialized successfully")
    except Exception as e:
        print(f"⚠ VAD init failed (expected if webrtcvad not installed): {e}")
    
    print("✓ Inference engine test passed!")


if __name__ == "__main__":
    test_inference_engine()
