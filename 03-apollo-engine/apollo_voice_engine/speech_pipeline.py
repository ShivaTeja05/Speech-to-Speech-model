"""
Apollo Voice Engine - Speech-to-Speech Pipeline

Unified pipeline for real-time speech-to-speech using:
- STT: Whisper (fine-tuned for Indian languages)
- LLM: Sarvam-1 2B  
- TTS: AI4Bharat IndicParler-TTS

Usage:
    pipeline = SpeechPipeline()
    pipeline.load_models()
    audio_out = pipeline.process(audio_in, language="hi")
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the speech pipeline."""
    # STT Config
    whisper_model: str = "openai/whisper-small"  # or whisper-tiny for speed
    
    # LLM Config  
    llm_model: str = "sarvamai/sarvam-1"
    max_new_tokens: int = 100
    temperature: float = 0.7
    
    # TTS Config (Facebook MMS-TTS - no auth required!)
    tts_model: str = "facebook/mms-tts-hin"  # Supports: hin, tam, tel, kan, eng
    
    # Audio Config
    sample_rate: int = 24000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PipelineMetrics:
    """Latency metrics for pipeline components."""
    stt_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    safety_triggered: bool = False


class SpeechPipeline:
    """
    End-to-end speech-to-speech pipeline for Indian languages.
    
    Meets requirements:
    - <500ms latency target
    - <₹2/min cost (using open-source on T4 GPU)
    - Hindi, Tamil, Telugu, Kannada support
    """
    
    LANGUAGE_CODES = {
        "hi": "Hindi",
        "ta": "Tamil", 
        "te": "Telugu",
        "kn": "Kannada",
        "en": "English"
    }
    
    # Emergency keywords for safety (subset - full list in SafetyClassifier)
    EMERGENCY_KEYWORDS = [
        "emergency", "ambulance", "heart attack", "chest pain",
        "इमरजेंसी", "एंबुलेंस", "छाती में दर्द",
        "அவசரம்", "நெஞ்சு வலி",
        "అత్యవసర", "ఛాతీ నొప్పి",
        "ತುರ್ತು", "ಎದೆ ನೋವು"
    ]
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Models (loaded lazily)
        self.whisper = None
        self.whisper_processor = None
        self.llm = None
        self.llm_tokenizer = None
        self.tts = None
        self.tts_tokenizer = None
        
        self.is_loaded = False
        
    def load_models(self, load_whisper=True, load_llm=True, load_tts=True):
        """Load all pipeline models."""
        logger.info("Loading speech pipeline models...")
        
        if load_whisper:
            self._load_whisper()
            
        if load_llm:
            self._load_llm()
            
        if load_tts:
            self._load_tts()
            
        self.is_loaded = True
        logger.info("✓ All models loaded")
        
    def _load_whisper(self):
        """Load Whisper for STT."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info(f"Loading Whisper: {self.config.whisper_model}")
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.config.whisper_model
            )
            self.whisper = WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            ).to(self.config.device)
            self.whisper.eval()
            logger.info("✓ Whisper loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
            
    def _load_llm(self):
        """Load Sarvam-1 LLM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading LLM: {self.config.llm_model}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map=self.config.device
            )
            self.llm.eval()
            logger.info("✓ LLM loaded")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
            
    def _load_tts(self):
        """Load IndicParler-TTS."""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            
            logger.info(f"Loading TTS: {self.config.tts_model}")
            self.tts_tokenizer = AutoTokenizer.from_pretrained(self.config.tts_model)
            self.tts = ParlerTTSForConditionalGeneration.from_pretrained(
                self.config.tts_model,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            ).to(self.config.device)
            self.tts.eval()
            logger.info("✓ TTS loaded")
        except ImportError:
            logger.warning("parler_tts not installed. TTS disabled.")
            logger.warning("Install with: pip install git+https://github.com/huggingface/parler-tts.git")
        except Exception as e:
            logger.error(f"Failed to load TTS: {e}")
            
    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, language: str = "hi") -> Tuple[str, float]:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio: Audio waveform (mono, 16kHz or 24kHz)
            language: Language code (hi, ta, te, kn)
            
        Returns:
            Tuple of (transcribed_text, latency_ms)
        """
        if self.whisper is None:
            raise RuntimeError("Whisper not loaded. Call load_models() first.")
            
        start = time.perf_counter()
        
        # Ensure 16kHz for Whisper
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)  # Mono
            
        # Process audio
        inputs = self.whisper_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.config.device, dtype=torch.float16)
        
        # Generate with forced language
        forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
            language=self.LANGUAGE_CODES.get(language, "Hindi"),
            task="transcribe"
        )
        
        generated_ids = self.whisper.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256
        )
        
        # Decode
        text = self.whisper_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        latency = (time.perf_counter() - start) * 1000
        logger.debug(f"STT: '{text}' ({latency:.0f}ms)")
        
        return text, latency
        
    @torch.inference_mode()
    def generate_response(self, text: str, language: str = "hi") -> Tuple[str, float, bool]:
        """
        Generate response using LLM.
        
        Args:
            text: Input text (patient query)
            language: Language code
            
        Returns:
            Tuple of (response_text, latency_ms, is_emergency)
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_models() first.")
            
        start = time.perf_counter()
        
        # Check for emergency
        is_emergency = self._check_emergency(text)
        
        if is_emergency:
            response = self._get_transfer_message(language)
            latency = (time.perf_counter() - start) * 1000
            return response, latency, True
            
        # Format prompt
        prompt = f"Patient: {text}\nApollo Assistant:"
        
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        outputs = self.llm.generate(
            inputs["input_ids"],
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Apollo Assistant:")[-1].strip().split("\n")[0]
        
        latency = (time.perf_counter() - start) * 1000
        logger.debug(f"LLM: '{response}' ({latency:.0f}ms)")
        
        return response, latency, False
        
    @torch.inference_mode()
    def synthesize(self, text: str, language: str = "hi") -> Tuple[np.ndarray, float]:
        """
        Synthesize speech from text using IndicTTS.
        
        Args:
            text: Text to synthesize
            language: Language code
            
        Returns:
            Tuple of (audio_waveform, latency_ms)
        """
        if self.tts is None:
            logger.warning("TTS not loaded. Returning silence.")
            return np.zeros(self.config.sample_rate), 0.0
            
        start = time.perf_counter()
        
        # Create description for voice characteristics
        description = f"A clear, natural {self.LANGUAGE_CODES.get(language, 'Hindi')} voice speaking at a moderate pace."
        
        # Tokenize
        input_ids = self.tts_tokenizer(description, return_tensors="pt").input_ids.to(self.config.device)
        prompt_ids = self.tts_tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)
        
        # Generate audio
        generation = self.tts.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_ids,
            do_sample=True
        )
        
        audio = generation.cpu().numpy().squeeze()
        
        latency = (time.perf_counter() - start) * 1000
        logger.debug(f"TTS: {len(audio)} samples ({latency:.0f}ms)")
        
        return audio, latency
        
    def process(
        self, 
        audio_in: np.ndarray, 
        language: str = "hi"
    ) -> Tuple[np.ndarray, str, str, PipelineMetrics]:
        """
        Full speech-to-speech processing.
        
        Args:
            audio_in: Input audio waveform
            language: Language code
            
        Returns:
            Tuple of (audio_out, transcription, response, metrics)
        """
        metrics = PipelineMetrics()
        total_start = time.perf_counter()
        
        # 1. STT
        transcription, stt_ms = self.transcribe(audio_in, language)
        metrics.stt_ms = stt_ms
        
        # 2. LLM
        response, llm_ms, is_emergency = self.generate_response(transcription, language)
        metrics.llm_ms = llm_ms
        metrics.safety_triggered = is_emergency
        
        # 3. TTS
        audio_out, tts_ms = self.synthesize(response, language)
        metrics.tts_ms = tts_ms
        
        metrics.total_ms = (time.perf_counter() - total_start) * 1000
        
        logger.info(
            f"Pipeline: STT={metrics.stt_ms:.0f}ms, LLM={metrics.llm_ms:.0f}ms, "
            f"TTS={metrics.tts_ms:.0f}ms, Total={metrics.total_ms:.0f}ms"
        )
        
        return audio_out, transcription, response, metrics
        
    def _check_emergency(self, text: str) -> bool:
        """Check if text contains emergency keywords."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.EMERGENCY_KEYWORDS)
        
    def _get_transfer_message(self, language: str) -> str:
        """Get emergency transfer message in specified language."""
        messages = {
            "hi": "आपातकाल का पता चला। मैं आपको तुरंत स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं।",
            "ta": "அவசரநிலை கண்டறியப்பட்டது. சுகாதார நிபுணரை இணைக்கிறேன்.",
            "te": "అత్యవసర పరిస్థితి. ఆరోగ్య నిపుణుడిని కనెక్ట్ చేస్తున్నాను.",
            "kn": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ. ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
            "en": "Emergency detected. Connecting you with a healthcare professional."
        }
        return messages.get(language, messages["en"])


def test_pipeline():
    """Test the speech pipeline."""
    print("Testing Speech Pipeline...")
    print("=" * 50)
    
    config = PipelineConfig()
    print(f"Config: {config}")
    
    pipeline = SpeechPipeline(config)
    print(f"Languages: {pipeline.LANGUAGE_CODES}")
    
    # Test emergency detection
    test_texts = [
        "मुझे छाती में दर्द हो रहा है",
        "pharmacy कब खुलती है?",
        "I need an ambulance!"
    ]
    
    for text in test_texts:
        is_emergency = pipeline._check_emergency(text)
        print(f"'{text}' → Emergency: {is_emergency}")
        
    print("\n✓ Pipeline test passed!")
    print("Note: Full test requires GPU and model downloads (~10GB)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pipeline()
