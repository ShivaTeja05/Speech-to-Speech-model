"""
Apollo Hospital Voice AI Assistant - FastAPI Backend v2.0
Real-time regional-language voice AI for patient health queries
Supports: Tamil, Telugu, Kannada, Hindi
Target: <500ms latency, <₹2/min cost

v2.0 Features:
- Redis-backed session management
- Admin panel with config/doctors/FAQs CRUD
- RAG retrieval for context-aware responses
- Multi-turn conversation memory
- Enhanced safety gate with 5 escalation rules
"""

import os
import io
import time
import json
import base64
import tempfile
import logging
import uuid
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import fasttext  # Added for FastText support

# Import our models
from models.redis_store import get_redis, RedisStore
from models.conversation import ConversationManager, ContextExtractor
from models.embeddings import RAGRetriever, load_documents_from_json, load_documents_from_redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_LANGUAGES = {
    'kn': 'Kannada',
    'ta': 'Tamil', 
    'te': 'Telugu',
    'hi': 'Hindi',
    'en': 'English'
}

# Unicode script ranges for language detection
SCRIPT_RANGES = {
    'kn': [(0x0C80, 0x0CFF)],  # Kannada
    'ta': [(0x0B80, 0x0BFF)],  # Tamil
    'te': [(0x0C00, 0x0C7F)],  # Telugu
    'hi': [(0x0900, 0x097F)],  # Hindi/Devanagari
    'en': [(0x0041, 0x005A), (0x0061, 0x007A)],  # English A-Z, a-z
}

# Multi-language keyword dictionaries for Layer-1 signals
INTENT_KEYWORDS = {
    'medical_query': {
        'en': ['pain', 'fever', 'headache', 'cough', 'cold', 'stomach', 'doctor', 'medicine', 'treatment', 'symptom', 'sick', 'ill', 'hurt', 'ache'],
        'hi': ['दर्द', 'बुखार', 'सिरदर्द', 'खांसी', 'सर्दी', 'पेट', 'डॉक्टर', 'दवाई', 'इलाज', 'बीमार'],
        'ta': ['வலி', 'காய்ச்சல்', 'தலைவலி', 'இருமல்', 'சளி', 'வயிறு', 'மருத்துவர்', 'மருந்து'],
        'te': ['నొప్పి', 'జ్వరం', 'తలనొప్పి', 'దగ్గు', 'జలుబు', 'కడుపు', 'డాక్టర్', 'మందు'],
        'kn': ['ನೋವು', 'ಜ್ವರ', 'ತಲೆನೋವು', 'ಕೆಮ್ಮು', 'ಶೀತ', 'ಹೊಟ್ಟೆ', 'ವೈದ್ಯರು', 'ಔಷಧಿ'],
    },
    'appointment': {
        'en': ['appointment', 'book', 'schedule', 'slot', 'available', 'timing', 'visit', 'meet doctor', 'see doctor'],
        'hi': ['अपॉइंटमेंट', 'बुक', 'समय', 'मिलना', 'डॉक्टर से मिलना'],
        'ta': ['நேரம்', 'முன்பதிவு', 'சந்திப்பு'],
        'te': ['అపాయింట్మెంట్', 'బుక్', 'సమయం'],
        'kn': ['ಅಪಾಯಿಂಟ್ಮೆಂಟ್', 'ಬುಕ್', 'ಸಮಯ', 'ಭೇಟಿ'],
    },
    'greeting': {
        'en': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'namaste'],
        'hi': ['नमस्ते', 'नमस्कार', 'हैलो'],
        'ta': ['வணக்கம்', 'ஹலோ'],
        'te': ['నమస్కారం', 'హలో'],
        'kn': ['ನಮಸ್ಕಾರ', 'ಹಲೋ'],
    },
    'admin': {
        'en': ['bill', 'payment', 'insurance', 'cost', 'price', 'charge', 'report', 'record', 'visiting hours', 'parking'],
        'hi': ['बिल', 'पेमेंट', 'बीमा', 'रिपोर्ट'],
        'ta': ['பில்', 'கட்டணம்', 'காப்பீடு'],
        'te': ['బిల్', 'చెల్లింపు', 'బీమా'],
        'kn': ['ಬಿಲ್', 'ಪಾವತಿ', 'ವಿಮೆ'],
    }
}

# Urgency keywords - HIGH priority
URGENCY_HIGH = {
    'en': ['severe', 'extreme', 'unbearable', 'emergency', 'urgent', 'immediately', "can't breathe", 'chest pain', 'unconscious', 'bleeding', 'accident', 'critical', 'dying', 'collapsed'],
    'hi': ['गंभीर', 'बहुत', 'असहनीय', 'इमरजेंसी', 'तुरंत', 'सांस नहीं', 'छाती में दर्द', 'बेहोश', 'खून'],
    'ta': ['கடுமையான', 'அவசரம்', 'உடனடி', 'மூச்சு', 'நெஞ்சு வலி', 'மயக்கம்', 'இரத்தம்'],
    'te': ['తీవ్రమైన', 'అత్యవసరం', 'వెంటనే', 'ఊపిరి', 'ఛాతీ నొప్పి', 'స్పృహ', 'రక్తం'],
    'kn': ['ತೀವ್ರ', 'ತುರ್ತು', 'ತಕ್ಷಣ', 'ಉಸಿರು', 'ಎದೆ ನೋವು', 'ಪ್ರಜ್ಞೆ', 'ರಕ್ತ'],
}

# Critical symptoms that always require escalation
CRITICAL_SYMPTOMS = {
    'en': ['chest pain', 'heart attack', "can't breathe", 'difficulty breathing', 'unconscious', 'seizure', 'stroke', 'severe bleeding', 'choking', 'suicide', 'poisoning', 'overdose'],
    'hi': ['छाती में दर्द', 'हार्ट अटैक', 'सांस नहीं', 'बेहोश', 'दौरा', 'खून बह रहा', 'जहर'],
    'ta': ['நெஞ்சு வலி', 'மாரடைப்பு', 'மூச்சு திணறல்', 'மயக்கம்', 'வலிப்பு'],
    'te': ['ఛాతీ నొప్పి', 'గుండెపోటు', 'ఊపిరి ఆడటం లేదు', 'స్పృహ లేదు'],
    'kn': ['ಎದೆ ನೋವು', 'ಹೃದಯಾಘಾತ', 'ಉಸಿರಾಟ ತೊಂದರೆ', 'ಪ್ರಜ್ಞೆ ತಪ್ಪು'],
}

# Stress indicators - emotional distress signals
STRESS_INDICATORS = {
    'en': ['help me', 'please help', 'very worried', 'scared', 'afraid', 'anxious', "can't sleep", 'desperate', 'terrible', 'worst pain', 'unbearable'],
    'hi': ['मदद करो', 'बहुत चिंता', 'डर', 'नींद नहीं', 'असहनीय'],
    'ta': ['உதவி', 'பயம்', 'கவலை', 'தூக்கமின்மை'],
    'te': ['సహాయం', 'భయం', 'ఆందోళన'],
    'kn': ['ಸಹಾಯ', 'ಭಯ', 'ಚಿಂತೆ', 'ನಿದ್ರೆ ಬರುತ್ತಿಲ್ಲ'],
}

# ============================================================================
# PYDANTIC MODELS FOR ADMIN API
# ============================================================================

class ConfigUpdate(BaseModel):
    hospital_name: Optional[str] = None
    city: Optional[str] = None
    emergency_number: Optional[str] = None
    helpline: Optional[str] = None
    tone: Optional[str] = None
    max_words: Optional[int] = None
    disclaimer_text: Optional[str] = None
    disclaimer_required: Optional[bool] = None
    always_recommend_doctor: Optional[bool] = None
    tts_enabled: Optional[bool] = None
    session_timeout_minutes: Optional[int] = None

class DoctorCreate(BaseModel):
    id: Optional[str] = None
    name: str
    specialization: str
    department: str
    available_days: List[str]
    timings: str
    languages: List[str] = ["en"]
    consultation_fee: Optional[str] = None
    name_kn: Optional[str] = None
    name_hi: Optional[str] = None
    name_ta: Optional[str] = None
    name_te: Optional[str] = None

class DepartmentCreate(BaseModel):
    id: Optional[str] = None
    name: str
    floor: str
    timings: str
    contact: Optional[str] = None
    name_kn: Optional[str] = None

class FAQCreate(BaseModel):
    id: Optional[str] = None
    question: str
    answer: str
    category: Optional[str] = "general"
    question_kn: Optional[str] = None
    question_hi: Optional[str] = None
    answer_kn: Optional[str] = None

# ============================================================================
# GLOBAL STATE
# ============================================================================

class ModelState:
    """Global state for loaded models"""
    vad_model = None
    vad_utils = None
    whisper_model = None
    llm_model = None
    llm_tokenizer = None
    tts_model = None
    tts_tokenizer = None
    tts_description_tokenizer = None
    tts_tokenizer = None
    tts_description_tokenizer = None
    fasttext_model = None  # Added for FastText
    device = None
    models_loaded = False

models = ModelState()

# Global services
redis_store: Optional[RedisStore] = None
conversation_manager: Optional[ConversationManager] = None
rag_retriever: Optional[RAGRetriever] = None

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vad_model():
    """Load Silero VAD model"""
    logger.info("Loading Silero VAD model...")
    start = time.time()
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    model = model.to(models.device)
    logger.info(f"VAD loaded in {time.time() - start:.2f}s")
    return model, utils

def load_whisper_model():
    """Load Faster-Whisper STT model"""
    logger.info("Loading Faster-Whisper model...")
    start = time.time()
    from faster_whisper import WhisperModel
    
    # Use GPU if available, otherwise CPU
    if models.device == "cuda":
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    else:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    
    logger.info(f"Whisper loaded in {time.time() - start:.2f}s")
    return model

def load_llm_model():
    """Load LLaMA model with 4-bit quantization"""
    logger.info("Loading LLaMA model...")
    start = time.time()
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Export your Hugging Face token "
            "(see .env.example) before loading the LLM."
        )

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    
    logger.info(f"LLM loaded in {time.time() - start:.2f}s")
    return model, tokenizer

def load_tts_model():
    """Load AI4Bharat Indic Parler-TTS model"""
    logger.info("Loading TTS model...")
    start = time.time()
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    
    model_id = "ai4bharat/indic-parler-tts"
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(models.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    description_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    
    logger.info(f"TTS loaded in {time.time() - start:.2f}s")
    return model, tokenizer, description_tokenizer

def load_fasttext_model():
    """Load FastText language detection model"""
    logger.info("Loading FastText model...")
    start = time.time()
    try:
        # Load the model we downloaded to models/lid.176.ftz
        model_path = os.path.join("models", "lid.176.ftz")
        if not os.path.exists(model_path):
             # Fallback if not found locally, though we should have downloaded it
             logger.warning("FastText model not found locally, attempting download...")
             os.system("curl -o models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
        
        model = fasttext.load_model(model_path)
        logger.info(f"FastText loaded in {time.time() - start:.2f}s")
        return model
    except Exception as e:
        logger.error(f"Failed to load FastText: {e}")
        return None

def init_services():
    """Initialize Redis, conversation manager, and RAG"""
    global redis_store, conversation_manager, rag_retriever
    
    # Initialize Redis
    redis_store = get_redis()
    logger.info(f"Redis connected: {redis_store.is_connected}")
    
    # Initialize conversation manager
    conversation_manager = ConversationManager(redis_store)
    logger.info("Conversation manager initialized")
    
    # Initialize RAG
    rag_retriever = RAGRetriever()
    
    # Load documents for RAG
    if redis_store.is_connected:
        docs = load_documents_from_redis(redis_store)
    else:
        # Fallback to JSON file
        json_path = os.path.join(os.path.dirname(__file__), "data", "sample_hospital.json")
        docs = load_documents_from_json(json_path)
    
    if docs:
        rag_retriever.add_documents(docs)
        # Try to build index (will fail gracefully if sentence-transformers not installed)
        try:
            rag_retriever.load()
            rag_retriever.build_index()
            logger.info("RAG index built successfully")
        except Exception as e:
            logger.warning(f"RAG index build failed: {e}. RAG features disabled.")

def load_all_models():
    """Load all models on startup"""
    models.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {models.device}")
    
    # Initialize services first (Redis, RAG)
    init_services()
    
    try:
        # models.vad_model, models.vad_utils = load_vad_model()
        models.vad_model = None
        models.vad_utils = None
        models.whisper_model = load_whisper_model()
        models.llm_model, models.llm_tokenizer = load_llm_model()
        models.llm_model, models.llm_tokenizer = load_llm_model()
        models.tts_model, models.tts_tokenizer, models.tts_description_tokenizer = load_tts_model()
        models.fasttext_model = load_fasttext_model() # Load FastText
        models.models_loaded = True
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def run_vad(audio_data: np.ndarray, sample_rate: int) -> dict:
    """Run Voice Activity Detection (Mocked if model missing)"""
    start = time.time()
    
    if models.vad_model is None:
        # Bypass VAD if no model loaded
        return {
            "has_speech": True,
            "speech_segments": [],
            "latency_ms": 0.0
        }
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio_data).to(models.device)
    
    # Run VAD
    get_speech_timestamps = models.vad_utils[0]
    speech_timestamps = get_speech_timestamps(audio_tensor, models.vad_model, sampling_rate=sample_rate)
    
    has_speech = len(speech_timestamps) > 0
    latency = (time.time() - start) * 1000
    
    return {
        "has_speech": has_speech,
        "speech_segments": speech_timestamps,
        "latency_ms": round(latency, 2)
    }

def run_stt(audio_data: np.ndarray, sample_rate: int, language: str = None) -> dict:
    """Run Speech-to-Text with Faster-Whisper"""
    start = time.time()
    
    # Save to temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_data, sample_rate)
        temp_path = f.name
    
    try:
        segments, info = models.whisper_model.transcribe(
            temp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            task="transcribe"  # Force transcription
        )
        
        # Collect all segments
        text = " ".join([seg.text for seg in segments])
        detected_language = info.language
        confidence = info.language_probability
        
    finally:
        os.unlink(temp_path)
    
    latency = (time.time() - start) * 1000
    
    return {
        "text": text.strip(),
        "detected_language": detected_language,
        "confidence": round(confidence, 3),
        "latency_ms": round(latency, 2)
    }

def detect_script_language(text: str, audio_lang_hint: str = None, audio_conf_hint: float = 0.0) -> dict:
    """
    Detect language using Hybrid Consensus:
    1. Unicode Script (Ground truth for native text)
    2. FastText (Validation/Disambiguation)
    3. Audio Language Hint (Whisper's detection - crucial for translated native speech)
    """
    start = time.time()
    
    char_counts = {lang: 0 for lang in SCRIPT_RANGES}
    
    for char in text:
        code_point = ord(char)
        for lang, ranges in SCRIPT_RANGES.items():
            for range_start, range_end in ranges:
                if range_start <= code_point <= range_end:
                    char_counts[lang] += 1
                    break
    
    total = sum(char_counts.values())
    
    # Layer 0: Script-based detection
    if total == 0:
        script_detected = 'en'
        script_confidence = 0.5
    else:
        script_detected = max(char_counts, key=char_counts.get)
        script_confidence = char_counts[script_detected] / total

    # Layer 1: FastText Detection
    ft_detected = None
    ft_confidence = 0.0
    
    if models.fasttext_model:
        try:
            clean_text = text.replace('\n', ' ').strip()
            labels, scores = models.fasttext_model.predict(clean_text)
            if labels:
                ft_lang = labels[0].replace("__label__", "")
                ft_score = float(scores[0])
                if ft_lang in SUPPORTED_LANGUAGES:
                    ft_detected = ft_lang
                    ft_confidence = ft_score
        except Exception as e:
            logger.warning(f"FastText prediction failed: {e}")

    # Decision Logic: Combine layers
    final_lang = script_detected
    final_conf = script_confidence
    
    # 1. FastText Refinement
    if ft_detected and ft_confidence > 0.4:
        if script_detected == 'en' and ft_detected != 'en':
             final_lang = ft_detected
             final_conf = ft_confidence
        elif script_confidence < 0.6:
             final_lang = ft_detected
             final_conf = ft_confidence
             
    # 2. Audio Signal Consensus (The "Anti-Translation" Check)
    # Problem: Whisper hears Kannada ("Namaskara") but outputs English ("Hello").
    # Signal: Audio=kn (high conf), Text=en (high conf).
    # Solution: Trust Audio for language separation.
    
    is_translation = False
    
    if audio_lang_hint in SUPPORTED_LANGUAGES:
        # If Audio is confident and Native, but Text is English
        if audio_conf_hint > 0.6 and audio_lang_hint != 'en':
            if final_lang == 'en':
                logger.info(f"Consensus Override: Audio({audio_lang_hint}, {audio_conf_hint:.2f}) overrules Text({final_lang}) - Likely Translation")
                final_lang = audio_lang_hint
                final_conf = audio_conf_hint
                is_translation = True
            
            # If Audio and Text disagree on Native languages (e.g. Audio=te, Text=kn)
            # This is harder. Text is usually ground truth content-wise.
            # But if Text confidence is low, maybe trust Audio? 
            # Sticking to text if it's native script is usually safer unless it's garbage.
            
    latency = (time.time() - start) * 1000
    
    return {
        "language": final_lang,
        "language_name": SUPPORTED_LANGUAGES.get(final_lang, "Unknown"),
        "confidence": round(final_conf, 3),
        "char_distribution": char_counts,
        "fasttext_prediction": {"lang": ft_detected, "conf": ft_confidence},
        "audio_signal": {"lang": audio_lang_hint, "conf": audio_conf_hint, "is_translation": is_translation},
        "latency_ms": round(latency, 2)
    }

def extract_layer1_signals(text: str, language: str) -> dict:
    """Extract intent, urgency, tone, and stress signals"""
    start = time.time()
    
    text_lower = text.lower()
    
    # Detect intent
    intent = "general"
    intent_confidence = 0.5
    
    for intent_type, lang_keywords in INTENT_KEYWORDS.items():
        keywords = lang_keywords.get(language, []) + lang_keywords.get('en', [])
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        if matches > 0:
            confidence = min(matches / 3, 1.0)
            if confidence > intent_confidence:
                intent = intent_type
                intent_confidence = confidence
    
    # Detect urgency
    urgency = "normal"
    urgency_count = 0
    urgency_keywords = URGENCY_HIGH.get(language, []) + URGENCY_HIGH.get('en', [])
    for kw in urgency_keywords:
        if kw.lower() in text_lower:
            urgency_count += 1
    
    if urgency_count >= 2:
        urgency = "high"
    elif urgency_count == 1:
        urgency = "medium"
    
    # Detect stress indicators
    stress_level = "calm"
    stress_keywords = STRESS_INDICATORS.get(language, []) + STRESS_INDICATORS.get('en', [])
    if any(kw.lower() in text_lower for kw in stress_keywords):
        stress_level = "stressed"
    
    # Detect tone (based on urgency, stress, and punctuation)
    if urgency == "high" or stress_level == "stressed" or text.count('!') > 0:
        tone = "stressed"
    else:
        tone = "calm"
    
    latency = (time.time() - start) * 1000
    
    return {
        "intent": intent,
        "intent_confidence": round(intent_confidence, 3),
        "urgency": urgency,
        "urgency_count": urgency_count,
        "tone": tone,
        "stress_level": stress_level,
        "latency_ms": round(latency, 2)
    }

def run_safety_gate(text: str, language: str, signals: dict, session_id: str = None) -> dict:
    """
    Enhanced Safety Gate with 5 escalation rules:
    1. Critical symptoms → always escalate
    2. Chest/breathing + medical query → escalate
    3. Low confidence + medical query → escalate
    4. High urgency (>=2 keywords) → escalate
    5. Repeated similar query → escalate
    """
    start = time.time()
    
    text_lower = text.lower()
    should_escalate = False
    escalation_reason = None
    escalation_rules_triggered = []
    
    # Rule 1: Check for critical symptoms
    critical_keywords = CRITICAL_SYMPTOMS.get(language, []) + CRITICAL_SYMPTOMS.get('en', [])
    for symptom in critical_keywords:
        if symptom.lower() in text_lower:
            should_escalate = True
            escalation_reason = f"Critical symptom detected: {symptom}"
            escalation_rules_triggered.append("rule_1_critical_symptom")
            break
    
    # Rule 2: Chest/breathing + medical query
    breathing_keywords = ['chest', 'breathe', 'breathing', 'heart', 'छाती', 'सांस', 'ಎದೆ', 'ಉಸಿರು']
    has_breathing_concern = any(kw in text_lower for kw in breathing_keywords)
    if has_breathing_concern and signals['intent'] == 'medical_query':
        should_escalate = True
        if not escalation_reason:
            escalation_reason = "Chest/breathing concern with medical query"
        escalation_rules_triggered.append("rule_2_breathing_medical")
    
    # Rule 3: Low confidence on medical query
    if signals['intent'] == 'medical_query' and signals['intent_confidence'] < 0.3:
        should_escalate = True
        if not escalation_reason:
            escalation_reason = "Low confidence medical query - needs clarification"
        escalation_rules_triggered.append("rule_3_low_confidence")
    
    # Rule 4: High urgency (2+ urgency keywords)
    if signals.get('urgency_count', 0) >= 2:
        should_escalate = True
        if not escalation_reason:
            escalation_reason = "Multiple urgency indicators detected"
        escalation_rules_triggered.append("rule_4_high_urgency")
    
    # Rule 5: Repeated similar query (requires session)
    if session_id and conversation_manager:
        repeat_count = conversation_manager.get_repeated_query_count(session_id, text)
        if repeat_count >= 2:
            should_escalate = True
            if not escalation_reason:
                escalation_reason = f"Repeated query ({repeat_count}x) - patient may need direct assistance"
            escalation_rules_triggered.append("rule_5_repeated_query")
    
    latency = (time.time() - start) * 1000
    
    return {
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "rules_triggered": escalation_rules_triggered,
        "latency_ms": round(latency, 2)
    }

def apply_policy(signals: dict, safety: dict, language: str) -> dict:
    """Apply Apollo Hospital policy constraints from admin config"""
    start = time.time()
    
    # Get config from Redis or use defaults
    if redis_store and redis_store.is_connected:
        config = redis_store.get_config()
    else:
        config = {
            "hospital_name": "Apollo Hospital",
            "emergency_number": "108",
            "helpline": "1860-500-1066",
            "tone": "formal",
            "max_words": 50,
            "disclaimer_required": True,
            "disclaimer_text": "This is general information only. Please consult a doctor for proper medical advice.",
            "always_recommend_doctor": True
        }
    
    policy = {
        "hospital_name": config.get("hospital_name", "Apollo Hospital"),
        "response_language": language,
        "max_words": config.get("max_words", 50),
        "tone": config.get("tone", "formal"),
        "include_doctor_recommendation": config.get("always_recommend_doctor", True),
        "emergency_number": config.get("emergency_number", "108"),
        "helpline": config.get("helpline", "1860-500-1066"),
        "disclaimer": config.get("disclaimer_required", True),
        "disclaimer_text": config.get("disclaimer_text", "Please consult a doctor for proper medical advice.")
    }
    
    # Adjust policy based on signals
    if safety['should_escalate']:
        policy['priority'] = 'urgent'
        policy['include_emergency_info'] = True
        policy['max_words'] = 80  # Allow longer response for emergencies
    else:
        policy['priority'] = 'normal'
        policy['include_emergency_info'] = False
    
    latency = (time.time() - start) * 1000
    policy['latency_ms'] = round(latency, 2)
    
    return policy

def generate_llm_response(
    text: str, 
    language: str, 
    signals: dict, 
    safety: dict, 
    policy: dict,
    session_id: str = None
) -> dict:
    """Generate response using LLaMA with context and RAG"""
    start = time.time()
    
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    
    # Get conversation context if session exists
    context_string = ""
    if session_id and conversation_manager:
        ctx = conversation_manager.get_conversation_context(session_id)
        context_string = ctx.get('context_string', '')
    
    # Get relevant FAQs/doctors using RAG
    rag_context = ""
    if rag_retriever and rag_retriever.is_ready:
        results = rag_retriever.search(text, top_k=2, language=language)
        if results:
            rag_context = rag_retriever.format_results_for_prompt(results)
    
    # Build system prompt with policy
    system_prompt = f"""You are a helpful medical assistant for {policy['hospital_name']}. 
Follow these rules strictly:
1. Respond in {lang_name} language using the native script
2. Keep responses under {policy['max_words']} words
3. Use a {policy['tone']} tone
4. Always recommend consulting a doctor for medical issues
5. For emergencies, mention calling {policy['emergency_number']}
6. Never diagnose or prescribe medication
7. Be empathetic and professional
8. Helpline: {policy['helpline']}

Patient intent: {signals['intent']}
Urgency level: {signals['urgency']}
{"URGENT: This may be an emergency situation. Prioritize safety and recommend immediate medical attention." if safety['should_escalate'] else ""}
"""

    # Add conversation context
    if context_string:
        system_prompt += f"\n\nCONVERSATION CONTEXT:\n{context_string}\n"
    
    # Add RAG context
    if rag_context:
        system_prompt += f"\n\nRELEVANT INFORMATION:\n{rag_context}\n"
    
    # Add disclaimer instruction
    if policy.get('disclaimer'):
        system_prompt += f"\n\nInclude a brief disclaimer: {policy['disclaimer_text']}"

    # Build prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    
    # Format for LLaMA
    prompt = models.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = models.llm_tokenizer(prompt, return_tensors="pt").to(models.device)
    
    with torch.no_grad():
        outputs = models.llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=models.llm_tokenizer.eos_token_id
        )
    
    response = models.llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    latency = (time.time() - start) * 1000
    
    return {
        "response": response.strip(),
        "used_rag": bool(rag_context),
        "used_context": bool(context_string),
        "latency_ms": round(latency, 2)
    }

def generate_tts(text: str, language: str) -> dict:
    """Generate speech using Indic Parler-TTS"""
    start = time.time()
    
    lang_name = SUPPORTED_LANGUAGES.get(language, "Hindi")
    
    # Description for the TTS voice
    description = f"A female speaker delivers a clear, professional medical response in {lang_name} with a calm and reassuring tone."
    
    # Tokenize
    description_tokens = models.tts_description_tokenizer(description, return_tensors="pt").to(models.device)
    prompt_tokens = models.tts_tokenizer(text, return_tensors="pt").to(models.device)
    
    # Generate audio
    with torch.no_grad():
        generation = models.tts_model.generate(
            input_ids=description_tokens.input_ids,
            attention_mask=description_tokens.attention_mask,
            prompt_input_ids=prompt_tokens.input_ids,
            prompt_attention_mask=prompt_tokens.attention_mask
        )
    
    audio_array = generation.cpu().numpy().squeeze()
    sample_rate = models.tts_model.config.sampling_rate
    
    # Convert to base64 for sending to frontend
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    latency = (time.time() - start) * 1000
    
    return {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
        "duration_ms": round(len(audio_array) / sample_rate * 1000, 2),
        "latency_ms": round(latency, 2)
    }

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Starting Apollo Hospital Voice AI Assistant v2.0...")
    load_all_models()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Apollo Hospital Voice AI Assistant",
    description="Real-time regional-language voice AI for patient health queries",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

@app.get("/admin")
async def admin_page():
    """Serve the admin panel"""
    return FileResponse("static/admin.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models.models_loaded else "loading",
        "models_loaded": models.models_loaded,
        "device": models.device,
        "redis_connected": redis_store.is_connected if redis_store else False,
        "rag_ready": rag_retriever.is_ready if rag_retriever else False,
        "supported_languages": SUPPORTED_LANGUAGES
    }

# ============================================================================
# ADMIN API ENDPOINTS
# ============================================================================

@app.get("/api/admin/config")
async def get_config():
    """Get hospital configuration"""
    if redis_store and redis_store.is_connected:
        return redis_store.get_config()
    return redis_store._get_default_config() if redis_store else {}

@app.post("/api/admin/config")
async def update_config(config: ConfigUpdate):
    """Update hospital configuration"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    # Get current config and update
    current = redis_store.get_config()
    updates = config.dict(exclude_none=True)
    current.update(updates)
    
    if redis_store.set_config(current):
        return {"success": True, "config": current}
    raise HTTPException(status_code=500, detail="Failed to update config")

@app.get("/api/admin/doctors")
async def get_doctors():
    """Get all doctors"""
    if redis_store and redis_store.is_connected:
        return {"doctors": redis_store.get_doctors()}
    return {"doctors": []}

@app.post("/api/admin/doctors")
async def add_doctor(doctor: DoctorCreate):
    """Add a new doctor"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    doc_id = redis_store.add_doctor(doctor.dict())
    if doc_id:
        # Rebuild RAG index
        if rag_retriever:
            docs = load_documents_from_redis(redis_store)
            rag_retriever.documents = []
            rag_retriever.add_documents(docs)
            rag_retriever.build_index()
        return {"success": True, "id": doc_id}
    raise HTTPException(status_code=500, detail="Failed to add doctor")

@app.put("/api/admin/doctors/{doc_id}")
async def update_doctor(doc_id: str, doctor: DoctorCreate):
    """Update a doctor"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    if redis_store.update_doctor(doc_id, doctor.dict()):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Doctor not found")

@app.delete("/api/admin/doctors/{doc_id}")
async def delete_doctor(doc_id: str):
    """Delete a doctor"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    if redis_store.delete_doctor(doc_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Doctor not found")

@app.get("/api/admin/departments")
async def get_departments():
    """Get all departments"""
    if redis_store and redis_store.is_connected:
        return {"departments": redis_store.get_departments()}
    return {"departments": []}

@app.post("/api/admin/departments")
async def add_department(department: DepartmentCreate):
    """Add a new department"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    dept_id = redis_store.add_department(department.dict())
    if dept_id:
        return {"success": True, "id": dept_id}
    raise HTTPException(status_code=500, detail="Failed to add department")

@app.delete("/api/admin/departments/{dept_id}")
async def delete_department(dept_id: str):
    """Delete a department"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    if redis_store.delete_department(dept_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="Department not found")

@app.get("/api/admin/faqs")
async def get_faqs():
    """Get all FAQs"""
    if redis_store and redis_store.is_connected:
        return {"faqs": redis_store.get_faqs()}
    return {"faqs": []}

@app.post("/api/admin/faqs")
async def add_faq(faq: FAQCreate):
    """Add a new FAQ"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    faq_id = redis_store.add_faq(faq.dict())
    if faq_id:
        # Rebuild RAG index
        if rag_retriever:
            docs = load_documents_from_redis(redis_store)
            rag_retriever.documents = []
            rag_retriever.add_documents(docs)
            rag_retriever.build_index()
        return {"success": True, "id": faq_id}
    raise HTTPException(status_code=500, detail="Failed to add FAQ")

@app.put("/api/admin/faqs/{faq_id}")
async def update_faq(faq_id: str, faq: FAQCreate):
    """Update a FAQ"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    if redis_store.update_faq(faq_id, faq.dict()):
        return {"success": True}
    raise HTTPException(status_code=404, detail="FAQ not found")

@app.delete("/api/admin/faqs/{faq_id}")
async def delete_faq(faq_id: str):
    """Delete a FAQ"""
    if not redis_store or not redis_store.is_connected:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    if redis_store.delete_faq(faq_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="FAQ not found")

# ============================================================================
# SESSION API
# ============================================================================

@app.post("/api/session")
async def create_session():
    """Create a new conversation session"""
    if conversation_manager:
        session_id = conversation_manager.create_session()
        return {"session_id": session_id}
    # Fallback to simple UUID
    return {"session_id": str(uuid.uuid4())[:8]}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session data"""
    if conversation_manager:
        session = conversation_manager.get_session(session_id)
        if session:
            return session
    raise HTTPException(status_code=404, detail="Session not found")

# ============================================================================
# MAIN PROCESS ENDPOINT
# ============================================================================

@app.post("/process")
async def process_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
):
    """Process audio through the full pipeline with session support"""
    
    if not models.models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading")
    
    total_start = time.time()
    
    # Create session if not provided
    if not session_id and conversation_manager:
        session_id = conversation_manager.create_session()
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Ensure float32
        audio_data = audio_data.astype(np.float32)
        
        # ====== PIPELINE ======
        
        # 1. VAD
        vad_result = run_vad(audio_data, sample_rate)
        
        if not vad_result['has_speech']:
            return JSONResponse({
                "success": False,
                "error": "No speech detected in audio",
                "session_id": session_id,
                "pipeline": {"vad": vad_result}
            })
        
        # 2. STT
        stt_result = run_stt(audio_data, sample_rate, language)
        
        if not stt_result['text']:
            return JSONResponse({
                "success": False,
                "error": "Could not transcribe audio",
                "session_id": session_id,
                "pipeline": {"vad": vad_result, "stt": stt_result}
            })
        
        # 3. Language Detection (Hybrid: Text + Audio Consensus)
        lang_result = detect_script_language(
            stt_result['text'], 
            audio_lang_hint=stt_result.get('language'),
            audio_conf_hint=stt_result.get('confidence', 0.0)
        )
        detected_lang = language or lang_result['language']
        
        # 4. Layer-1 Signals
        signals_result = extract_layer1_signals(stt_result['text'], detected_lang)
        
        # 5. Safety Gate (with session for repeat detection)
        safety_result = run_safety_gate(stt_result['text'], detected_lang, signals_result, session_id)
        
        # 6. Policy Engine
        policy_result = apply_policy(signals_result, safety_result, detected_lang)
        
        # 7. LLM Response (with context and RAG)
        llm_result = generate_llm_response(
            stt_result['text'], 
            detected_lang, 
            signals_result, 
            safety_result, 
            policy_result,
            session_id
        )
        
        # 8. TTS
        tts_result = generate_tts(llm_result['response'], detected_lang)
        
        # 9. Update conversation history
        if session_id and conversation_manager:
            conversation_manager.add_turn(
                session_id,
                stt_result['text'],
                llm_result['response'],
                detected_lang
            )
        
        # Calculate total latency
        total_latency = (time.time() - total_start) * 1000
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "transcription": stt_result['text'],
            "response": llm_result['response'],
            "audio_response": tts_result['audio_base64'],
            "audio_sample_rate": tts_result['sample_rate'],
            "language": detected_lang,
            "language_name": SUPPORTED_LANGUAGES.get(detected_lang, "Unknown"),
            "escalation": {
                "should_escalate": safety_result['should_escalate'],
                "reason": safety_result['escalation_reason'],
                "rules_triggered": safety_result.get('rules_triggered', [])
            },
            "context": {
                "used_rag": llm_result.get('used_rag', False),
                "used_history": llm_result.get('used_context', False)
            },
            "pipeline": {
                "vad": {"latency_ms": vad_result['latency_ms']},
                "stt": {"latency_ms": stt_result['latency_ms'], "confidence": stt_result['confidence']},
                "language_detection": lang_result,
                "layer1_signals": {
                    "latency_ms": signals_result['latency_ms'],
                    "intent": signals_result['intent'],
                    "urgency": signals_result['urgency'],
                    "tone": signals_result['tone'],
                    "stress_level": signals_result.get('stress_level', 'calm')
                },
                "safety_gate": {"latency_ms": safety_result['latency_ms']},
                "policy": {"latency_ms": policy_result['latency_ms']},
                "llm": {"latency_ms": llm_result['latency_ms']},
                "tts": {"latency_ms": tts_result['latency_ms'], "duration_ms": tts_result['duration_ms']}
            },
            "total_latency_ms": round(total_latency, 2)
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_languages():
    """Get supported languages"""
    return {"languages": SUPPORTED_LANGUAGES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
