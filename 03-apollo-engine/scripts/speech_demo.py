"""
Apollo Voice Engine - Real-Time Speech-to-Speech Demo with Whisper

Uses:
- OpenAI Whisper for STT (supports all Indian languages)
- Edge-TTS for TTS (Microsoft voices, supports Kannada, Tamil, Telugu, Hindi)
- Browser-based audio recording and playback

Usage:
    python scripts/speech_demo.py

Then open http://localhost:8000 in any modern browser
"""

import sys
import os
import io
import time
import json
import base64
import asyncio
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import safety classifier
from apollo_voice_engine.safety.classifier import SafetyClassifier, SafetyLevel

# Lazy load heavy modules
whisper_model = None
edge_tts = None

def get_whisper_model():
    """Lazy load Whisper model."""
    global whisper_model
    if whisper_model is None:
        import whisper
        print("Loading Whisper model (this may take a moment)...")
        # Use "small" for better accuracy, especially with Indian languages
        # Options: tiny, base, small, medium, large
        # small provides good balance of speed and accuracy
        whisper_model = whisper.load_model("small")
        print("✓ Whisper 'small' model loaded")
    return whisper_model

# ============================================================================
# Configuration
# ============================================================================

app = FastAPI(
    title="Apollo Omni-Indic Voice Engine",
    description="Real-time Speech-to-Speech AI with Whisper + Edge-TTS",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = SafetyClassifier()

METRICS = {
    "avg_stt_ms": 0,
    "avg_llm_ms": 0,
    "avg_tts_ms": 0,
    "avg_total_ms": 0,
    "cost_per_min_inr": 0.05,
    "total_requests": 0,
    "emergency_transfers": 0,
    "session_start": datetime.now().isoformat()
}

# ============================================================================
# Edge-TTS Voice Mapping for Indian Languages
# ============================================================================

VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural",      # Hindi - Female
    "ta": "ta-IN-PallaviNeural",    # Tamil - Female  
    "te": "te-IN-ShrutiNeural",     # Telugu - Female
    "kn": "kn-IN-SapnaNeural",      # Kannada - Female
    "en": "en-IN-NeerjaNeural"      # English (India) - Female
}

# Alternative male voices
VOICE_MAP_MALE = {
    "hi": "hi-IN-MadhurNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural", 
    "kn": "kn-IN-GaganNeural",
    "en": "en-IN-PrabhatNeural"
}

# ============================================================================
# Multi-language Response Templates
# ============================================================================

GREETINGS = {
    "hi": "नमस्ते! मैं अपोलो वॉयस असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
    "ta": "வணக்கம்! நான் அப்பல்லோ குரல் உதவியாளர். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "నమస్కారం! నేను అపోలో వాయిస్ అసిస్టెంట్. నేను మీకు ఎలా సహాయం చేయగలను?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ಅಪೊಲೊ ಧ್ವನಿ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "en": "Hello! I'm Apollo Voice Assistant. How can I help you today?"
}

RESPONSES = {
    "appointment": {
        "hi": "आपकी अगली अपॉइंटमेंट कल सुबह 10 बजे डॉ. शर्मा के साथ है।",
        "ta": "உங்கள் அடுத்த சந்திப்பு நாளை காலை 10 மணிக்கு டாக்டர் ஷர்மாவுடன்.",
        "te": "మీ తదుపరి అపాయింట్‌మెంట్ రేపు ఉదయం 10 గంటలకు డాక్టర్ శర్మతో.",
        "kn": "ನಿಮ್ಮ ಮುಂದಿನ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಡಾ. ಶರ್ಮಾ ಅವರೊಂದಿಗೆ.",
        "en": "Your next appointment is tomorrow at 10 AM with Dr. Sharma."
    },
    "pharmacy": {
        "hi": "फार्मेसी सुबह 8 बजे से रात 9 बजे तक खुली है।",
        "ta": "மருந்தகம் காலை 8 மணி முதல் இரவு 9 மணி வரை திறந்திருக்கும்.",
        "te": "ఫార్మసీ ఉదయం 8 నుండి రాత్రి 9 వరకు తెరిచి ఉంటుంది.",
        "kn": "ಫಾರ್ಮಸಿ ಬೆಳಿಗ್ಗೆ 8 ರಿಂದ ರಾತ್ರಿ 9 ರವರೆಗೆ ತೆರೆದಿರುತ್ತದೆ.",
        "en": "The pharmacy is open from 8 AM to 9 PM."
    },
    "cardiology": {
        "hi": "कार्डियोलॉजी विभाग तीसरी मंजिल पर है।",
        "ta": "இருதயவியல் துறை மூன்றாவது மாடியில் உள்ளது.",
        "te": "కార్డియాలజీ విభాగం మూడవ అంతస్తులో ఉంది.",
        "kn": "ಹೃದ್ರೋಗ ವಿಭಾಗ ಮೂರನೇ ಮಹಡಿಯಲ್ಲಿದೆ.",
        "en": "The Cardiology department is on the 3rd floor."
    },
    "default": {
        "hi": "मैं समझ गया। क्या मैं और जानकारी दे सकता हूँ?",
        "ta": "புரிந்துகொண்டேன். மேலும் தகவல் தர வேண்டுமா?",
        "te": "అర్థమైంది. మరింత సమాచారం ఇవ్వాలా?",
        "kn": "ಅರ್ಥವಾಯಿತು. ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ನೀಡಲೇ?",
        "en": "I understand. Would you like more information?"
    }
}

TRANSFER_MESSAGES = {
    "hi": "आपातकाल का पता चला। मैं आपको स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं।",
    "ta": "அவசரநிலை கண்டறியப்பட்டது. சுகாதார நிபுணரை இணைக்கிறேன்.",
    "te": "అత్యవసర పరిస్థితి. ఆరోగ్య నిపుణుడిని కనెక్ట్ చేస్తున్నాను.",
    "kn": "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಪತ್ತೆಯಾಗಿದೆ. ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ.",
    "en": "Emergency detected. Connecting you with a healthcare professional."
}

# ============================================================================
# Response Generation
# ============================================================================

def generate_response(text: str, lang: str) -> str:
    """Generate contextual response based on keywords."""
    text_lower = text.lower()
    
    keywords = {
        "appointment": ["appointment", "अपॉइंटमेंट", "சந்திப்பு", "అపాయింట్‌మెంట్", "ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್", "book"],
        "pharmacy": ["pharmacy", "medicine", "दवा", "மருந்து", "మందులు", "ಔಷಧ"],
        "cardiology": ["cardiology", "heart", "दिल", "இருதயம்", "గుండె", "ಹೃದಯ"],
        "greeting": ["hello", "hi", "नमस्ते", "வணக்கம்", "నమస్కారం", "ನಮಸ್ಕಾರ"]
    }
    
    for intent, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            if intent == "greeting":
                return GREETINGS.get(lang, GREETINGS["en"])
            return RESPONSES.get(intent, RESPONSES["default"]).get(lang, RESPONSES["default"]["en"])
    
    return RESPONSES["default"].get(lang, RESPONSES["default"]["en"])

# ============================================================================
# TTS using Edge-TTS
# ============================================================================

async def synthesize_speech(text: str, lang: str) -> bytes:
    """Generate speech audio using Edge-TTS."""
    import edge_tts
    
    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    
    communicate = edge_tts.Communicate(text, voice)
    
    audio_data = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.write(chunk["data"])
    
    return audio_data.getvalue()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the speech demo UI."""
    return get_whisper_demo_html()

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "3.0.0 (Whisper + Edge-TTS)",
        "stt": "OpenAI Whisper",
        "tts": "Microsoft Edge-TTS",
        "languages": list(VOICE_MAP.keys())
    }

@app.get("/api/languages")
async def languages():
    return {
        "languages": [
            {"code": "hi", "name": "Hindi", "native": "हिंदी", "voice": VOICE_MAP["hi"]},
            {"code": "ta", "name": "Tamil", "native": "தமிழ்", "voice": VOICE_MAP["ta"]},
            {"code": "te", "name": "Telugu", "native": "తెలుగు", "voice": VOICE_MAP["te"]},
            {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ", "voice": VOICE_MAP["kn"]},
            {"code": "en", "name": "English", "native": "English", "voice": VOICE_MAP["en"]}
        ]
    }

@app.get("/api/metrics")
async def metrics():
    return METRICS

class TextChatRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/api/chat")
async def chat_text(request: TextChatRequest):
    """Process text input and return text + audio response."""
    global METRICS
    
    start_time = time.time()
    lang = request.language
    text = request.text
    
    # Safety check
    safety_result = classifier.classify(text)
    METRICS["total_requests"] += 1
    
    # Generate response
    if safety_result.should_transfer:
        METRICS["emergency_transfers"] += 1
        response_text = TRANSFER_MESSAGES.get(lang, TRANSFER_MESSAGES["en"])
        action = "TRANSFER_TO_HUMAN"
    else:
        response_text = generate_response(text, lang)
        action = "RESPOND"
    
    # Generate TTS audio
    tts_start = time.time()
    try:
        audio_bytes = await synthesize_speech(response_text, lang)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        tts_ms = (time.time() - tts_start) * 1000
    except Exception as e:
        print(f"TTS Error: {e}")
        audio_b64 = ""
        tts_ms = 0
    
    total_ms = (time.time() - start_time) * 1000
    METRICS["avg_tts_ms"] = tts_ms
    METRICS["avg_total_ms"] = total_ms
    
    return {
        "transcription": text,
        "response": response_text,
        "audio": audio_b64,
        "action": action,
        "language": lang,
        "safety_level": safety_result.level.value,
        "latency_ms": total_ms,
        "tts_ms": tts_ms
    }

@app.post("/api/speech-to-speech")
async def speech_to_speech(
    audio: UploadFile = File(...),
    language: str = Form("en")
):
    """Full speech-to-speech: Audio in -> Audio out."""
    global METRICS
    
    start_time = time.time()
    
    # Save uploaded audio to temp file
    audio_bytes = await audio.read()
    print(f"Received audio: {len(audio_bytes)} bytes, type: {audio.content_type}")
    
    if len(audio_bytes) < 1000:
        return {
            "transcription": "[Audio too short]",
            "response": "Please speak longer. Recording was too short.",
            "audio": "",
            "action": "RESPOND",
            "language": language,
            "safety_level": "safe",
            "stt_ms": 0,
            "llm_ms": 0,
            "tts_ms": 0,
            "total_ms": 0
        }
    
    webm_path = None
    wav_path = None
    
    try:
        # Save the webm file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name
        
        print(f"Saved audio to: {webm_path}")
        
        # Convert webm to wav using pydub (more reliable) or ffmpeg
        wav_path = webm_path.replace(".webm", ".wav")
        
        try:
            from pydub import AudioSegment
            # Load the audio file
            audio_segment = AudioSegment.from_file(webm_path)
            # Convert to 16kHz mono WAV
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            audio_segment.export(wav_path, format="wav")
            print(f"Converted audio: duration={len(audio_segment)}ms, sample_rate=16000")
        except Exception as pydub_error:
            print(f"Pydub error: {pydub_error}, trying ffmpeg...")
            
            import subprocess
            try:
                result = subprocess.run([
                    "ffmpeg", "-y", "-i", webm_path,
                    "-ar", "16000",  # 16kHz sample rate
                    "-ac", "1",       # Mono
                    "-c:a", "pcm_s16le",
                    wav_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    wav_path = webm_path
                else:
                    print("FFmpeg conversion successful")
            except Exception as ffmpeg_error:
                print(f"FFmpeg failed: {ffmpeg_error}")
                wav_path = webm_path
        
        # Verify WAV file exists and has content
        if wav_path != webm_path and os.path.exists(wav_path):
            wav_size = os.path.getsize(wav_path)
            print(f"WAV file size: {wav_size} bytes")
        
        # Map language codes for Whisper
        whisper_lang_map = {
            "hi": "hi",      # Hindi
            "ta": "ta",      # Tamil
            "te": "te",      # Telugu  
            "kn": "kn",      # Kannada
            "en": "en"       # English
        }
        whisper_lang = whisper_lang_map.get(language, "en")
        
        # STT with Whisper
        stt_start = time.time()
        model = get_whisper_model()
        
        print(f"Transcribing with Whisper (lang={whisper_lang})...")
        
        # Use better transcription settings
        result = model.transcribe(
            wav_path,
            language=whisper_lang,
            task="transcribe",
            fp16=False,  # Use FP32 for CPU
            verbose=False
        )
        
        transcription = result["text"].strip()
        stt_ms = (time.time() - stt_start) * 1000
        
        # Handle empty transcription
        if not transcription:
            transcription = "[No speech detected]"
        
        print(f"Transcribed ({language}): '{transcription}' in {stt_ms:.0f}ms")
        
        # Safety check
        safety_result = classifier.classify(transcription)
        METRICS["total_requests"] += 1
        
        # Generate response
        llm_start = time.time()
        if safety_result.should_transfer:
            METRICS["emergency_transfers"] += 1
            response_text = TRANSFER_MESSAGES.get(language, TRANSFER_MESSAGES["en"])
            action = "TRANSFER_TO_HUMAN"
        else:
            response_text = generate_response(transcription, language)
            action = "RESPOND"
        llm_ms = (time.time() - llm_start) * 1000
        
        # TTS
        tts_start = time.time()
        audio_out = await synthesize_speech(response_text, language)
        audio_b64 = base64.b64encode(audio_out).decode()
        tts_ms = (time.time() - tts_start) * 1000
        
        total_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        METRICS["avg_stt_ms"] = stt_ms
        METRICS["avg_llm_ms"] = llm_ms
        METRICS["avg_tts_ms"] = tts_ms
        METRICS["avg_total_ms"] = total_ms
        
        return {
            "transcription": transcription,
            "response": response_text,
            "audio": audio_b64,
            "action": action,
            "language": language,
            "safety_level": safety_result.level.value,
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms
        }
        
    finally:
        # Clean up temp files
        try:
            if webm_path and os.path.exists(webm_path):
                os.unlink(webm_path)
            if wav_path and wav_path != webm_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

# ============================================================================
# Web UI
# ============================================================================

def get_whisper_demo_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apollo Voice Engine - Whisper Demo</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #00d4ff;
            --secondary: #00ff94;
            --danger: #ff4757;
            --warning: #ffc800;
            --bg-dark: #0a0a1a;
            --bg-card: rgba(255,255,255,0.05);
            --text: #ffffff;
            --text-muted: #8888aa;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at 20% 20%, rgba(0,212,255,0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(0,255,148,0.1) 0%, transparent 50%);
            min-height: 100vh;
            color: var(--text);
        }
        
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        
        header { text-align: center; padding: 30px 0; }
        
        h1 {
            font-size: 2.2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        
        .subtitle { color: var(--text-muted); margin-bottom: 15px; }
        
        .tech-badges {
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .tech-badge {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
        }
        
        .tech-badge.highlight {
            background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,255,148,0.2));
            border-color: var(--secondary);
        }
        
        .main-card {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .lang-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .lang-btn {
            padding: 10px 18px;
            border: 2px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            background: transparent;
            color: var(--text);
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .lang-btn:hover { border-color: var(--primary); }
        .lang-btn.active {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-color: transparent;
            color: var(--bg-dark);
        }
        
        .mic-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 25px 0;
        }
        
        .mic-btn {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--bg-dark);
            font-size: 40px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 8px 30px rgba(0,212,255,0.3);
        }
        
        .mic-btn:hover { transform: scale(1.05); }
        
        .mic-btn.recording {
            animation: pulse 1s infinite;
            background: linear-gradient(135deg, var(--danger), #ff6b7a);
        }
        
        .mic-btn.processing {
            background: linear-gradient(135deg, var(--warning), #ffda44);
            animation: spin 1s linear infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .status-text {
            margin-top: 12px;
            font-size: 14px;
            color: var(--text-muted);
        }
        
        .status-text.recording { color: var(--danger); font-weight: 600; }
        .status-text.processing { color: var(--warning); }
        .status-text.speaking { color: var(--secondary); }
        
        .chat-container {
            max-height: 350px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 10px;
            max-width: 85%;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .chat-bubble.user {
            background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,255,148,0.1));
            border: 1px solid rgba(0,212,255,0.3);
            margin-left: auto;
        }
        
        .chat-bubble.assistant {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .chat-bubble.emergency {
            background: rgba(255,71,87,0.2);
            border-color: var(--danger);
        }
        
        .chat-label {
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        
        .safety-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .safety-badge.safe { background: var(--secondary); color: var(--bg-dark); }
        .safety-badge.caution { background: var(--warning); color: var(--bg-dark); }
        .safety-badge.emergency { background: var(--danger); color: white; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-label { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
        
        .examples {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .example-btn {
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            color: var(--text);
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .example-btn:hover { background: rgba(255,255,255,0.1); }
        .example-btn.emergency { border-color: var(--danger); color: var(--danger); }
        
        @media (max-width: 600px) {
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            h1 { font-size: 1.6rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 Apollo Voice Engine</h1>
            <p class="subtitle">Real-Time Speech-to-Speech with Whisper + Edge-TTS</p>
            <div class="tech-badges">
                <span class="tech-badge highlight">🎤 OpenAI Whisper STT</span>
                <span class="tech-badge highlight">🔊 Microsoft Edge TTS</span>
                <span class="tech-badge">💰 ₹0.05/min</span>
                <span class="tech-badge">🔐 Safety First</span>
            </div>
        </header>
        
        <div class="main-card">
            <div class="lang-buttons">
                <button class="lang-btn" data-lang="hi">हिंदी</button>
                <button class="lang-btn" data-lang="ta">தமிழ்</button>
                <button class="lang-btn" data-lang="te">తెలుగు</button>
                <button class="lang-btn" data-lang="kn">ಕನ್ನಡ</button>
                <button class="lang-btn active" data-lang="en">English</button>
            </div>
            
            <div class="mic-container">
                <button id="micBtn" class="mic-btn">🎤</button>
                <p id="statusText" class="status-text">Click and hold to speak</p>
            </div>
            
            <div id="chatContainer" class="chat-container"></div>
            
            <div class="examples">
                <span style="color: var(--text-muted); font-size: 12px;">Try:</span>
                <button class="example-btn" onclick="sendText('What time does the pharmacy open?')">Pharmacy</button>
                <button class="example-btn" onclick="sendText('I need to book an appointment')">Appointment</button>
                <button class="example-btn" onclick="sendText('Where is cardiology?')">Cardiology</button>
                <button class="example-btn emergency" onclick="sendText('I have chest pain')">⚠️ Emergency</button>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="sttMetric">--</div>
                <div class="metric-label">STT (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="ttsMetric">--</div>
                <div class="metric-label">TTS (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="totalMetric">--</div>
                <div class="metric-label">Total (ms)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="requestsMetric">0</div>
                <div class="metric-label">Requests</div>
            </div>
        </div>
    </div>
    
    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let selectedLang = 'en';
        let isRecording = false;
        let stream = null;
        
        // DOM elements
        const micBtn = document.getElementById('micBtn');
        const statusText = document.getElementById('statusText');
        
        // Language selection
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedLang = btn.dataset.lang;
            });
        });
        
        // Request microphone permission on page load
        async function requestMicPermission() {
            try {
                statusText.textContent = '🎤 Requesting microphone access...';
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                statusText.textContent = '✅ Microphone ready! Click to start recording';
                statusText.style.color = 'var(--secondary)';
                
                // Stop the stream for now, we'll restart when recording
                stream.getTracks().forEach(t => t.stop());
                stream = null;
                
                return true;
            } catch (err) {
                console.error('Microphone permission error:', err);
                if (err.name === 'NotAllowedError') {
                    statusText.textContent = '❌ Microphone access denied. Please allow microphone in browser settings.';
                } else if (err.name === 'NotFoundError') {
                    statusText.textContent = '❌ No microphone found. Please connect a microphone.';
                } else {
                    statusText.textContent = '❌ Error: ' + err.message;
                }
                statusText.style.color = 'var(--danger)';
                return false;
            }
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {
            requestMicPermission();
        });
        
        // Click to toggle recording
        micBtn.addEventListener('click', toggleRecording);
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }
        
        async function startRecording() {
            if (isRecording) return;
            
            try {
                // Get fresh audio stream
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    } 
                });
                
                // Check for supported MIME types
                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    mimeType = 'audio/mp4';
                } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                    mimeType = 'audio/ogg';
                }
                
                mediaRecorder = new MediaRecorder(stream, { mimeType });
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: mimeType });
                        console.log('Recording complete, size:', audioBlob.size);
                        await processAudio(audioBlob);
                    }
                    // Stop all tracks
                    if (stream) {
                        stream.getTracks().forEach(t => t.stop());
                        stream = null;
                    }
                };
                
                mediaRecorder.onerror = (e) => {
                    console.error('MediaRecorder error:', e);
                    statusText.textContent = '❌ Recording error: ' + e.error;
                    statusText.style.color = 'var(--danger)';
                };
                
                // Request data every 250ms for more reliable recording
                mediaRecorder.start(250);
                isRecording = true;
                
                micBtn.classList.add('recording');
                micBtn.textContent = '⏹️';
                statusText.textContent = '🔴 Recording... Click to stop';
                statusText.className = 'status-text recording';
                
            } catch (err) {
                console.error('Recording error:', err);
                
                if (err.name === 'NotAllowedError') {
                    statusText.textContent = '❌ Microphone blocked. Click the 🔒 icon in the address bar to allow.';
                } else if (err.name === 'NotFoundError') {
                    statusText.textContent = '❌ No microphone detected.';
                } else {
                    statusText.textContent = '❌ Error: ' + err.message;
                }
                statusText.style.color = 'var(--danger)';
            }
        }
        
        function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            try {
                mediaRecorder.stop();
            } catch (e) {
                console.error('Stop error:', e);
            }
            
            isRecording = false;
            micBtn.classList.remove('recording');
            micBtn.classList.add('processing');
            micBtn.textContent = '⏳';
            statusText.textContent = '⏳ Processing with Whisper...';
            statusText.className = 'status-text processing';
        }
        
        async function processAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            formData.append('language', selectedLang);
            
            try {
                const response = await fetch('/api/speech-to-speech', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                handleResponse(data);
                
            } catch (err) {
                console.error('API error:', err);
                statusText.textContent = 'Error processing audio';
                micBtn.classList.remove('processing');
            }
        }
        
        async function sendText(text) {
            micBtn.classList.add('processing');
            statusText.textContent = '⏳ Processing...';
            statusText.className = 'status-text processing';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, language: selectedLang })
                });
                
                const data = await response.json();
                handleResponse(data);
                
            } catch (err) {
                console.error('Error:', err);
                statusText.textContent = 'Error processing request';
                micBtn.classList.remove('processing');
            }
        }
        
        function handleResponse(data) {
            micBtn.classList.remove('processing');
            micBtn.textContent = '🎤';
            
            // Add chat bubbles
            addChatBubble('user', data.transcription);
            addChatBubble('assistant', data.response, data.safety_level, data.action);
            
            // Update metrics
            if (data.stt_ms) document.getElementById('sttMetric').textContent = Math.round(data.stt_ms);
            if (data.tts_ms) document.getElementById('ttsMetric').textContent = Math.round(data.tts_ms);
            document.getElementById('totalMetric').textContent = Math.round(data.total_ms || data.latency_ms);
            
            updateMetrics();
            
            // Play audio response
            if (data.audio) {
                playAudio(data.audio);
            } else {
                statusText.textContent = '✅ Click to record';
                statusText.className = 'status-text';
                statusText.style.color = '';
            }
        }
        
        function addChatBubble(type, text, safetyLevel = null, action = null) {
            const container = document.getElementById('chatContainer');
            const bubble = document.createElement('div');
            
            let bubbleClass = `chat-bubble ${type}`;
            if (action === 'TRANSFER_TO_HUMAN') bubbleClass += ' emergency';
            bubble.className = bubbleClass;
            
            let label = type === 'user' ? '👤 You' : '🤖 Apollo';
            let badge = '';
            if (safetyLevel && type === 'assistant') {
                badge = `<span class="safety-badge ${safetyLevel}">${safetyLevel.toUpperCase()}</span>`;
            }
            
            bubble.innerHTML = `<div class="chat-label">${label}${badge}</div><div>${text}</div>`;
            container.appendChild(bubble);
            container.scrollTop = container.scrollHeight;
        }
        
        function playAudio(base64Audio) {
            statusText.textContent = '🔊 Speaking...';
            statusText.className = 'status-text speaking';
            statusText.style.color = 'var(--secondary)';
            
            const audio = new Audio('data:audio/mp3;base64,' + base64Audio);
            audio.onended = () => {
                statusText.textContent = '✅ Click to record';
                statusText.className = 'status-text';
                statusText.style.color = '';
            };
            audio.onerror = (e) => {
                console.error('Audio playback error:', e);
                statusText.textContent = '❌ Audio playback error. Try again.';
                statusText.style.color = 'var(--danger)';
            };
            audio.play().catch(err => {
                console.error('Audio play failed:', err);
                statusText.textContent = '❌ Could not play audio';
                statusText.style.color = 'var(--danger)';
            });
        }
        
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                document.getElementById('requestsMetric').textContent = data.total_requests;
            } catch (err) {}
        }
        
        updateMetrics();
    </script>
</body>
</html>
"""

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║         🏥 Apollo Voice Engine - Whisper + Edge-TTS Demo               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  🌐 Local:  http://localhost:8000                                      ║
║  📚 Docs:   http://localhost:8000/docs                                 ║
║                                                                        ║
║  ✨ Features:                                                          ║
║     🎤 OpenAI Whisper for Speech-to-Text                              ║
║     🔊 Microsoft Edge-TTS for Text-to-Speech                          ║
║     🌍 Hindi, Tamil, Telugu, Kannada, English                         ║
║     🔐 Safety classification with emergency handoff                    ║
║                                                                        ║
║  💡 Usage: Click and HOLD the mic button to record                     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
