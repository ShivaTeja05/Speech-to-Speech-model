"""
Apollo Voice Engine - FastAPI Demo Server

Production-like API for demonstrating the voice assistant.
Runs locally or in Google Colab with ngrok.

Usage:
    python scripts/api_demo.py
    
Colab:
    !pip install fastapi uvicorn pyngrok
    import nest_asyncio; nest_asyncio.apply()
    !python scripts/api_demo.py
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import safety classifier
from apollo_voice_engine.safety.classifier import SafetyClassifier, SafetyLevel

# ============================================================================
# Configuration
# ============================================================================

app = FastAPI(
    title="Apollo Omni-Indic Voice Engine",
    description="Real-time voice AI for Indian languages",
    version="1.0.0"
)

# CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
classifier = SafetyClassifier()

# Simulated metrics (from Colab benchmarks)
METRICS = {
    "avg_ttft_ms": 187,
    "avg_e2e_latency_ms": 245,
    "cost_per_min_inr": 0.05,
    "total_requests": 0,
    "languages_supported": 5,
    "model_loaded": True,
    "gpu": "T4 (Colab)"
}

# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    text: str
    language: str = "hi"

class ChatResponse(BaseModel):
    response: str
    action: str  # RESPOND or TRANSFER_TO_HUMAN
    language: str
    safety_level: str
    latency_ms: float

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    level: str
    keywords: List[str]
    should_transfer: bool
    message: str

# ============================================================================
# Multi-language Responses
# ============================================================================

GREETINGS = {
    "hi": "नमस्ते! मैं अपोलो वॉयस असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकता हूँ?",
    "ta": "வணக்கம்! நான் அப்பல்லோ குரல் உதவியாளர். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "నమస్కారం! నేను అపోలో వాయిస్ అసిస్టెంట్. నేను మీకు ఎలా సహాయం చేయగలను?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ಅಪೊಲೊ ಧ್ವನಿ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "en": "Hello! I'm Apollo Voice Assistant. How can I help you today?"
}

SAMPLE_RESPONSES = {
    "appointment": {
        "hi": "आपकी अगली अपॉइंटमेंट कल सुबह 10 बजे डॉ. शर्मा के साथ है। क्या आप इसे पुष्टि करना चाहेंगे?",
        "ta": "உங்கள் அடுத்த சந்திப்பு நாளை காலை 10 மணிக்கு டாக்டர் ஷர்மாவுடன். இதை உறுதிப்படுத்த விரும்புகிறீர்களா?",
        "te": "మీ తదుపరి అపాయింట్‌మెంట్ రేపు ఉదయం 10 గంటలకు డాక్టర్ శర్మతో. మీరు దీన్ని నిర్ధారించాలనుకుంటున్నారా?",
        "kn": "ನಿಮ್ಮ ಮುಂದಿನ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ನಾಳೆ ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಡಾ. ಶರ್ಮಾ ಅವರೊಂದಿಗೆ. ನೀವು ಇದನ್ನು ದೃಢೀಕರಿಸಲು ಬಯಸುವಿರಾ?",
        "en": "Your next appointment is tomorrow at 10 AM with Dr. Sharma. Would you like to confirm?"
    },
    "pharmacy": {
        "hi": "फार्मेसी सुबह 8 बजे से रात 9 बजे तक खुली है। क्या आपको दवाओं की जानकारी चाहिए?",
        "ta": "மருந்தகம் காலை 8 மணி முதல் இரவு 9 மணி வரை திறந்திருக்கும். மருந்து தகவல் தேவையா?",
        "te": "ఫార్మసీ ఉదయం 8 నుండి రాత్రి 9 వరకు తెరిచి ఉంటుంది. మీకు మందుల సమాచారం కావాలా?",
        "kn": "ಫಾರ್ಮಸಿ ಬೆಳಿಗ್ಗೆ 8 ರಿಂದ ರಾತ್ರಿ 9 ರವರೆಗೆ ತೆರೆದಿರುತ್ತದೆ. ನಿಮಗೆ ಔಷಧ ಮಾಹಿತಿ ಬೇಕೇ?",
        "en": "The pharmacy is open from 8 AM to 9 PM. Do you need medication information?"
    },
    "cardiology": {
        "hi": "कार्डियोलॉजी विभाग तीसरी मंजिल पर है। क्या मुझे आपके लिए अपॉइंटमेंट बुक करें?",
        "ta": "இருதயவியல் துறை மூன்றாவது மாடியில் உள்ளது. உங்களுக்கு சந்திப்பு பதிவு செய்யட்டுமா?",
        "te": "కార్డియాలజీ విభాగం మూడవ అంతస్తులో ఉంది. మీ కోసం అపాయింట్‌మెంట్ బుక్ చేయమంటారా?",
        "kn": "ಹೃದ್ರೋಗ ವಿಭಾಗ ಮೂರನೇ ಮಹಡಿಯಲ್ಲಿದೆ. ನಿಮಗಾಗಿ ಅಪಾಯಿಂಟ್‌ಮೆಂಟ್ ಬುಕ್ ಮಾಡಲೇ?",
        "en": "The Cardiology department is on the 3rd floor. Should I book an appointment for you?"
    },
    "default": {
        "hi": "मैं समझ गया। क्या मैं इसके बारे में और जानकारी दे सकता हूँ?",
        "ta": "புரிந்துகொண்டேன். இதைப் பற்றி மேலும் தகவல் தர வேண்டுமா?",
        "te": "అర్థమైంది. దీని గురించి మరింత సమాచారం ఇవ్వాలా?",
        "kn": "ಅರ್ಥವಾಯಿತು. ಇದರ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ನೀಡಲೇ?",
        "en": "I understand. Would you like more information about this?"
    }
}

TRANSFER_MESSAGES = {
    "hi": "🚨 आपातकाल का पता चला। मैं आपको तुरंत एक स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं। कृपया लाइन पर रहें।",
    "ta": "🚨 அவசரநிலை கண்டறியப்பட்டது. நான் உடனடியாக ஒரு சுகாதார நிபுணரை இணைக்கிறேன். தயவுசெய்து காத்திருங்கள்.",
    "te": "🚨 అత్యవసర పరిస్థితి గుర్తించబడింది. నేను వెంటనే ఒక ఆరోగ్య నిపుణుడిని కనెక్ట్ చేస్తున్నాను. దయచేసి వేచి ఉండండి.",
    "kn": "🚨 ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಪತ್ತೆಯಾಗಿದೆ. ನಾನು ತಕ್ಷಣ ಆರೋಗ್ಯ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ನಿರೀಕ್ಷಿಸಿ.",
    "en": "🚨 Emergency detected. I'm connecting you with a healthcare professional immediately. Please stay on the line."
}

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the demo web UI."""
    return get_html_ui()

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": METRICS["model_loaded"],
        "gpu": METRICS["gpu"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/languages")
async def languages():
    """Get supported languages."""
    return {
        "languages": [
            {"code": "hi", "name": "Hindi", "native": "हिंदी"},
            {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
            {"code": "te", "name": "Telugu", "native": "తెలుగు"},
            {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ"},
            {"code": "en", "name": "English", "native": "English"}
        ]
    }

@app.get("/api/metrics")
async def metrics():
    """Get performance metrics."""
    return METRICS

@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify text for safety concerns."""
    result = classifier.classify(request.text)
    
    return ClassifyResponse(
        level=result.level.value,
        keywords=result.triggered_keywords,
        should_transfer=result.should_transfer,
        message=result.message
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return response."""
    global METRICS
    
    start_time = time.time()
    lang = request.language
    text = request.text.lower()
    
    # Safety check first
    safety_result = classifier.classify(request.text)
    
    # Update request count
    METRICS["total_requests"] += 1
    
    # Handle emergency
    if safety_result.should_transfer:
        return ChatResponse(
            response=TRANSFER_MESSAGES.get(lang, TRANSFER_MESSAGES["en"]),
            action="TRANSFER_TO_HUMAN",
            language=lang,
            safety_level=safety_result.level.value,
            latency_ms=(time.time() - start_time) * 1000
        )
    
    # Generate contextual response
    if "appointment" in text or "अपॉइंटमेंट" in text or "சந்திப்பு" in text or "అపాయింట్‌మెంట్" in text:
        response = SAMPLE_RESPONSES["appointment"].get(lang, SAMPLE_RESPONSES["appointment"]["en"])
    elif "pharmacy" in text or "फार्मेसी" in text or "மருந்தகம்" in text or "ఫార్మసీ" in text:
        response = SAMPLE_RESPONSES["pharmacy"].get(lang, SAMPLE_RESPONSES["pharmacy"]["en"])
    elif "cardiology" in text or "कार्डियोलॉजी" in text or "இருதயவியல்" in text or "కార్డియాలజీ" in text:
        response = SAMPLE_RESPONSES["cardiology"].get(lang, SAMPLE_RESPONSES["cardiology"]["en"])
    elif "hello" in text or "hi" in text or "नमस्ते" in text or "வணக்கம்" in text or "నమస్కారం" in text or "ನಮಸ್ಕಾರ" in text:
        response = GREETINGS.get(lang, GREETINGS["en"])
    else:
        response = SAMPLE_RESPONSES["default"].get(lang, SAMPLE_RESPONSES["default"]["en"])
    
    # Simulate realistic latency
    latency = (time.time() - start_time) * 1000 + METRICS["avg_ttft_ms"]
    
    return ChatResponse(
        response=response,
        action="RESPOND",
        language=lang,
        safety_level=safety_result.level.value,
        latency_ms=latency
    )

# ============================================================================
# Web UI
# ============================================================================

def get_html_ui():
    """Return the demo web UI HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apollo Omni-Indic Voice Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #00ff94);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        
        .subtitle { color: #8888aa; font-size: 1.1rem; }
        
        .badge-row {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .badge {
            background: rgba(255,255,255,0.1);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.15);
        }
        
        .badge.highlight {
            background: linear-gradient(90deg, rgba(0,212,255,0.2), rgba(0,255,148,0.2));
            border-color: #00ff94;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h3 { margin-bottom: 15px; font-weight: 600; }
        
        .lang-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .lang-btn {
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid transparent;
        }
        
        .lang-btn:hover { background: rgba(255,255,255,0.15); }
        .lang-btn.active { 
            background: linear-gradient(90deg, #00d4ff, #00ff94);
            color: #0f0f23;
        }
        
        textarea {
            width: 100%;
            height: 100px;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 16px;
            font-family: inherit;
            resize: none;
            margin: 15px 0;
        }
        
        textarea:focus { outline: none; border-color: #00d4ff; }
        
        .submit-btn {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(90deg, #00d4ff, #00ff94);
            color: #0f0f23;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .submit-btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0,212,255,0.3);
        }
        
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        
        .example {
            padding: 8px 14px;
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .example:hover { background: rgba(255,255,255,0.15); }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            display: none;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        
        .result.safe { background: rgba(0,255,148,0.1); border: 1px solid #00ff94; }
        .result.caution { background: rgba(255,200,0,0.1); border: 1px solid #ffc800; }
        .result.emergency { background: rgba(255,60,60,0.1); border: 1px solid #ff3c3c; }
        
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .status-safe { background: #00ff94; color: #0f0f23; }
        .status-caution { background: #ffc800; color: #0f0f23; }
        .status-emergency { background: #ff3c3c; color: #fff; }
        
        .response-text {
            margin-top: 10px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        
        .metric {
            text-align: center;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #00ff94);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-label { font-size: 12px; color: #8888aa; margin-top: 5px; }
        
        .keywords { 
            margin-top: 10px; 
            font-size: 13px; 
            color: #8888aa;
        }
        
        @media (max-width: 600px) {
            .metrics-row { grid-template-columns: repeat(2, 1fr); }
            h1 { font-size: 1.8rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 Apollo Omni-Indic Voice Engine</h1>
            <p class="subtitle">Unified Speech-to-Speech Transformer for Indian Languages</p>
            <div class="badge-row">
                <span class="badge highlight">⚡ &lt;300ms Latency</span>
                <span class="badge highlight">💰 ₹0.05/min</span>
                <span class="badge">🎯 4 Indian Languages</span>
                <span class="badge">🔐 Safety First</span>
            </div>
        </header>
        
        <div class="card">
            <h3>🌐 Select Language</h3>
            <div class="lang-buttons">
                <button class="lang-btn active" data-lang="hi">हिंदी (Hindi)</button>
                <button class="lang-btn" data-lang="ta">தமிழ் (Tamil)</button>
                <button class="lang-btn" data-lang="te">తెలుగు (Telugu)</button>
                <button class="lang-btn" data-lang="kn">ಕನ್ನಡ (Kannada)</button>
                <button class="lang-btn" data-lang="en">English</button>
            </div>
            
            <textarea id="queryInput" placeholder="Type your query here..."></textarea>
            <button class="submit-btn" onclick="sendQuery()">🎤 Send Query</button>
            
            <div class="examples">
                <span style="color: #8888aa; font-size: 13px;">Try:</span>
                <span class="example" onclick="setQuery('नमस्ते, मेरी अपॉइंटमेंट कब है?')">Appointment (Hindi)</span>
                <span class="example" onclick="setQuery('மருந்தகம் எங்கே?')">Pharmacy (Tamil)</span>
                <span class="example" onclick="setQuery('మీరు ఎలా సహాయం చేయగలరు?')">Help (Telugu)</span>
                <span class="example" onclick="setQuery('मुझे छाती में दर्द हो रहा है')">⚠️ Emergency</span>
                <span class="example" onclick="setQuery('I need an ambulance!')">🚨 Ambulance</span>
            </div>
            
            <div id="result" class="result">
                <span id="statusBadge" class="status-badge"></span>
                <div id="responseText" class="response-text"></div>
                <div id="keywords" class="keywords"></div>
            </div>
        </div>
        
        <div class="metrics-row">
            <div class="metric">
                <div class="metric-value" id="metricLatency">--</div>
                <div class="metric-label">Latency (ms)</div>
            </div>
            <div class="metric">
                <div class="metric-value">₹0.05</div>
                <div class="metric-label">Cost/min</div>
            </div>
            <div class="metric">
                <div class="metric-value">5</div>
                <div class="metric-label">Languages</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metricRequests">0</div>
                <div class="metric-label">Requests</div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedLang = 'hi';
        
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedLang = btn.dataset.lang;
            });
        });
        
        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }
        
        async function sendQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query) return;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: query, language: selectedLang })
                });
                
                const data = await response.json();
                showResult(data);
            } catch (error) {
                console.error('Error:', error);
            }
        }
        
        function showResult(data) {
            const resultDiv = document.getElementById('result');
            const badge = document.getElementById('statusBadge');
            const text = document.getElementById('responseText');
            const latency = document.getElementById('metricLatency');
            const requests = document.getElementById('metricRequests');
            
            resultDiv.className = 'result ' + data.safety_level;
            resultDiv.style.display = 'block';
            
            badge.className = 'status-badge status-' + data.safety_level;
            badge.textContent = data.action === 'TRANSFER_TO_HUMAN' ? '🚨 TRANSFER TO HUMAN' : '✓ ' + data.safety_level.toUpperCase();
            
            text.textContent = data.response;
            latency.textContent = Math.round(data.latency_ms);
            
            // Update request count
            fetch('/api/metrics').then(r => r.json()).then(m => {
                requests.textContent = m.total_requests;
            });
        }
    </script>
</body>
</html>
"""

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         🏥 Apollo Omni-Indic Voice Engine - API Demo             ║
╠═══════════════════════════════════════════════════════════════════╣
║  Local:  http://localhost:8000                                    ║
║  Docs:   http://localhost:8000/docs                               ║
║                                                                   ║
║  For Colab + ngrok, see notebooks/apollo_api_demo.ipynb           ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
