# Apollo Hospital Voice AI Assistant v2.0
## Technical Architecture Document

**Version:** 2.0  
**Last Updated:** January 2026  
**Target:** Hackathon Demo - Regional Language Voice AI for Healthcare

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Voice AI Pipeline (8 Stages)](#3-voice-ai-pipeline-8-stages)
4. [Multi-Language Support](#4-multi-language-support)
5. [Safety & Escalation System](#5-safety--escalation-system)
6. [Session & Context Management](#6-session--context-management)
7. [RAG (Retrieval-Augmented Generation)](#7-rag-retrieval-augmented-generation)
8. [Admin Panel & Configuration](#8-admin-panel--configuration)
9. [API Reference](#9-api-reference)
10. [Data Models](#10-data-models)
11. [Deployment & Setup](#11-deployment--setup)
12. [Performance Metrics](#12-performance-metrics)

---

## 1. Executive Summary

### Problem Statement
Patients in Indian hospitals need quick access to health information in their native languages. Current systems require human operators, leading to long wait times and language barriers.

### Solution
A real-time voice AI assistant that:
- Understands patient queries in **5 regional languages** (Kannada, Tamil, Telugu, Hindi, English)
- Responds in the **same language** using native script
- Detects **medical emergencies** and escalates appropriately
- Provides **context-aware responses** using hospital knowledge base
- Achieves **<500ms latency** at **<₹2/min** cost

### Key Features

| Feature | Description |
|---------|-------------|
| **8-Stage Pipeline** | VAD → STT → Language Detection → Signals → Safety → Policy → LLM → TTS |
| **5 Languages** | Kannada, Tamil, Telugu, Hindi, English with auto-detection |
| **5 Safety Rules** | Critical symptoms, urgency, confidence, breathing issues, repeated queries |
| **RAG System** | Semantic search over FAQs, doctors, departments |
| **Session Memory** | Multi-turn conversations with context extraction |
| **Admin Panel** | Full CRUD for hospital configuration |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PATIENT INTERFACE                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Microphone │    │ File Upload │    │  Language   │    │   Session   │  │
│  │   Input     │    │   (.wav)    │    │  Selector   │    │   Display   │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘  │
│         └──────────────────┴──────────────────┘                             │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      8-STAGE VOICE PIPELINE                          │   │
│  │  ┌─────┐  ┌─────┐  ┌──────┐  ┌────────┐  ┌──────┐  ┌──────┐  ┌─────┐│   │
│  │  │ VAD │→│ STT │→│ Lang  │→│ Signals │→│Safety │→│Policy│→│ LLM ││   │
│  │  │     │  │     │  │Detect │  │Extract │  │ Gate │  │Engine│  │     ││   │
│  │  └─────┘  └─────┘  └──────┘  └────────┘  └──────┘  └──────┘  └─────┘│   │
│  │                                                              ↓       │   │
│  │                                                          ┌─────┐    │   │
│  │                                                          │ TTS │    │   │
│  │                                                          └─────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    Redis     │  │ Conversation │  │     RAG      │  │    Admin     │    │
│  │    Store     │  │   Manager    │  │  Retriever   │  │     API      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────────────┐
│                           ML MODELS                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Silero VAD  │  │Faster-Whisper│  │ LLaMA 3.1-8B │  │ Indic Parler │    │
│  │              │  │   large-v3   │  │  (4-bit QT)  │  │     TTS      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  ┌────────────────────────────────┐                                          │
│  │ Sentence Transformers          │                                          │
│  │ (paraphrase-multilingual)      │                                          │
│  └────────────────────────────────┘                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────────────┐
│                           DATA LAYER                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                            REDIS                                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │   Config   │  │  Doctors   │  │    FAQs    │  │  Sessions  │     │   │
│  │  │   (hash)   │  │   (hash)   │  │   (hash)   │  │ (hash+TTL) │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         FAISS INDEX                                   │   │
│  │  Embedding vectors for semantic search over FAQs, doctors, depts     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | File | Description |
|-----------|------|-------------|
| **FastAPI App** | `app.py` | Main application with pipeline and API endpoints |
| **Redis Store** | `models/redis_store.py` | CRUD operations for config, doctors, FAQs, sessions |
| **Conversation Manager** | `models/conversation.py` | Session lifecycle, context extraction, history |
| **RAG Retriever** | `models/embeddings.py` | Document embeddings, FAISS index, semantic search |
| **Patient UI** | `static/index.html`, `app.js` | Voice recording, pipeline visualization |
| **Admin Panel** | `static/admin.html`, `admin.js` | Hospital configuration management |
| **Seed Script** | `scripts/seed_data.py` | Load sample data into Redis |

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | HTML5, CSS3, JavaScript | Patient UI, Admin Panel |
| **Backend** | FastAPI (Python 3.10+) | REST API, Pipeline orchestration |
| **VAD** | Silero VAD | Voice Activity Detection |
| **STT** | Faster-Whisper (large-v3) | Speech-to-Text with language detection |
| **LLM** | LLaMA 3.1-8B (4-bit) | Response generation |
| **TTS** | AI4Bharat Indic Parler-TTS | Regional language speech synthesis |
| **Embeddings** | sentence-transformers (multilingual) | Semantic similarity |
| **Vector Store** | FAISS | Fast similarity search |
| **Database** | Redis | Session storage, configuration |
| **Quantization** | bitsandbytes (NF4) | LLM memory optimization |

---

## 3. Voice AI Pipeline (8 Stages)

### 3.1 Pipeline Overview

```
Audio Input → [1.VAD] → [2.STT] → [3.Lang] → [4.Signals] → [5.Safety] → [6.Policy] → [7.LLM] → [8.TTS] → Audio Output
                ↓          ↓         ↓           ↓             ↓            ↓           ↓          ↓
             Speech?    Text     Language    Intent/      Escalate?     Rules      Response     Speech
                               Detected    Urgency/Tone                Applied    Generated   Synthesized
```

### 3.2 Stage Details

#### Stage 1: Voice Activity Detection (VAD)
**Model:** Silero VAD  
**Purpose:** Detect if audio contains speech  
**Target Latency:** <50ms

```python
# Input: Audio numpy array (16kHz, mono)
# Output: {"has_speech": bool, "speech_segments": [...], "latency_ms": float}

def run_vad(audio_data: np.ndarray, sample_rate: int) -> dict:
    # Resample to 16kHz if needed
    # Run Silero VAD model
    # Return speech timestamps
```

**Key Features:**
- Resamples audio to 16kHz if needed
- Returns speech segment timestamps
- Rejects audio with no speech detected

---

#### Stage 2: Speech-to-Text (STT)
**Model:** Faster-Whisper large-v3  
**Purpose:** Transcribe audio to text with language detection  
**Target Latency:** <500ms

```python
# Input: Audio array, optional language hint
# Output: {"text": str, "detected_language": str, "confidence": float, "latency_ms": float}

def run_stt(audio_data: np.ndarray, sample_rate: int, language: str = None) -> dict:
    # Save to temp WAV file
    # Run Whisper transcription with VAD filter
    # Return transcribed text and detected language
```

**Key Features:**
- Supports 5 languages: kn, ta, te, hi, en
- Auto-detects language if not specified
- Returns language probability/confidence
- Uses beam_size=5 for accuracy

---

#### Stage 3: Language Detection (Script-Based)
**Method:** Unicode script range matching  
**Purpose:** Verify/detect language from text script  
**Target Latency:** <5ms

```python
# Unicode ranges for each script
SCRIPT_RANGES = {
    'kn': [(0x0C80, 0x0CFF)],  # Kannada
    'ta': [(0x0B80, 0x0BFF)],  # Tamil
    'te': [(0x0C00, 0x0C7F)],  # Telugu
    'hi': [(0x0900, 0x097F)],  # Hindi/Devanagari
    'en': [(0x0041, 0x005A), (0x0061, 0x007A)],  # English
}
```

**Key Features:**
- Character-level script analysis
- Returns confidence based on character distribution
- Confirms or overrides Whisper's detection

---

#### Stage 4: Layer-1 Signal Extraction
**Purpose:** Extract intent, urgency, tone, and stress level  
**Target Latency:** <10ms

```python
# Output signals
{
    "intent": "medical_query" | "appointment" | "greeting" | "admin" | "general",
    "intent_confidence": 0.0-1.0,
    "urgency": "normal" | "medium" | "high",
    "urgency_count": int,  # Number of urgency keywords found
    "tone": "calm" | "stressed",
    "stress_level": "calm" | "stressed"
}
```

**Intent Keywords (Multi-language):**

| Intent | English Examples | Kannada Examples |
|--------|------------------|------------------|
| medical_query | pain, fever, headache, cough | ನೋವು, ಜ್ವರ, ತಲೆನೋವು, ಕೆಮ್ಮು |
| appointment | book, schedule, timing, visit | ಅಪಾಯಿಂಟ್ಮೆಂಟ್, ಬುಕ್, ಸಮಯ |
| greeting | hello, namaste, good morning | ನಮಸ್ಕಾರ, ಹಲೋ |
| admin | bill, payment, insurance, report | ಬಿಲ್, ಪಾವತಿ, ವಿಮೆ |

**Urgency Keywords (High Priority):**
```
English: severe, emergency, can't breathe, chest pain, unconscious, bleeding
Hindi: गंभीर, इमरजेंसी, सांस नहीं, छाती में दर्द, बेहोश
Kannada: ತೀವ್ರ, ತುರ್ತು, ಉಸಿರು, ಎದೆ ನೋವು, ಪ್ರಜ್ಞೆ
```

---

#### Stage 5: Safety Gate
**Purpose:** Determine if query requires escalation to human  
**Target Latency:** <10ms

**5 Escalation Rules:**

| Rule | Trigger | Example |
|------|---------|---------|
| **Rule 1** | Critical symptom detected | "I have chest pain" → ESCALATE |
| **Rule 2** | Chest/breathing + medical query | "difficulty breathing" + medical intent |
| **Rule 3** | Low confidence (<0.3) + medical query | Unclear medical question |
| **Rule 4** | High urgency (2+ keywords) | "severe emergency bleeding" |
| **Rule 5** | Repeated similar query (2+ times) | Same question asked repeatedly |

```python
# Critical symptoms that always trigger escalation
CRITICAL_SYMPTOMS = {
    'en': ['chest pain', 'heart attack', "can't breathe", 'unconscious', 
           'seizure', 'stroke', 'severe bleeding', 'choking', 'suicide'],
    'hi': ['छाती में दर्द', 'हार्ट अटैक', 'सांस नहीं', 'बेहोश'],
    'kn': ['ಎದೆ ನೋವು', 'ಹೃದಯಾಘಾತ', 'ಉಸಿರಾಟ ತೊಂದರೆ', 'ಪ್ರಜ್ಞೆ ತಪ್ಪು'],
    # ... ta, te
}
```

**Output:**
```python
{
    "should_escalate": bool,
    "escalation_reason": str | None,
    "rules_triggered": ["rule_1_critical_symptom", ...]
}
```

---

#### Stage 6: Policy Engine
**Purpose:** Apply hospital-specific rules and constraints  
**Target Latency:** <5ms

```python
# Policy output (loaded from admin config)
{
    "hospital_name": "Apollo Hospital",
    "response_language": "kn",
    "max_words": 50,
    "tone": "formal",
    "include_doctor_recommendation": True,
    "emergency_number": "108",
    "helpline": "1860-500-1066",
    "disclaimer": True,
    "disclaimer_text": "Please consult a doctor...",
    "priority": "urgent" | "normal",
    "include_emergency_info": bool
}
```

**Dynamic Adjustments:**
- If escalation triggered → `priority: urgent`, `max_words: 80`
- Loads configuration from Redis (admin-configurable)

---

#### Stage 7: LLM Response Generation
**Model:** LLaMA 3.1-8B with 4-bit quantization (NF4)  
**Purpose:** Generate contextual, policy-compliant response  
**Target Latency:** <2000ms

**Prompt Structure:**
```
SYSTEM PROMPT:
- Hospital identity and contact info
- Language instruction (respond in {language} using native script)
- Word limit and tone constraints
- Doctor recommendation policy
- Emergency handling instructions

CONVERSATION CONTEXT (from session):
- Patient name, age (if extracted)
- Symptoms mentioned
- Recent conversation history (last 3 turns)

RAG CONTEXT (from semantic search):
- Matching FAQs
- Relevant doctor information
- Department details

USER MESSAGE:
{transcribed text}
```

**Generation Parameters:**
```python
{
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}
```

---

#### Stage 8: Text-to-Speech (TTS)
**Model:** AI4Bharat Indic Parler-TTS  
**Purpose:** Synthesize speech in detected language  
**Target Latency:** <1000ms

```python
# Voice description template
description = f"A female speaker delivers a clear, professional medical 
response in {language} with a calm and reassuring tone."
```

**Output:**
```python
{
    "audio_base64": str,      # Base64-encoded WAV
    "sample_rate": int,       # Usually 22050 or 24000
    "duration_ms": float,
    "latency_ms": float
}
```

### 3.3 Latency Budget

| Stage | Target | Typical |
|-------|--------|---------|
| VAD | <50ms | 20-40ms |
| STT | <500ms | 200-400ms |
| Language Detection | <5ms | 1-2ms |
| Signal Extraction | <10ms | 2-5ms |
| Safety Gate | <10ms | 2-5ms |
| Policy Engine | <5ms | 1-2ms |
| LLM | <2000ms | 500-1500ms |
| TTS | <1000ms | 300-800ms |
| **Total** | **<3580ms** | **~1500-2500ms** |

---

## 4. Multi-Language Support

### 4.1 Supported Languages

| Code | Language | Script | Native Name |
|------|----------|--------|-------------|
| `kn` | Kannada | Kannada | ಕನ್ನಡ |
| `ta` | Tamil | Tamil | தமிழ் |
| `te` | Telugu | Telugu | తెలుగు |
| `hi` | Hindi | Devanagari | हिन्दी |
| `en` | English | Latin | English |

### 4.2 Script Detection (Unicode Ranges)

```python
SCRIPT_RANGES = {
    'kn': [(0x0C80, 0x0CFF)],  # Kannada Unicode block
    'ta': [(0x0B80, 0x0BFF)],  # Tamil Unicode block
    'te': [(0x0C00, 0x0C7F)],  # Telugu Unicode block
    'hi': [(0x0900, 0x097F)],  # Devanagari Unicode block
    'en': [(0x0041, 0x005A), (0x0061, 0x007A)],  # A-Z, a-z
}
```

**Detection Algorithm:**
1. Count characters in each script range
2. Determine dominant script (highest count)
3. Calculate confidence = dominant_count / total_count

### 4.3 Language-Specific Keyword Dictionaries

#### Medical Query Keywords
```python
INTENT_KEYWORDS['medical_query'] = {
    'en': ['pain', 'fever', 'headache', 'cough', 'cold', 'stomach', 
           'doctor', 'medicine', 'treatment', 'symptom'],
    'hi': ['दर्द', 'बुखार', 'सिरदर्द', 'खांसी', 'सर्दी', 'पेट', 
           'डॉक्टर', 'दवाई', 'इलाज', 'बीमार'],
    'kn': ['ನೋವು', 'ಜ್ವರ', 'ತಲೆನೋವು', 'ಕೆಮ್ಮು', 'ಶೀತ', 
           'ಹೊಟ್ಟೆ', 'ವೈದ್ಯರು', 'ಔಷಧಿ'],
    'ta': ['வலி', 'காய்ச்சல்', 'தலைவலி', 'இருமல்', 'சளி', 
           'வயிறு', 'மருத்துவர்', 'மருந்து'],
    'te': ['నొప్పి', 'జ్వరం', 'తలనొప్పి', 'దగ్గు', 'జలుబు', 
           'కడుపు', 'డాక్టర్', 'మందు'],
}
```

#### Stress Indicators
```python
STRESS_INDICATORS = {
    'en': ['help me', 'please help', 'very worried', 'scared', 
           'afraid', "can't sleep", 'desperate', 'unbearable'],
    'hi': ['मदद करो', 'बहुत चिंता', 'डर', 'नींद नहीं', 'असहनीय'],
    'kn': ['ಸಹಾಯ', 'ಭಯ', 'ಚಿಂತೆ', 'ನಿದ್ರೆ ಬರುತ್ತಿಲ್ಲ'],
    'ta': ['உதவி', 'பயம்', 'கவலை', 'தூக்கமின்மை'],
    'te': ['సహాయం', 'భయం', 'ఆందోళన'],
}
```

---

## 5. Safety & Escalation System

### 5.1 The 5 Escalation Rules

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SAFETY GATE                                     │
│                                                                          │
│  Input: text, language, signals, session_id                              │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ RULE 1: Critical Symptom Detection                                  │ │
│  │ ─────────────────────────────────                                   │ │
│  │ IF text contains ["chest pain", "heart attack", "can't breathe",   │ │
│  │                   "unconscious", "seizure", "stroke", "suicide"]    │ │
│  │ THEN → ESCALATE (Critical symptom: {symptom})                       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓ (if not triggered)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ RULE 2: Chest/Breathing + Medical Query                             │ │
│  │ ─────────────────────────────────────────                           │ │
│  │ IF text contains ["chest", "breathe", "breathing", "heart"]         │ │
│  │ AND intent == "medical_query"                                       │ │
│  │ THEN → ESCALATE (Chest/breathing concern with medical query)        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓ (if not triggered)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ RULE 3: Low Confidence Medical Query                                │ │
│  │ ─────────────────────────────────────                               │ │
│  │ IF intent == "medical_query"                                        │ │
│  │ AND intent_confidence < 0.3                                         │ │
│  │ THEN → ESCALATE (Low confidence - needs clarification)              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓ (if not triggered)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ RULE 4: High Urgency (Multiple Keywords)                            │ │
│  │ ───────────────────────────────────────                             │ │
│  │ IF urgency_keyword_count >= 2                                       │ │
│  │ THEN → ESCALATE (Multiple urgency indicators detected)              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              ↓ (if not triggered)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ RULE 5: Repeated Similar Query                                      │ │
│  │ ──────────────────────────────                                      │ │
│  │ IF session exists                                                   │ │
│  │ AND similar_query_count_in_history >= 2                             │ │
│  │ THEN → ESCALATE (Patient may need direct assistance)                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Output: {should_escalate, escalation_reason, rules_triggered}           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Critical Symptoms (Always Escalate)

| Category | English | Hindi | Kannada |
|----------|---------|-------|---------|
| Cardiac | chest pain, heart attack | छाती में दर्द, हार्ट अटैक | ಎದೆ ನೋವು, ಹೃದಯಾಘಾತ |
| Respiratory | can't breathe, difficulty breathing | सांस नहीं | ಉಸಿರಾಟ ತೊಂದರೆ |
| Neurological | unconscious, seizure, stroke | बेहोश, दौरा | ಪ್ರಜ್ಞೆ ತಪ್ಪು |
| Trauma | severe bleeding, choking | खून बह रहा | ರಕ್ತ |
| Mental | suicide, poisoning, overdose | जहर | - |

### 5.3 Stress Indicators

Detected phrases indicating patient emotional distress:
- "help me please"
- "I'm very scared"
- "can't sleep because of this"
- "unbearable pain"
- Hindi: "मदद करो", "बहुत चिंता"
- Kannada: "ಸಹಾಯ ಮಾಡಿ", "ಭಯವಾಗುತ್ತಿದೆ"

### 5.4 Urgency Classification

| Level | Keyword Count | Action |
|-------|---------------|--------|
| Normal | 0 | Standard response |
| Medium | 1 | Include doctor recommendation |
| High | 2+ | Escalate + Emergency info |

---

## 6. Session & Context Management

### 6.1 Session Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                       SESSION LIFECYCLE                              │
│                                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐   │
│  │  Create  │ ──→ │  Active  │ ──→ │  Update  │ ──→ │  Expire  │   │
│  │ (8-char  │     │ (30 min  │     │ (on each │     │  (TTL    │   │
│  │   UUID)  │     │   TTL)   │     │   turn)  │     │ reached) │   │
│  └──────────┘     └──────────┘     └──────────┘     └──────────┘   │
│        │                │                │                          │
│        ▼                ▼                ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SESSION DATA                               │  │
│  │  {                                                            │  │
│  │    "session_id": "a1b2c3d4",                                  │  │
│  │    "created_at": "2026-01-22T10:00:00",                       │  │
│  │    "patient_name": "Ramesh",        ← Extracted               │  │
│  │    "symptoms": ["fever", "headache"], ← Accumulated           │  │
│  │    "context": {                                               │  │
│  │      "patient_age": 35,                                       │  │
│  │      "duration": "2 days",                                    │  │
│  │      "requested_doctor": "Dr. Sharma"                         │  │
│  │    },                                                         │  │
│  │    "history": [                     ← Last 10 turns           │  │
│  │      {"role": "user", "text": "...", "timestamp": "..."},     │  │
│  │      {"role": "assistant", "text": "...", "timestamp": "..."}│  │
│  │    ],                                                         │  │
│  │    "language": "kn",                                          │  │
│  │    "turn_count": 3                                            │  │
│  │  }                                                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Context Extraction Patterns

**Name Extraction:**
```python
NAME_PATTERNS = {
    'en': [r"(?:my name is|i am|i'm)\s+([A-Z][a-z]+)"],
    'hi': [r"(?:मेरा नाम|मैं)\s+(\S+)"],
    'kn': [r"(?:ನನ್ನ ಹೆಸರು|ನಾನು)\s+(\S+)"],
}
```

**Symptom Extraction:**
```python
SYMPTOM_KEYWORDS = {
    'en': {
        'fever': ['fever', 'temperature', 'hot'],
        'headache': ['headache', 'head pain', 'migraine'],
        'cough': ['cough', 'coughing'],
        'chest_pain': ['chest pain', 'heart pain'],
        'breathing': ['breathless', 'breathing difficulty'],
    },
    'kn': {
        'fever': ['ಜ್ವರ', 'ಬಿಸಿ'],
        'headache': ['ತಲೆನೋವು'],
        # ...
    }
}
```

**Duration Extraction:**
```python
DURATION_PATTERNS = {
    'en': [r"(?:since|for)\s+(\d+)\s+(day|days|week|hours)"],
    'hi': [r"(\d+)\s+(?:दिन|हफ्ते)\s+से"],
}
```

### 6.3 Conversation History

- **Max Turns:** 10 (20 messages: user + assistant pairs)
- **Trimming:** Oldest messages removed when limit exceeded
- **Context String:** Last 3 turns formatted for LLM prompt
- **Repeat Detection:** Word overlap analysis (>70% = similar)

---

## 7. RAG (Retrieval-Augmented Generation)

### 7.1 Document Types

| Type | Fields | Example |
|------|--------|---------|
| **FAQ** | question, answer, question_kn, answer_kn, category | "What are visiting hours?" |
| **Doctor** | name, name_kn, specialization, department, timings, available_days | "Dr. Ananya Sharma - General Medicine" |
| **Department** | name, name_kn, floor, timings, contact | "Emergency - Ground Floor - 24/7" |

### 7.2 Embedding Model

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Dimensions | 384 |
| Languages | 50+ including Indic |
| Size | ~120MB |
| Speed | Fast inference |

### 7.3 Search & Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG RETRIEVER                                 │
│                                                                      │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐  │
│  │   Document     │     │   Embedding    │     │     FAISS      │  │
│  │    Loader      │ ──→ │    Manager     │ ──→ │     Index      │  │
│  │  (Redis/JSON)  │     │ (Multilingual) │     │  (L2 Distance) │  │
│  └────────────────┘     └────────────────┘     └────────────────┘  │
│                                                        │            │
│                                                        ▼            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      SEARCH FLOW                              │  │
│  │                                                               │  │
│  │  Query: "ಡಾಕ್ಟರ್ ಶರ್ಮಾ ಯಾವಾಗ available?"                       │  │
│  │           ↓                                                   │  │
│  │  [1] Embed query using multilingual model                     │  │
│  │           ↓                                                   │  │
│  │  [2] FAISS k-NN search (k=3)                                  │  │
│  │           ↓                                                   │  │
│  │  [3] Rerank: Boost results with language-specific fields      │  │
│  │           ↓                                                   │  │
│  │  [4] Format for LLM prompt                                    │  │
│  │                                                               │  │
│  │  Result:                                                      │  │
│  │  "Doctor 1:                                                   │  │
│  │   Name: Dr. Ananya Sharma                                     │  │
│  │   Specialization: General Medicine                            │  │
│  │   Available: Monday, Wednesday, Friday                        │  │
│  │   Timing: 9:00 AM - 1:00 PM"                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.4 Prompt Augmentation

**LLM Prompt with RAG Context:**
```
SYSTEM: You are Apollo Hospital's AI assistant...

PATIENT INFORMATION:
Name: Ramesh
Symptoms: fever, headache
Duration: 2 days

RECENT CONVERSATION:
Patient: I have fever since 2 days
Assistant: I understand you have fever...

RELEVANT INFORMATION:
Doctor 1:
Name: Dr. Ananya Sharma
Specialization: General Medicine
Department: Internal Medicine
Available: Monday, Wednesday, Friday
Timing: 9:00 AM - 1:00 PM

FAQ 1:
Q: How do I book an appointment?
A: You can book through our website apollo.com or call 1860-500-1066...

USER: {current query}
```

---

## 8. Admin Panel & Configuration

### 8.1 Admin Panel Interface

**URL:** `http://localhost:8000/admin`

**Tabs:**

| Tab | Purpose | CRUD Operations |
|-----|---------|-----------------|
| **Settings** | Hospital config, AI behavior | Read, Update |
| **Doctors** | Manage doctor information | Create, Read, Update, Delete |
| **Departments** | Manage department info | Create, Read, Delete |
| **FAQs** | Manage knowledge base | Create, Read, Update, Delete |
| **Escalation** | Custom escalation keywords | Read, Update |

### 8.2 Settings Configuration

```javascript
{
  // Hospital Information
  "hospital_name": "Apollo Hospital",
  "city": "Bengaluru",
  "emergency_number": "108",
  "helpline": "1860-500-1066",
  
  // AI Behavior
  "tone": "formal" | "friendly",
  "max_words": 50,
  "primary_language": "en",
  "session_timeout_minutes": 30,
  
  // Options
  "disclaimer_required": true,
  "always_recommend_doctor": true,
  "tts_enabled": true,
  
  // Custom Disclaimer
  "disclaimer_text": "This is general information only..."
}
```

### 8.3 Doctor Management

**Fields:**
| Field | Type | Required | Multi-lang |
|-------|------|----------|------------|
| name | string | Yes | name_kn, name_hi, name_ta, name_te |
| specialization | string | Yes | - |
| department | string | Yes | - |
| available_days | array | Yes | - |
| timings | string | No | - |
| languages | array | No | - |
| consultation_fee | string | No | - |

**Example:**
```json
{
  "id": "d001",
  "name": "Dr. Ananya Sharma",
  "name_kn": "ಡಾ. ಅನನ್ಯ ಶರ್ಮಾ",
  "specialization": "General Medicine",
  "department": "Internal Medicine",
  "available_days": ["Monday", "Wednesday", "Friday"],
  "timings": "9:00 AM - 1:00 PM",
  "languages": ["en", "hi", "kn"],
  "consultation_fee": "500"
}
```

### 8.4 FAQ Management

**Fields:**
| Field | Type | Required | Multi-lang |
|-------|------|----------|------------|
| question | string | Yes | question_kn, question_hi |
| answer | string | Yes | answer_kn, answer_hi |
| category | string | No | - |

**Categories:** general, appointment, billing, emergency, visiting

---

## 9. API Reference

### 9.1 Main Endpoints

#### `GET /`
**Description:** Serve patient UI  
**Response:** `text/html`

#### `GET /admin`
**Description:** Serve admin panel  
**Response:** `text/html`

#### `GET /health`
**Description:** Health check  
**Response:**
```json
{
  "status": "healthy" | "loading",
  "models_loaded": true,
  "device": "cuda" | "cpu",
  "redis_connected": true,
  "rag_ready": true,
  "supported_languages": {"kn": "Kannada", ...}
}
```

#### `POST /process`
**Description:** Process audio through full pipeline  
**Content-Type:** `multipart/form-data`

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio | File | Yes | Audio file (wav, webm, mp3) |
| language | string | No | Language hint (kn, ta, te, hi) |
| session_id | string | No | Existing session ID |

**Response:**
```json
{
  "success": true,
  "session_id": "a1b2c3d4",
  "transcription": "ನನಗೆ ಜ್ವರ ಇದೆ",
  "response": "ನಿಮಗೆ ಜ್ವರ ಇದ್ದರೆ...",
  "audio_response": "base64...",
  "audio_sample_rate": 22050,
  "language": "kn",
  "language_name": "Kannada",
  "escalation": {
    "should_escalate": false,
    "reason": null,
    "rules_triggered": []
  },
  "context": {
    "used_rag": true,
    "used_history": true
  },
  "pipeline": {
    "vad": {"latency_ms": 25.5},
    "stt": {"latency_ms": 350.2, "confidence": 0.95},
    "language_detection": {"latency_ms": 1.2},
    "layer1_signals": {
      "latency_ms": 3.5,
      "intent": "medical_query",
      "urgency": "normal",
      "tone": "calm",
      "stress_level": "calm"
    },
    "safety_gate": {"latency_ms": 2.1},
    "policy": {"latency_ms": 1.5},
    "llm": {"latency_ms": 850.3},
    "tts": {"latency_ms": 450.8, "duration_ms": 2500}
  },
  "total_latency_ms": 1685.1
}
```

#### `GET /languages`
**Response:**
```json
{
  "languages": {
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "en": "English"
  }
}
```

### 9.2 Session Endpoints

#### `POST /api/session`
**Description:** Create new session  
**Response:**
```json
{"session_id": "a1b2c3d4"}
```

#### `GET /api/session/{session_id}`
**Description:** Get session data  
**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "patient_name": "Ramesh",
  "symptoms": ["fever", "headache"],
  "context": {...},
  "history": [...],
  "turn_count": 3
}
```

### 9.3 Admin Endpoints

#### Config
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/config` | Get hospital config |
| POST | `/api/admin/config` | Update config |

#### Doctors
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/doctors` | List all doctors |
| POST | `/api/admin/doctors` | Add doctor |
| PUT | `/api/admin/doctors/{id}` | Update doctor |
| DELETE | `/api/admin/doctors/{id}` | Delete doctor |

#### Departments
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/departments` | List all departments |
| POST | `/api/admin/departments` | Add department |
| DELETE | `/api/admin/departments/{id}` | Delete department |

#### FAQs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/admin/faqs` | List all FAQs |
| POST | `/api/admin/faqs` | Add FAQ |
| PUT | `/api/admin/faqs/{id}` | Update FAQ |
| DELETE | `/api/admin/faqs/{id}` | Delete FAQ |

---

## 10. Data Models

### 10.1 Redis Schema

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REDIS KEY STRUCTURE                          │
│                                                                      │
│  hospital:config                    HASH                             │
│  ├── hospital_name: "Apollo Hospital"                                │
│  ├── city: "Bengaluru"                                               │
│  ├── emergency_number: "108"                                         │
│  ├── tone: "formal"                                                  │
│  ├── max_words: "50"                                                 │
│  ├── disclaimer_required: "true"                                     │
│  └── ...                                                             │
│                                                                      │
│  hospital:doctors:ids               SET                              │
│  └── {"d001", "d002", "d003", ...}                                   │
│                                                                      │
│  hospital:doctors:{id}              HASH                             │
│  ├── name: "Dr. Ananya Sharma"                                       │
│  ├── name_kn: "ಡಾ. ಅನನ್ಯ ಶರ್ಮಾ"                                       │
│  ├── specialization: "General Medicine"                              │
│  ├── department: "Internal Medicine"                                 │
│  ├── available_days: "[\"Monday\", \"Wednesday\"]"  (JSON)           │
│  ├── timings: "9:00 AM - 1:00 PM"                                    │
│  └── ...                                                             │
│                                                                      │
│  hospital:departments:ids           SET                              │
│  hospital:departments:{id}          HASH                             │
│                                                                      │
│  hospital:faqs:ids                  SET                              │
│  hospital:faqs:{id}                 HASH                             │
│                                                                      │
│  session:{session_id}               HASH (with 30min TTL)            │
│  ├── session_id: "a1b2c3d4"                                          │
│  ├── created_at: "2026-01-22T10:00:00"                               │
│  ├── patient_name: "Ramesh"                                          │
│  ├── symptoms: "[\"fever\", \"headache\"]"  (JSON)                   │
│  ├── context: "{...}"  (JSON)                                        │
│  ├── history: "[...]"  (JSON)                                        │
│  └── updated_at: "2026-01-22T10:05:00"                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.2 Pydantic Models

```python
class ConfigUpdate(BaseModel):
    hospital_name: Optional[str]
    city: Optional[str]
    emergency_number: Optional[str]
    helpline: Optional[str]
    tone: Optional[str]  # "formal" | "friendly"
    max_words: Optional[int]
    disclaimer_text: Optional[str]
    disclaimer_required: Optional[bool]
    always_recommend_doctor: Optional[bool]
    tts_enabled: Optional[bool]
    session_timeout_minutes: Optional[int]

class DoctorCreate(BaseModel):
    id: Optional[str]
    name: str
    specialization: str
    department: str
    available_days: List[str]
    timings: str
    languages: List[str] = ["en"]
    consultation_fee: Optional[str]
    name_kn: Optional[str]
    name_hi: Optional[str]
    name_ta: Optional[str]
    name_te: Optional[str]

class FAQCreate(BaseModel):
    id: Optional[str]
    question: str
    answer: str
    category: Optional[str] = "general"
    question_kn: Optional[str]
    question_hi: Optional[str]
    answer_kn: Optional[str]
```

---

## 11. Deployment & Setup

### 11.1 Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| Redis | 6.0+ | Session/Config storage |
| CUDA (optional) | 11.8+ | GPU acceleration |
| Memory | 16GB+ RAM | Model loading |
| Storage | 20GB+ | Model weights |

### 11.2 Installation

```bash
# Clone repository
cd /path/to/Demo\ Day

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis
redis-server

# Seed sample data
python scripts/seed_data.py --clear

# Set HuggingFace token (for LLaMA access)
export HF_TOKEN=your_token_here

# Start server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 11.3 Quick Start Script

```bash
chmod +x run.sh
./run.sh
```

**Script performs:**
1. Creates virtual environment (if not exists)
2. Installs dependencies
3. Checks CUDA availability
4. Starts Redis (if not running)
5. Seeds sample data
6. Starts FastAPI server

### 11.4 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace API token for LLaMA |
| `REDIS_HOST` | localhost | Redis server host |
| `REDIS_PORT` | 6379 | Redis server port |

### 11.5 Endpoints After Startup

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Patient Voice UI |
| http://localhost:8000/admin | Admin Panel |
| http://localhost:8000/docs | Swagger API Docs |
| http://localhost:8000/health | Health Check |

---

## 12. Performance Metrics

### 12.1 Latency Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Total Pipeline Latency | <3500ms | ~1500-2500ms |
| First Token (LLM) | <1000ms | ~500-800ms |
| TTS Generation | <1000ms | ~300-800ms |
| End-to-End (user speaks → audio plays) | <5000ms | ~3000-4000ms |

### 12.2 Cost Analysis

| Component | Cost Estimate |
|-----------|---------------|
| Whisper (STT) | Free (local) |
| LLaMA 3.1-8B | Free (local) |
| Indic Parler TTS | Free (local) |
| Redis | Free (self-hosted) |
| GPU (if rented) | ~₹50-100/hour |
| **Per Minute Cost** | **<₹2/min** (target met) |

### 12.3 Model Memory Footprint

| Model | Memory (GPU) | Memory (CPU) |
|-------|--------------|--------------|
| Silero VAD | ~50MB | ~50MB |
| Faster-Whisper large-v3 | ~3GB | ~6GB |
| LLaMA 3.1-8B (4-bit) | ~5GB | N/A |
| Indic Parler TTS | ~1GB | ~2GB |
| Sentence Transformers | ~500MB | ~500MB |
| **Total** | **~10GB** | ~8GB (no LLM) |

### 12.4 Scalability Considerations

| Aspect | Current | Scalable Option |
|--------|---------|-----------------|
| Sessions | Redis (single) | Redis Cluster |
| RAG Index | In-memory FAISS | Pinecone/Weaviate |
| LLM | Single GPU | vLLM + load balancer |
| TTS | Single instance | Queue-based workers |

---

## Appendix A: Sample Data Structure

**File:** `data/sample_hospital.json`

```json
{
  "hospital": {
    "name": "Apollo Hospital",
    "city": "Bengaluru",
    "emergency_number": "108",
    "helpline": "1860-500-1066"
  },
  "doctors": [
    {
      "id": "d001",
      "name": "Dr. Ananya Sharma",
      "name_kn": "ಡಾ. ಅನನ್ಯ ಶರ್ಮಾ",
      "specialization": "General Medicine",
      "department": "Internal Medicine",
      "available_days": ["Monday", "Wednesday", "Friday"],
      "timings": "9:00 AM - 1:00 PM",
      "languages": ["en", "hi", "kn"]
    }
    // ... 7 more doctors
  ],
  "departments": [
    {
      "id": "dept001",
      "name": "Emergency",
      "name_kn": "ತುರ್ತು ವಿಭಾಗ",
      "floor": "Ground Floor",
      "timings": "24/7",
      "contact": "080-2630-4050"
    }
    // ... 7 more departments
  ],
  "faqs": [
    {
      "id": "faq001",
      "question": "What are the visiting hours?",
      "question_kn": "ಭೇಟಿ ಸಮಯ ಏನು?",
      "answer": "Visiting hours are 10:00 AM - 12:00 PM and 5:00 PM - 7:00 PM",
      "answer_kn": "ಭೇಟಿ ಸಮಯ 10:00 AM ರಿಂದ 12:00 PM ಮತ್ತು 5:00 PM ರಿಂದ 7:00 PM",
      "category": "visiting"
    }
    // ... 11 more FAQs
  ]
}
```

---

## Appendix B: File Structure

```
/Demo Day/
├── app.py                          # Main FastAPI application (900+ lines)
├── requirements.txt                # Python dependencies
├── run.sh                          # Startup script
│
├── models/
│   ├── __init__.py                 # Package exports
│   ├── redis_store.py              # Redis CRUD operations (480+ lines)
│   ├── conversation.py             # Session management (490+ lines)
│   └── embeddings.py               # RAG system (370+ lines)
│
├── static/
│   ├── index.html                  # Patient voice UI
│   ├── app.js                      # Patient UI JavaScript
│   ├── styles.css                  # Patient UI styles
│   ├── admin.html                  # Admin panel UI
│   ├── admin.js                    # Admin panel JavaScript
│   └── admin.css                   # Admin panel styles
│
├── data/
│   └── sample_hospital.json        # Sample Apollo Hospital data
│
├── scripts/
│   └── seed_data.py                # Load JSON data into Redis
│
└── docs/
    └── ARCHITECTURE.md             # This document
```

---

## Appendix C: Demo Talking Points

### For Hackathon Presentation

1. **Problem**: "Patients wait 5-10 minutes to speak with hospital staff, and language barriers cause miscommunication"

2. **Solution**: "Real-time voice AI that understands 5 Indian languages and responds in under 3 seconds"

3. **Key Demo Moments**:
   - Speak in Kannada → Get response in Kannada (native script)
   - Say "chest pain" → Watch escalation alert trigger
   - Ask about a doctor → See RAG retrieve correct info
   - Ask follow-up → Observe context retention

4. **Technical Highlights**:
   - 8-stage pipeline with latency visualization
   - 5 safety rules with explanations
   - Admin panel for hospital customization
   - Works offline (all models local)

5. **Cost**: "<₹2/minute - 10x cheaper than human operators"

---

**Document End**

*Generated for Apollo Hospital Voice AI Assistant v2.0 - Hackathon Demo*
