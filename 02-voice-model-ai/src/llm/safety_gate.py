"""
Text-based Safety Gate for the Voice AI pipeline.

Ported from the Demo-THIT prototype (`app.py`) and adapted to run inside the
voice-model-ai LLM layer. This is a *text*-side complement to the acoustic
emergency classifier: it inspects the ASR transcript for critical symptoms,
urgency, and distress, and decides whether the turn should be escalated
(handover / emergency guidance) before the LLM answers.

Self-contained: no Redis or Flask dependency. Optional repeated-query
detection is passed in by the caller (`repeat_count`).
"""

import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Keyword lexicons (multilingual). Ported verbatim from Demo-THIT.
# ---------------------------------------------------------------------------

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
    },
}

URGENCY_HIGH = {
    'en': ['severe', 'extreme', 'unbearable', 'emergency', 'urgent', 'immediately', "can't breathe", 'chest pain', 'unconscious', 'bleeding', 'accident', 'critical', 'dying', 'collapsed'],
    'hi': ['गंभीर', 'बहुत', 'असहनीय', 'इमरजेंसी', 'तुरंत', 'सांस नहीं', 'छाती में दर्द', 'बेहोश', 'खून'],
    'ta': ['கடுமையான', 'அவசரம்', 'உடனடி', 'மூச்சு', 'நெஞ்சு வலி', 'மயக்கம்', 'இரத்தம்'],
    'te': ['తీవ్రమైన', 'అత్యవసరం', 'వెంటనే', 'ఊపిరి', 'ఛాతీ నొప్పి', 'స్పృహ', 'రక్తం'],
    'kn': ['ತೀವ್ರ', 'ತುರ್ತು', 'ತಕ್ಷಣ', 'ಉಸಿರು', 'ಎದೆ ನೋವು', 'ಪ್ರಜ್ಞೆ', 'ರಕ್ತ'],
}

CRITICAL_SYMPTOMS = {
    'en': ['chest pain', 'heart attack', "can't breathe", 'difficulty breathing', 'unconscious', 'seizure', 'stroke', 'severe bleeding', 'choking', 'suicide', 'poisoning', 'overdose'],
    'hi': ['छाती में दर्द', 'हार्ट अटैक', 'सांस नहीं', 'बेहोश', 'दौरा', 'खून बह रहा', 'जहर'],
    'ta': ['நெஞ்சு வலி', 'மாரடைப்பு', 'மூச்சு திணறல்', 'மயக்கம்', 'வலிப்பு'],
    'te': ['ఛాతీ నొప్పి', 'గుండెపోటు', 'ఊపిరి ఆడటం లేదు', 'స్పృహ లేదు'],
    'kn': ['ಎದೆ ನೋವು', 'ಹೃದಯಾಘಾತ', 'ಉಸಿರಾಟ ತೊಂದರೆ', 'ಪ್ರಜ್ಞೆ ತಪ್ಪು'],
}

STRESS_INDICATORS = {
    'en': ['help me', 'please help', 'very worried', 'scared', 'afraid', 'anxious', "can't sleep", 'desperate', 'terrible', 'worst pain', 'unbearable'],
    'hi': ['मदद करो', 'बहुत चिंता', 'डर', 'नींद नहीं', 'असहनीय'],
    'ta': ['உதவி', 'பயம்', 'கவலை', 'தூக்கமின்மை'],
    'te': ['సహాయం', 'భయం', 'ఆందోళన'],
    'kn': ['ಸಹಾಯ', 'ಭಯ', 'ಚಿಂತೆ', 'ನಿದ್ರೆ ಬರುತ್ತಿಲ್ಲ'],
}

BREATHING_KEYWORDS = ['chest', 'breathe', 'breathing', 'heart', 'छाती', 'सांस', 'ಎದೆ', 'ಉಸಿರು', 'நெஞ்சு', 'மூச்சு']


def _kw(lexicon: Dict[str, List[str]], language: str) -> List[str]:
    """Language-specific keywords plus the English fallback set."""
    return lexicon.get(language, []) + lexicon.get('en', [])


def extract_layer1_signals(text: str, language: str = "en") -> dict:
    """Extract intent, urgency, tone, and stress signals from a transcript."""
    start = time.time()
    text_lower = text.lower()

    # Intent (best-scoring keyword bucket)
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

    # Urgency
    urgency_count = sum(1 for kw in _kw(URGENCY_HIGH, language) if kw.lower() in text_lower)
    if urgency_count >= 2:
        urgency = "high"
    elif urgency_count == 1:
        urgency = "medium"
    else:
        urgency = "normal"

    # Stress
    stress_level = "stressed" if any(
        kw.lower() in text_lower for kw in _kw(STRESS_INDICATORS, language)
    ) else "calm"

    # Tone
    tone = "stressed" if (urgency == "high" or stress_level == "stressed" or text.count('!') > 0) else "calm"

    return {
        "intent": intent,
        "intent_confidence": round(intent_confidence, 3),
        "urgency": urgency,
        "urgency_count": urgency_count,
        "tone": tone,
        "stress_level": stress_level,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }


def run_safety_gate(
    text: str,
    language: str = "en",
    signals: Optional[dict] = None,
    repeat_count: int = 0,
) -> dict:
    """
    Decide whether the turn must be escalated, using 5 rules:
      1. Critical symptom present            → escalate
      2. Chest/breathing + medical intent    → escalate
      3. Low-confidence medical query        → escalate (clarify)
      4. High urgency (>=2 urgency keywords)  → escalate
      5. Repeated query (repeat_count >= 2)   → escalate

    `signals` may be precomputed via extract_layer1_signals; if omitted it is
    computed here. `repeat_count` is supplied by the caller (e.g. from Redis
    context) — 0 disables rule 5.
    """
    start = time.time()
    if signals is None:
        signals = extract_layer1_signals(text, language)

    text_lower = text.lower()
    should_escalate = False
    escalation_reason = None
    rules_triggered: List[str] = []

    # Rule 1: critical symptom
    for symptom in _kw(CRITICAL_SYMPTOMS, language):
        if symptom.lower() in text_lower:
            should_escalate = True
            escalation_reason = f"Critical symptom detected: {symptom}"
            rules_triggered.append("rule_1_critical_symptom")
            break

    # Rule 2: breathing/chest concern with a medical query
    if any(kw in text_lower for kw in BREATHING_KEYWORDS) and signals.get("intent") == "medical_query":
        should_escalate = True
        escalation_reason = escalation_reason or "Chest/breathing concern with medical query"
        rules_triggered.append("rule_2_breathing_medical")

    # Rule 3: low-confidence medical query
    if signals.get("intent") == "medical_query" and signals.get("intent_confidence", 1.0) < 0.3:
        should_escalate = True
        escalation_reason = escalation_reason or "Low confidence medical query - needs clarification"
        rules_triggered.append("rule_3_low_confidence")

    # Rule 4: multiple urgency indicators
    if signals.get("urgency_count", 0) >= 2:
        should_escalate = True
        escalation_reason = escalation_reason or "Multiple urgency indicators detected"
        rules_triggered.append("rule_4_high_urgency")

    # Rule 5: repeated query
    if repeat_count >= 2:
        should_escalate = True
        escalation_reason = escalation_reason or f"Repeated query ({repeat_count}x) - may need direct assistance"
        rules_triggered.append("rule_5_repeated_query")

    return {
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "rules_triggered": rules_triggered,
        "signals": signals,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }


def build_safety_directive(safety: dict, language: str = "en") -> Optional[str]:
    """
    Produce a system-prompt injection that instructs the LLM how to respond
    when the safety gate escalates. Returns None when no escalation is needed.
    """
    if not safety.get("should_escalate"):
        return None
    reason = safety.get("escalation_reason", "safety escalation")
    return (
        "\n\n## SAFETY OVERRIDE (HIGHEST PRIORITY)\n"
        f"The safety gate flagged this turn: {reason}.\n"
        "- Acknowledge the concern immediately and calmly, in the user's language.\n"
        "- Advise contacting emergency services now: Ambulance 108 / 102, Police 100.\n"
        "- Keep the reply short, reassuring, and action-oriented.\n"
        "- Offer to connect the user to a human right away.\n"
        "User safety overrides all other goals."
    )
