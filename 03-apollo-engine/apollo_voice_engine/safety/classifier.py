"""
Apollo Voice Engine - Safety Classifier

Detects emergency keywords and unsafe medical situations
for immediate escalation to human operators.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety classification levels."""
    SAFE = "safe"
    CAUTION = "caution"  # Needs monitoring
    EMERGENCY = "emergency"  # Immediate human transfer


@dataclass
class SafetyResult:
    """Result of safety classification."""
    level: SafetyLevel
    triggered_keywords: List[str]
    confidence: float
    should_transfer: bool
    message: str


class SafetyClassifier:
    """
    Multi-language emergency and safety keyword detector.
    
    Detects emergency situations across Hindi, Tamil, Telugu, and Kannada
    for immediate transfer to human operators.
    """
    
    # Emergency keywords requiring immediate human transfer
    EMERGENCY_KEYWORDS = {
        # English
        "en": [
            "emergency", "accident", "heart attack", "stroke", "bleeding",
            "unconscious", "not breathing", "chest pain", "suicide", "dying",
            "ambulance", "icu", "critical"
        ],
        # Hindi
        "hi": [
            "इमरजेंसी", "दुर्घटना", "हार्ट अटैक", "दिल का दौरा", "खून",
            "बेहोश", "सांस नहीं", "छाती में दर्द", "सीने में दर्द", "मर रहा",
            "एंबुलेंस", "आईसीयू", "गंभीर", "एक्सीडेंट"
        ],
        # Tamil
        "ta": [
            "அவசரம்", "விபத்து", "மாரடைப்பு", "இரத்தப்போக்கு",
            "மயக்கம்", "மூச்சு விடவில்லை", "நெஞ்சு வலி", "ஆம்புலன்ஸ்",
            "தீவிர சிகிச்சை", "ஐசியு"
        ],
        # Telugu
        "te": [
            "అత్యవసర", "ప్రమాదం", "గుండెపోటు", "రక్తస్రావం",
            "స్పృహ లేదు", "ఊపిరి ఆడటం లేదు", "ఛాతీ నొప్పి", "అంబులెన్స్",
            "ఐసీయూ", "క్రిటికల్"
        ],
        # Kannada
        "kn": [
            "ತುರ್ತು", "ಅಪಘಾತ", "ಹೃದಯಾಘಾತ", "ರಕ್ತಸ್ರಾವ",
            "ಪ್ರಜ್ಞಾಹೀನ", "ಉಸಿರಾಡುತ್ತಿಲ್ಲ", "ಎದೆ ನೋವು", "ಆಂಬುಲೆನ್ಸ್",
            "ಐಸಿಯು", "ಗಂಭೀರ"
        ]
    }
    
    # Caution keywords requiring monitoring
    CAUTION_KEYWORDS = {
        "en": ["pain", "fever", "vomiting", "dizzy", "faint", "weak", "blood pressure"],
        "hi": ["दर्द", "बुखार", "उल्टी", "चक्कर", "कमज़ोर", "ब्लड प्रेशर"],
        "ta": ["வலி", "காய்ச்சல்", "வாந்தி", "தலைச்சுற்று", "பலவீனம்"],
        "te": ["నొప్పి", "జ్వరం", "వాంతి", "తలతిరుగుట", "బలహీనత"],
        "kn": ["ನೋವು", "ಜ್ವರ", "ವಾಂತಿ", "ತಲೆಸುತ್ತು", "ದೌರ್ಬಲ್ಯ"]
    }
    
    # Unsafe medical advice patterns (should not answer)
    UNSAFE_PATTERNS = [
        r"prescribe|prescription|dosage|mg|tablet|medicine.*take",
        r"should.*take.*medicine",
        r"diagnosis|diagnose",
    ]
    
    def __init__(self):
        # Compile all keywords into a single lookup
        self._emergency_set = set()
        self._caution_set = set()
        
        for lang_keywords in self.EMERGENCY_KEYWORDS.values():
            self._emergency_set.update(k.lower() for k in lang_keywords)
            
        for lang_keywords in self.CAUTION_KEYWORDS.values():
            self._caution_set.update(k.lower() for k in lang_keywords)
            
        # Compile regex patterns
        self._unsafe_pattern = re.compile(
            "|".join(self.UNSAFE_PATTERNS), 
            re.IGNORECASE
        )
    
    def classify(self, text: str) -> SafetyResult:
        """
        Classify text for safety concerns.
        
        Args:
            text: Input text (patient query or model response)
            
        Returns:
            SafetyResult with classification and triggered keywords
        """
        text_lower = text.lower()
        
        # Check for emergency keywords
        emergency_matches = []
        for keyword in self._emergency_set:
            if keyword in text_lower:
                emergency_matches.append(keyword)
        
        if emergency_matches:
            return SafetyResult(
                level=SafetyLevel.EMERGENCY,
                triggered_keywords=emergency_matches,
                confidence=1.0,
                should_transfer=True,
                message="Emergency detected. Transferring to human operator immediately."
            )
        
        # Check for caution keywords
        caution_matches = []
        for keyword in self._caution_set:
            if keyword in text_lower:
                caution_matches.append(keyword)
        
        if caution_matches:
            return SafetyResult(
                level=SafetyLevel.CAUTION,
                triggered_keywords=caution_matches,
                confidence=0.7,
                should_transfer=False,
                message="Medical concern detected. Monitoring conversation."
            )
        
        # Safe
        return SafetyResult(
            level=SafetyLevel.SAFE,
            triggered_keywords=[],
            confidence=1.0,
            should_transfer=False,
            message="No safety concerns detected."
        )
    
    def check_response_safety(self, response: str) -> Tuple[bool, str]:
        """
        Check if model response contains unsafe medical advice.
        
        Args:
            response: Model's generated response
            
        Returns:
            (is_safe, reason) tuple
        """
        if self._unsafe_pattern.search(response):
            return False, "Response may contain medical advice. Needs review."
        return True, "Response is safe."
    
    def get_transfer_message(self, lang: str = "en") -> str:
        """Get human transfer message in specified language."""
        messages = {
            "en": "I'm connecting you with a healthcare professional right away. Please stay on the line.",
            "hi": "मैं आपको तुरंत एक स्वास्थ्य विशेषज्ञ से जोड़ रहा हूं। कृपया लाइन पर रहें।",
            "ta": "நான் உங்களை உடனடியாக ஒரு சுகாதார நிபுணருடன் இணைக்கிறேன். தயவுசெய்து காத்திருங்கள்.",
            "te": "నేను మిమ్మల్ని వెంటనే ఒక ఆరోగ్య నిపుణుడితో అనుసంధానం చేస్తున్నాను. దయచేసి వేచి ఉండండి.",
            "kn": "ನಾನು ನಿಮ್ಮನ್ನು ತಕ್ಷಣ ಆರೋಗ್ಯ ತಜ್ಞರೊಂದಿಗೆ ಸಂಪರ್ಕಿಸುತ್ತೇನೆ. ದಯವಿಟ್ಟು ಕಾಯಿರಿ."
        }
        return messages.get(lang, messages["en"])


# Convenience function
def check_safety(text: str) -> SafetyResult:
    """Quick safety check on text."""
    classifier = SafetyClassifier()
    return classifier.classify(text)


if __name__ == "__main__":
    # Test the classifier
    classifier = SafetyClassifier()
    
    test_cases = [
        ("I need an ambulance now!", "en"),
        ("मुझे छाती में दर्द हो रहा है", "hi"),
        ("நெஞ்சு வலி", "ta"),
        ("నా అపాయింట్‌మెంట్ ఏమిటి?", "te"),
        ("I have a slight headache", "en"),
    ]
    
    print("Safety Classifier Test")
    print("=" * 50)
    
    for text, lang in test_cases:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Level: {result.level.value}")
        print(f"Keywords: {result.triggered_keywords}")
        print(f"Transfer: {result.should_transfer}")
