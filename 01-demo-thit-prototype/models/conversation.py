"""
Conversation Memory System for Apollo Hospital Voice AI
Handles session management, context extraction, and conversation history
"""

import re
import json
import uuid
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .redis_store import get_redis, RedisStore

logger = logging.getLogger(__name__)

# ============================================================================
# CONTEXT EXTRACTION PATTERNS (Multi-language)
# ============================================================================

# Name extraction patterns
NAME_PATTERNS = {
    'en': [
        r"(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:name|call me)\s+([A-Z][a-z]+)",
    ],
    'hi': [
        r"(?:मेरा नाम|मैं)\s+(\S+)",
        r"(?:नाम)\s+(\S+)\s+है",
    ],
    'kn': [
        r"(?:ನನ್ನ ಹೆಸರು|ನಾನು)\s+(\S+)",
    ],
    'ta': [
        r"(?:என் பெயர்|நான்)\s+(\S+)",
    ],
    'te': [
        r"(?:నా పేరు|నేను)\s+(\S+)",
    ],
}

# Symptom keywords (multi-language)
SYMPTOM_KEYWORDS = {
    'en': {
        'fever': ['fever', 'temperature', 'hot'],
        'headache': ['headache', 'head pain', 'head ache', 'migraine'],
        'cough': ['cough', 'coughing'],
        'cold': ['cold', 'runny nose', 'sneezing'],
        'stomach_pain': ['stomach pain', 'stomach ache', 'abdominal pain', 'tummy'],
        'chest_pain': ['chest pain', 'chest', 'heart pain'],
        'breathing': ['breathless', 'breathing', 'breath', 'asthma'],
        'vomiting': ['vomiting', 'vomit', 'nausea', 'throwing up'],
        'diarrhea': ['diarrhea', 'loose motion', 'loose stool'],
        'body_pain': ['body pain', 'body ache', 'weakness', 'tired'],
    },
    'hi': {
        'fever': ['बुखार', 'तापमान', 'गर्मी'],
        'headache': ['सिरदर्द', 'सिर दर्द', 'माइग्रेन'],
        'cough': ['खांसी', 'खासी'],
        'cold': ['सर्दी', 'जुकाम', 'नाक बहना'],
        'stomach_pain': ['पेट दर्द', 'पेट में दर्द'],
        'chest_pain': ['छाती में दर्द', 'सीने में दर्द'],
        'breathing': ['सांस', 'दम'],
        'vomiting': ['उल्टी', 'मतली'],
    },
    'kn': {
        'fever': ['ಜ್ವರ', 'ಬಿಸಿ'],
        'headache': ['ತಲೆನೋವು', 'ತಲೆ ನೋವು'],
        'cough': ['ಕೆಮ್ಮು'],
        'cold': ['ಶೀತ', 'ನೆಗಡಿ'],
        'stomach_pain': ['ಹೊಟ್ಟೆ ನೋವು'],
        'chest_pain': ['ಎದೆ ನೋವು'],
        'breathing': ['ಉಸಿರು', 'ಉಸಿರಾಟ'],
    },
    'ta': {
        'fever': ['காய்ச்சல்'],
        'headache': ['தலைவலி'],
        'cough': ['இருமல்'],
        'cold': ['சளி', 'ஜலதோஷம்'],
        'stomach_pain': ['வயிற்று வலி'],
        'chest_pain': ['நெஞ்சு வலி'],
    },
    'te': {
        'fever': ['జ్వరం'],
        'headache': ['తలనొప్పి'],
        'cough': ['దగ్గు'],
        'cold': ['జలుబు'],
        'stomach_pain': ['కడుపు నొప్పి'],
        'chest_pain': ['ఛాతీ నొప్పి'],
    },
}

# Duration patterns
DURATION_PATTERNS = {
    'en': [
        r"(?:since|for|from)\s+(\d+)\s+(day|days|week|weeks|month|months|hour|hours)",
        r"(\d+)\s+(day|days|week|weeks|month|months|hour|hours)\s+(?:ago|back)",
        r"(?:last|past)\s+(\d+)\s+(day|days|week|weeks)",
    ],
    'hi': [
        r"(\d+)\s+(?:दिन|दिनों|हफ्ते|महीने)\s+से",
    ],
    'kn': [
        r"(\d+)\s+(?:ದಿನ|ದಿನಗಳಿಂದ|ವಾರ)",
    ],
}

# Doctor/Appointment patterns
DOCTOR_PATTERNS = {
    'en': [
        r"(?:dr\.?|doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:see|meet|consult|appointment with)\s+(?:dr\.?|doctor)?\s*([A-Z][a-z]+)",
    ],
    'hi': [
        r"(?:डॉक्टर|डॉ\.?)\s+(\S+)",
    ],
    'kn': [
        r"(?:ಡಾಕ್ಟರ್|ವೈದ್ಯರು)\s+(\S+)",
    ],
}

# Age patterns
AGE_PATTERNS = {
    'en': [
        r"(?:i am|i'm|age is|aged?)\s+(\d+)(?:\s+years?)?(?:\s+old)?",
        r"(\d+)\s+years?\s+old",
    ],
    'hi': [
        r"(?:उम्र|आयु)\s+(\d+)",
        r"(\d+)\s+साल",
    ],
    'kn': [
        r"(?:ವಯಸ್ಸು)\s+(\d+)",
        r"(\d+)\s+ವರ್�",
    ],
}


class ContextExtractor:
    """Extracts key-value context from user utterances"""
    
    def __init__(self):
        pass
    
    def extract(self, text: str, language: str = "en", existing_context: Dict = None) -> Dict[str, Any]:
        """
        Extract contextual information from text.
        
        Args:
            text: User utterance
            language: Language code (en, hi, kn, ta, te)
            existing_context: Previously extracted context to update
            
        Returns:
            Updated context dictionary
        """
        context = existing_context.copy() if existing_context else {}
        text_lower = text.lower()
        
        # Extract patient name
        name = self._extract_name(text, language)
        if name:
            context['patient_name'] = name
        
        # Extract symptoms
        symptoms = self._extract_symptoms(text_lower, language)
        if symptoms:
            existing_symptoms = context.get('symptoms', [])
            context['symptoms'] = list(set(existing_symptoms + symptoms))
        
        # Extract duration
        duration = self._extract_duration(text, language)
        if duration:
            context['duration'] = duration
        
        # Extract age
        age = self._extract_age(text, language)
        if age:
            context['patient_age'] = age
        
        # Extract requested doctor
        doctor = self._extract_doctor(text, language)
        if doctor:
            context['requested_doctor'] = doctor
        
        # Track language preference
        if language and language != 'en':
            context['preferred_language'] = language
        
        return context
    
    def _extract_name(self, text: str, language: str) -> Optional[str]:
        """Extract patient name from text"""
        patterns = NAME_PATTERNS.get(language, []) + NAME_PATTERNS.get('en', [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_symptoms(self, text_lower: str, language: str) -> List[str]:
        """Extract symptoms mentioned in text"""
        symptoms = []
        
        # Check language-specific keywords
        lang_symptoms = SYMPTOM_KEYWORDS.get(language, {})
        en_symptoms = SYMPTOM_KEYWORDS.get('en', {})
        
        for symptom_type, keywords in {**en_symptoms, **lang_symptoms}.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if symptom_type not in symptoms:
                        symptoms.append(symptom_type)
                    break
        
        return symptoms
    
    def _extract_duration(self, text: str, language: str) -> Optional[str]:
        """Extract duration of symptoms"""
        patterns = DURATION_PATTERNS.get(language, []) + DURATION_PATTERNS.get('en', [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return f"{groups[0]} {groups[1]}"
                elif len(groups) == 1:
                    return groups[0]
        return None
    
    def _extract_age(self, text: str, language: str) -> Optional[int]:
        """Extract patient age"""
        patterns = AGE_PATTERNS.get(language, []) + AGE_PATTERNS.get('en', [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        return None
    
    def _extract_doctor(self, text: str, language: str) -> Optional[str]:
        """Extract requested doctor name"""
        patterns = DOCTOR_PATTERNS.get(language, []) + DOCTOR_PATTERNS.get('en', [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


class ConversationManager:
    """Manages conversation sessions and history"""
    
    MAX_HISTORY_TURNS = 10
    
    def __init__(self, redis_store: RedisStore = None):
        self.redis = redis_store or get_redis()
        self.context_extractor = ContextExtractor()
        self._memory_sessions: Dict[str, Dict] = {}  # In-memory fallback
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())[:8]
        
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'patient_name': '',
            'symptoms': [],
            'context': {},
            'history': [],
            'language': 'en',
            'turn_count': 0
        }
        
        if self.redis.is_connected:
            config = self.redis.get_config()
            ttl = config.get('session_timeout_minutes', 30)
            self.redis.set_session(session_id, session_data, ttl)
        else:
            self._memory_sessions[session_id] = session_data
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if self.redis.is_connected:
            return self.redis.get_session(session_id)
        return self._memory_sessions.get(session_id)
    
    def add_turn(
        self, 
        session_id: str, 
        user_text: str, 
        assistant_text: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Add a conversation turn and extract context.
        
        Args:
            session_id: Session identifier
            user_text: User's utterance
            assistant_text: AI's response
            language: Detected language
            
        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        
        if not session:
            # Create new session if doesn't exist
            session_id = self.create_session()
            session = self.get_session(session_id)
        
        # Extract context from user text
        existing_context = session.get('context', {})
        if isinstance(existing_context, str):
            existing_context = json.loads(existing_context)
        
        new_context = self.context_extractor.extract(
            user_text, 
            language, 
            existing_context
        )
        
        # Update history
        history = session.get('history', [])
        if isinstance(history, str):
            history = json.loads(history)
        
        timestamp = datetime.now().isoformat()
        history.append({
            'role': 'user',
            'text': user_text,
            'timestamp': timestamp
        })
        history.append({
            'role': 'assistant', 
            'text': assistant_text,
            'timestamp': timestamp
        })
        
        # Trim history to max turns
        if len(history) > self.MAX_HISTORY_TURNS * 2:
            history = history[-(self.MAX_HISTORY_TURNS * 2):]
        
        # Update session
        updates = {
            'context': new_context,
            'history': history,
            'language': language,
            'turn_count': session.get('turn_count', 0) + 1
        }
        
        # Update specific fields from context
        if new_context.get('patient_name'):
            updates['patient_name'] = new_context['patient_name']
        if new_context.get('symptoms'):
            updates['symptoms'] = new_context['symptoms']
        
        if self.redis.is_connected:
            self.redis.update_session(session_id, updates)
        else:
            session.update(updates)
            self._memory_sessions[session_id] = session
        
        # Return updated session
        return self.get_session(session_id)
    
    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get formatted conversation context for LLM prompt.
        
        Returns dict with:
            - patient_info: Extracted patient information
            - history_summary: Recent conversation history
            - context_string: Formatted context for prompt
        """
        session = self.get_session(session_id)
        
        if not session:
            return {
                'patient_info': {},
                'history_summary': [],
                'context_string': "New conversation. No prior context."
            }
        
        # Get context
        context = session.get('context', {})
        if isinstance(context, str):
            context = json.loads(context)
        
        # Get history
        history = session.get('history', [])
        if isinstance(history, str):
            history = json.loads(history)
        
        # Build patient info string
        patient_info = []
        if context.get('patient_name'):
            patient_info.append(f"Name: {context['patient_name']}")
        if context.get('patient_age'):
            patient_info.append(f"Age: {context['patient_age']}")
        
        symptoms = context.get('symptoms') or session.get('symptoms', [])
        if isinstance(symptoms, str):
            symptoms = json.loads(symptoms)
        if symptoms:
            patient_info.append(f"Symptoms: {', '.join(symptoms)}")
        
        if context.get('duration'):
            patient_info.append(f"Duration: {context['duration']}")
        if context.get('requested_doctor'):
            patient_info.append(f"Requested Doctor: {context['requested_doctor']}")
        
        # Build history summary (last 3 turns)
        recent_history = history[-6:] if len(history) > 6 else history
        history_summary = []
        for turn in recent_history:
            role = "Patient" if turn['role'] == 'user' else "Assistant"
            # Truncate long messages
            text = turn['text'][:100] + "..." if len(turn['text']) > 100 else turn['text']
            history_summary.append(f"{role}: {text}")
        
        # Build context string for LLM
        context_parts = []
        
        if patient_info:
            context_parts.append("PATIENT INFORMATION:\n" + "\n".join(patient_info))
        
        if history_summary:
            context_parts.append("RECENT CONVERSATION:\n" + "\n".join(history_summary))
        
        context_string = "\n\n".join(context_parts) if context_parts else "New conversation. No prior context."
        
        return {
            'patient_info': context,
            'history_summary': recent_history,
            'context_string': context_string,
            'turn_count': session.get('turn_count', 0),
            'language': session.get('language', 'en')
        }
    
    def get_repeated_query_count(self, session_id: str, current_text: str) -> int:
        """Check if current query is similar to recent queries (for escalation)"""
        session = self.get_session(session_id)
        if not session:
            return 0
        
        history = session.get('history', [])
        if isinstance(history, str):
            history = json.loads(history)
        
        # Get last 4 user messages
        user_messages = [h['text'].lower() for h in history if h['role'] == 'user'][-4:]
        current_lower = current_text.lower()
        
        # Count similar messages (simple similarity)
        count = 0
        for msg in user_messages:
            # Check for high overlap
            current_words = set(current_lower.split())
            msg_words = set(msg.split())
            if current_words and msg_words:
                overlap = len(current_words & msg_words) / max(len(current_words), len(msg_words))
                if overlap > 0.7:
                    count += 1
        
        return count
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        if self.redis.is_connected:
            return self.redis.delete_session(session_id)
        elif session_id in self._memory_sessions:
            del self._memory_sessions[session_id]
            return True
        return False
