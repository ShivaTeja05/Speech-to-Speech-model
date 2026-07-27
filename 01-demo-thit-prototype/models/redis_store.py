"""
Redis Store for Apollo Hospital Voice AI
Handles all Redis operations for config, sessions, doctors, FAQs, etc.
"""

import json
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import redis

logger = logging.getLogger(__name__)

# ============================================================================
# REDIS CONNECTION
# ============================================================================

class RedisStore:
    """Redis connection and operations manager"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.client: Optional[redis.Redis] = None
        self._connected = False
        
    def connect(self) -> bool:
        """Establish Redis connection"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self._connected = False
            return False
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self.client is not None
    
    # ========================================================================
    # HOSPITAL CONFIG OPERATIONS
    # ========================================================================
    
    def get_config(self) -> Dict[str, Any]:
        """Get hospital configuration"""
        if not self.is_connected:
            return self._get_default_config()
        
        try:
            config = self.client.hgetall("hospital:config")
            if not config:
                return self._get_default_config()
            
            # Parse JSON fields
            for key in ['supported_languages', 'escalation_keywords']:
                if key in config and isinstance(config[key], str):
                    config[key] = json.loads(config[key])
            
            # Parse boolean fields
            for key in ['disclaimer_required', 'always_recommend_doctor', 'tts_enabled']:
                if key in config:
                    config[key] = config[key].lower() == 'true'
            
            # Parse integer fields
            for key in ['max_words', 'session_timeout_minutes']:
                if key in config:
                    config[key] = int(config[key])
                    
            return config
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return self._get_default_config()
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Set hospital configuration"""
        if not self.is_connected:
            return False
        
        try:
            # Serialize complex types
            config_to_store = {}
            for key, value in config.items():
                if isinstance(value, (list, dict)):
                    config_to_store[key] = json.dumps(value)
                elif isinstance(value, bool):
                    config_to_store[key] = str(value).lower()
                else:
                    config_to_store[key] = str(value)
            
            self.client.hset("hospital:config", mapping=config_to_store)
            return True
        except Exception as e:
            logger.error(f"Error setting config: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default hospital configuration"""
        return {
            "hospital_name": "Apollo Hospital",
            "city": "Bengaluru",
            "emergency_number": "108",
            "helpline": "1860-500-1066",
            "tone": "formal",
            "max_words": 50,
            "primary_language": "en",
            "supported_languages": ["kn", "ta", "te", "hi", "en"],
            "disclaimer_required": True,
            "disclaimer_text": "This is general information only. Please consult a doctor for medical advice.",
            "always_recommend_doctor": True,
            "tts_enabled": True,
            "session_timeout_minutes": 30,
            "escalation_keywords": []
        }
    
    # ========================================================================
    # DOCTOR OPERATIONS
    # ========================================================================
    
    def get_doctors(self) -> List[Dict[str, Any]]:
        """Get all doctors"""
        if not self.is_connected:
            return []
        
        try:
            doctor_ids = self.client.smembers("hospital:doctors:ids")
            doctors = []
            for doc_id in doctor_ids:
                doctor = self.client.hgetall(f"hospital:doctors:{doc_id}")
                if doctor:
                    # Parse JSON fields
                    for key in ['available_days', 'languages']:
                        if key in doctor and isinstance(doctor[key], str):
                            doctor[key] = json.loads(doctor[key])
                    doctor['id'] = doc_id
                    doctors.append(doctor)
            return doctors
        except Exception as e:
            logger.error(f"Error getting doctors: {e}")
            return []
    
    def get_doctor(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get single doctor by ID"""
        if not self.is_connected:
            return None
        
        try:
            doctor = self.client.hgetall(f"hospital:doctors:{doc_id}")
            if doctor:
                for key in ['available_days', 'languages']:
                    if key in doctor and isinstance(doctor[key], str):
                        doctor[key] = json.loads(doctor[key])
                doctor['id'] = doc_id
            return doctor if doctor else None
        except Exception as e:
            logger.error(f"Error getting doctor {doc_id}: {e}")
            return None
    
    def add_doctor(self, doctor: Dict[str, Any]) -> Optional[str]:
        """Add a new doctor"""
        if not self.is_connected:
            return None
        
        try:
            doc_id = doctor.get('id') or f"d{self.client.incr('hospital:doctors:counter'):03d}"
            
            # Serialize complex types
            doctor_to_store = {}
            for key, value in doctor.items():
                if key == 'id':
                    continue
                if isinstance(value, (list, dict)):
                    doctor_to_store[key] = json.dumps(value)
                else:
                    doctor_to_store[key] = str(value)
            
            self.client.hset(f"hospital:doctors:{doc_id}", mapping=doctor_to_store)
            self.client.sadd("hospital:doctors:ids", doc_id)
            return doc_id
        except Exception as e:
            logger.error(f"Error adding doctor: {e}")
            return None
    
    def update_doctor(self, doc_id: str, doctor: Dict[str, Any]) -> bool:
        """Update existing doctor"""
        if not self.is_connected:
            return False
        
        try:
            if not self.client.exists(f"hospital:doctors:{doc_id}"):
                return False
            
            doctor_to_store = {}
            for key, value in doctor.items():
                if key == 'id':
                    continue
                if isinstance(value, (list, dict)):
                    doctor_to_store[key] = json.dumps(value)
                else:
                    doctor_to_store[key] = str(value)
            
            self.client.hset(f"hospital:doctors:{doc_id}", mapping=doctor_to_store)
            return True
        except Exception as e:
            logger.error(f"Error updating doctor {doc_id}: {e}")
            return False
    
    def delete_doctor(self, doc_id: str) -> bool:
        """Delete a doctor"""
        if not self.is_connected:
            return False
        
        try:
            self.client.delete(f"hospital:doctors:{doc_id}")
            self.client.srem("hospital:doctors:ids", doc_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting doctor {doc_id}: {e}")
            return False
    
    # ========================================================================
    # DEPARTMENT OPERATIONS
    # ========================================================================
    
    def get_departments(self) -> List[Dict[str, Any]]:
        """Get all departments"""
        if not self.is_connected:
            return []
        
        try:
            dept_ids = self.client.smembers("hospital:departments:ids")
            departments = []
            for dept_id in dept_ids:
                dept = self.client.hgetall(f"hospital:departments:{dept_id}")
                if dept:
                    dept['id'] = dept_id
                    departments.append(dept)
            return departments
        except Exception as e:
            logger.error(f"Error getting departments: {e}")
            return []
    
    def add_department(self, department: Dict[str, Any]) -> Optional[str]:
        """Add a new department"""
        if not self.is_connected:
            return None
        
        try:
            dept_id = department.get('id') or f"dept{self.client.incr('hospital:departments:counter'):03d}"
            
            dept_to_store = {k: str(v) for k, v in department.items() if k != 'id'}
            
            self.client.hset(f"hospital:departments:{dept_id}", mapping=dept_to_store)
            self.client.sadd("hospital:departments:ids", dept_id)
            return dept_id
        except Exception as e:
            logger.error(f"Error adding department: {e}")
            return None
    
    def delete_department(self, dept_id: str) -> bool:
        """Delete a department"""
        if not self.is_connected:
            return False
        
        try:
            self.client.delete(f"hospital:departments:{dept_id}")
            self.client.srem("hospital:departments:ids", dept_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting department {dept_id}: {e}")
            return False
    
    # ========================================================================
    # FAQ OPERATIONS
    # ========================================================================
    
    def get_faqs(self) -> List[Dict[str, Any]]:
        """Get all FAQs"""
        if not self.is_connected:
            return []
        
        try:
            faq_ids = self.client.smembers("hospital:faqs:ids")
            faqs = []
            for faq_id in faq_ids:
                faq = self.client.hgetall(f"hospital:faqs:{faq_id}")
                if faq:
                    faq['id'] = faq_id
                    faqs.append(faq)
            return faqs
        except Exception as e:
            logger.error(f"Error getting FAQs: {e}")
            return []
    
    def add_faq(self, faq: Dict[str, Any]) -> Optional[str]:
        """Add a new FAQ"""
        if not self.is_connected:
            return None
        
        try:
            faq_id = faq.get('id') or f"faq{self.client.incr('hospital:faqs:counter'):03d}"
            
            faq_to_store = {k: str(v) for k, v in faq.items() if k != 'id'}
            
            self.client.hset(f"hospital:faqs:{faq_id}", mapping=faq_to_store)
            self.client.sadd("hospital:faqs:ids", faq_id)
            return faq_id
        except Exception as e:
            logger.error(f"Error adding FAQ: {e}")
            return None
    
    def update_faq(self, faq_id: str, faq: Dict[str, Any]) -> bool:
        """Update existing FAQ"""
        if not self.is_connected:
            return False
        
        try:
            if not self.client.exists(f"hospital:faqs:{faq_id}"):
                return False
            
            faq_to_store = {k: str(v) for k, v in faq.items() if k != 'id'}
            self.client.hset(f"hospital:faqs:{faq_id}", mapping=faq_to_store)
            return True
        except Exception as e:
            logger.error(f"Error updating FAQ {faq_id}: {e}")
            return False
    
    def delete_faq(self, faq_id: str) -> bool:
        """Delete a FAQ"""
        if not self.is_connected:
            return False
        
        try:
            self.client.delete(f"hospital:faqs:{faq_id}")
            self.client.srem("hospital:faqs:ids", faq_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting FAQ {faq_id}: {e}")
            return False
    
    # ========================================================================
    # SESSION OPERATIONS
    # ========================================================================
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.is_connected:
            return None
        
        try:
            session = self.client.hgetall(f"session:{session_id}")
            if not session:
                return None
            
            # Parse JSON fields
            for key in ['symptoms', 'context', 'history']:
                if key in session and isinstance(session[key], str):
                    session[key] = json.loads(session[key])
            
            return session
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def set_session(self, session_id: str, data: Dict[str, Any], ttl_minutes: int = 30) -> bool:
        """Set session data with TTL"""
        if not self.is_connected:
            return False
        
        try:
            # Serialize complex types
            session_to_store = {}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    session_to_store[key] = json.dumps(value)
                else:
                    session_to_store[key] = str(value)
            
            session_to_store['updated_at'] = datetime.now().isoformat()
            
            self.client.hset(f"session:{session_id}", mapping=session_to_store)
            self.client.expire(f"session:{session_id}", ttl_minutes * 60)
            return True
        except Exception as e:
            logger.error(f"Error setting session {session_id}: {e}")
            return False
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update specific session fields"""
        if not self.is_connected:
            return False
        
        try:
            session_to_store = {}
            for key, value in updates.items():
                if isinstance(value, (list, dict)):
                    session_to_store[key] = json.dumps(value)
                else:
                    session_to_store[key] = str(value)
            
            session_to_store['updated_at'] = datetime.now().isoformat()
            
            self.client.hset(f"session:{session_id}", mapping=session_to_store)
            # Refresh TTL
            self.client.expire(f"session:{session_id}", 30 * 60)
            return True
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if not self.is_connected:
            return False
        
        try:
            self.client.delete(f"session:{session_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    # ========================================================================
    # SEARCH HELPERS
    # ========================================================================
    
    def search_doctors(self, query: str, language: str = "en") -> List[Dict[str, Any]]:
        """Search doctors by name, specialization, or department"""
        doctors = self.get_doctors()
        query_lower = query.lower()
        
        results = []
        for doctor in doctors:
            # Check various fields
            searchable = [
                doctor.get('name', ''),
                doctor.get(f'name_{language}', ''),
                doctor.get('specialization', ''),
                doctor.get('department', '')
            ]
            
            if any(query_lower in field.lower() for field in searchable if field):
                results.append(doctor)
        
        return results
    
    def get_available_doctors(self, day: str = None) -> List[Dict[str, Any]]:
        """Get doctors available on a specific day"""
        if day is None:
            day = datetime.now().strftime("%A")
        
        doctors = self.get_doctors()
        return [d for d in doctors if day in d.get('available_days', [])]


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_redis_store: Optional[RedisStore] = None

def get_redis() -> RedisStore:
    """Get or create Redis store singleton"""
    global _redis_store
    if _redis_store is None:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", 6379))
        _redis_store = RedisStore(host=host, port=port)
        _redis_store.connect()
    return _redis_store
