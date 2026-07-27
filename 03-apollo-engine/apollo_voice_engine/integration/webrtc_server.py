"""
Apollo Voice Engine - WebRTC Server for Kiosk Deployment

Real-time voice streaming interface for hospital kiosks.
"""

import asyncio
import json
import logging
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WebRTCConfig:
    """WebRTC server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    stun_servers: list = None
    max_connections: int = 25
    
    def __post_init__(self):
        if self.stun_servers is None:
            self.stun_servers = ["stun:stun.l.google.com:19302"]


class KioskSession:
    """Represents a single kiosk voice session."""
    
    def __init__(self, session_id: str, language: str = "hi"):
        self.session_id = session_id
        self.language = language
        self.is_active = True
        self.audio_buffer = []
        self.transcript_history = []
        
    def add_audio_chunk(self, chunk: bytes):
        """Add incoming audio chunk to buffer."""
        self.audio_buffer.append(chunk)
        
    def get_audio_buffer(self) -> bytes:
        """Get and clear the audio buffer."""
        data = b"".join(self.audio_buffer)
        self.audio_buffer.clear()
        return data
        
    def add_to_history(self, role: str, text: str):
        """Add exchange to conversation history."""
        self.transcript_history.append({
            "role": role,
            "text": text
        })
        
    def close(self):
        """Close the session."""
        self.is_active = False


class WebRTCServer:
    """
    WebRTC server for Apollo Voice kiosks.
    
    Handles real-time audio streaming between kiosk tablets
    and the voice inference engine.
    """
    
    def __init__(
        self, 
        config: WebRTCConfig,
        on_audio_received: Optional[Callable] = None,
        on_session_start: Optional[Callable] = None,
        on_session_end: Optional[Callable] = None
    ):
        self.config = config
        self.sessions = {}
        self.on_audio_received = on_audio_received
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        
    async def handle_offer(self, offer_sdp: str, session_id: str) -> str:
        """
        Handle WebRTC offer from client.
        
        Args:
            offer_sdp: SDP offer from client
            session_id: Unique session identifier
            
        Returns:
            SDP answer for client
        """
        # Create new session
        session = KioskSession(session_id)
        self.sessions[session_id] = session
        
        if self.on_session_start:
            await self.on_session_start(session)
            
        logger.info(f"New kiosk session: {session_id}")
        
        # In production, this would create actual WebRTC peer connection
        # and return proper SDP answer
        answer_sdp = self._create_mock_answer(offer_sdp)
        
        return answer_sdp
    
    def _create_mock_answer(self, offer: str) -> str:
        """Create mock SDP answer (placeholder for actual WebRTC)."""
        return json.dumps({
            "type": "answer",
            "sdp": "mock_answer_sdp",
            "ice_servers": self.config.stun_servers
        })
        
    async def handle_audio_stream(self, session_id: str, audio_chunk: bytes):
        """
        Process incoming audio chunk from kiosk.
        
        Args:
            session_id: Session identifier
            audio_chunk: Raw audio bytes (24kHz, 16-bit PCM)
        """
        if session_id not in self.sessions:
            logger.warning(f"Unknown session: {session_id}")
            return
            
        session = self.sessions[session_id]
        session.add_audio_chunk(audio_chunk)
        
        if self.on_audio_received:
            await self.on_audio_received(session, audio_chunk)
            
    async def send_audio_response(self, session_id: str, audio_data: bytes):
        """
        Send audio response back to kiosk.
        
        Args:
            session_id: Session identifier
            audio_data: Response audio bytes
        """
        if session_id not in self.sessions:
            return
            
        # In production, this would stream through WebRTC data channel
        logger.info(f"Sending {len(audio_data)} bytes to session {session_id}")
        
    async def transfer_to_human(self, session_id: str, reason: str):
        """
        Transfer session to human operator via SIP.
        
        Args:
            session_id: Session identifier
            reason: Reason for transfer (e.g., "emergency detected")
        """
        if session_id not in self.sessions:
            return
            
        session = self.sessions[session_id]
        
        logger.warning(f"HUMAN TRANSFER: {session_id} - {reason}")
        
        # Log the transfer event
        transfer_event = {
            "event": "human_transfer",
            "session_id": session_id,
            "reason": reason,
            "conversation_history": session.transcript_history
        }
        
        # In production: Initiate SIP call to operator
        # sip_client.dial(OPERATOR_NUMBER, transfer_event)
        
        return transfer_event
        
    async def close_session(self, session_id: str):
        """Close a kiosk session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.close()
            
            if self.on_session_end:
                await self.on_session_end(session)
                
            del self.sessions[session_id]
            logger.info(f"Session closed: {session_id}")
            
    def get_active_sessions(self) -> int:
        """Get count of active sessions."""
        return len([s for s in self.sessions.values() if s.is_active])


# FastAPI integration example
def create_fastapi_app():
    """Create FastAPI app with WebRTC endpoints."""
    try:
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
        
    app = FastAPI(title="Apollo Voice Kiosk API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    server = WebRTCServer(WebRTCConfig())
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "active_sessions": server.get_active_sessions()
        }
    
    @app.post("/offer")
    async def create_offer(offer: dict):
        session_id = offer.get("session_id", "")
        sdp = offer.get("sdp", "")
        answer = await server.handle_offer(sdp, session_id)
        return {"answer": answer}
    
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await websocket.accept()
        
        try:
            while True:
                data = await websocket.receive_bytes()
                await server.handle_audio_stream(session_id, data)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await server.close_session(session_id)
            
    return app


if __name__ == "__main__":
    # Test server
    config = WebRTCConfig(port=8080)
    server = WebRTCServer(config)
    
    print("Apollo Voice Kiosk Server")
    print(f"Config: {config}")
    print(f"Max concurrent sessions: {config.max_connections}")
