"""
Piper TTS Demo Server
=====================
A web server for Piper TTS demonstration with SNAC integration.

Run with: python scripts/piper_tts_server.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import wave
import base64
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from typing import Optional

# Try to import piper
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: piper-tts not installed. Run: pip install 'piper-tts[http]'")
    PIPER_AVAILABLE = False

PORT = 8085
VOICES = {}

# Default voice directory
VOICE_DIR = Path.home() / ".local" / "share" / "piper_voices"

# Available voice configurations
VOICE_CONFIG = {
    "en_US-lessac-medium": {
        "name": "English (US) - Lessac",
        "language": "en",
        "description": "Clear American English voice"
    },
    "hi_IN-rohan-medium": {
        "name": "Hindi - Rohan",
        "language": "hi",
        "description": "Hindi male voice"
    }
}

def find_voice_file(voice_name: str) -> Optional[Path]:
    """Find the ONNX model file for a voice."""
    # Project directory where voices were downloaded
    project_dir = Path(__file__).parent.parent
    
    possible_paths = [
        project_dir / f"{voice_name}.onnx",  # Project root (where we downloaded)
        VOICE_DIR / f"{voice_name}.onnx",
        Path.home() / ".local" / "share" / "piper_voices" / f"{voice_name}.onnx",
        Path("/usr/share/piper-voices") / f"{voice_name}.onnx",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def load_voices():
    """Load available Piper voices."""
    global VOICES
    
    if not PIPER_AVAILABLE:
        return
    
    for voice_name in VOICE_CONFIG:
        voice_path = find_voice_file(voice_name)
        if voice_path:
            try:
                VOICES[voice_name] = {
                    "voice": PiperVoice.load(str(voice_path)),
                    "config": VOICE_CONFIG[voice_name],
                    "path": str(voice_path)
                }
                print(f"✓ Loaded voice: {voice_name}")
            except Exception as e:
                print(f"✗ Failed to load {voice_name}: {e}")
        else:
            print(f"! Voice not found: {voice_name}")

class PiperHandler(SimpleHTTPRequestHandler):
    """HTTP handler for Piper TTS API."""
    
    def do_GET(self):
        if self.path == "/":
            # Serve the demo page
            demo_path = Path(__file__).parent.parent / "demo" / "piper_demo.html"
            if demo_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(demo_path.read_bytes())
            else:
                self.send_error(404, "Demo page not found")
        
        elif self.path == "/voices":
            # Return available voices
            voices_list = []
            for voice_name, voice_data in VOICES.items():
                voices_list.append({
                    "id": voice_name,
                    **voice_data["config"]
                })
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"voices": voices_list}).encode())
        
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "piper_available": PIPER_AVAILABLE,
                "voices_loaded": len(VOICES)
            }).encode())
        
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == "/synthesize":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                text = data.get("text", "")
                voice_id = data.get("voice", "en_US-lessac-medium")
                length_scale = float(data.get("length_scale", 1.0))
                
                if not text:
                    self.send_error(400, "No text provided")
                    return
                
                if voice_id not in VOICES:
                    self.send_error(404, f"Voice '{voice_id}' not available")
                    return
                
                voice = VOICES[voice_id]["voice"]
                
                # Synthesize to WAV in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file)
                
                wav_data = wav_buffer.getvalue()
                
                # Check if client wants base64 or raw audio
                if data.get("format") == "base64":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "audio": base64.b64encode(wav_data).decode(),
                        "format": "wav",
                        "text": text,
                        "voice": voice_id
                    }).encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-type", "audio/wav")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Disposition", "attachment; filename=speech.wav")
                    self.end_headers()
                    self.wfile.write(wav_data)
                
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                print(f"Synthesis error: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not found")
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

def main():
    print("=" * 50)
    print("🔊 Piper TTS Demo Server")
    print("=" * 50)
    
    # Load voices
    print("\nLoading voices...")
    load_voices()
    
    if not VOICES:
        print("\n⚠️  No voices loaded! Run these commands to download:")
        print("   python3 -m piper.download_voices en_US-lessac-medium")
        print("   python3 -m piper.download_voices hi_IN-rohan-medium")
        print("\nStarting server anyway for demo purposes...\n")
    
    # Start server
    server = HTTPServer(("", PORT), PiperHandler)
    url = f"http://localhost:{PORT}"
    
    print(f"\n✓ Server running at {url}")
    print("  Endpoints:")
    print("    GET  /         - Demo page")
    print("    GET  /voices   - List voices")
    print("    GET  /health   - Health check")
    print("    POST /synthesize - Text-to-speech")
    print("\nPress Ctrl+C to stop\n")
    
    # Open browser
    webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
