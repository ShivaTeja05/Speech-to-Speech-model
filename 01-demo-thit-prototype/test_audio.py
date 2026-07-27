
import requests
import sys
import os
import time
import json

# Try importing sounddevice for recording
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    CAN_RECORD = True
except ImportError:
    CAN_RECORD = False
    print("Warning: 'sounddevice' or 'soundfile' not found. Recording disabled. File upload only.")

SERVER_URL = "http://localhost:8000/process"

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds... (Speak now)")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")
        filename = "test_input.wav"
        sf.write(filename, recording, fs)
        return filename
    except Exception as e:
        print(f"Recording failed: {e}")
        return None

def send_audio(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Sending {filepath} to {SERVER_URL}...")
    try:
        with open(filepath, 'rb') as f:
            files = {'audio': (filepath, f, 'audio/wav')}
            start = time.time()
            response = requests.post(SERVER_URL, files=files)
            latency = (time.time() - start) * 1000
            
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*50)
            print(" SUCCESS")
            print("="*50)
            print(f"Transcription: {data.get('transcription')}")
            print(f"Language:      \033[92m{data.get('language_name')} ({data.get('language')})\033[0m")
            
            # Show details from pipeline
            pipeline = data.get('pipeline', {})
            lang_det = pipeline.get('language_detection', {})
            
            print("\nLanguage Detection Details:")
            print(f"  Confidence: {lang_det.get('confidence')}")
            print(f"  FastText:   {lang_det.get('fasttext_prediction')}")
            print(f"  Total Latency: {data.get('total_latency_ms')}ms")
            
            if data.get('audio_response'):
                print("\n(Received Audio Response blob)")
        else:
            print(f"\nError {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")
        print("Ensure the server is running: 'uvicorn app:app --port 8000'")

def main():
    if len(sys.argv) > 1:
        # File provided as argument
        send_audio(sys.argv[1])
        return

    print("1. Upload existing .wav file")
    if CAN_RECORD:
        print("2. Record from microphone")
    
    choice = input("Select option: ").strip()
    
    if choice == '1':
        path = input("Enter path to .wav file: ").strip().strip("'").strip('"')
        send_audio(path)
    elif choice == '2' and CAN_RECORD:
        duration = float(input("Duration (seconds) [5]: ") or 5)
        filename = record_audio(duration)
        if filename:
            send_audio(filename)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
