
import os
import sys
import time
import fasttext
import numpy as np
from faster_whisper import WhisperModel

# --- Configuration ---
MODEL_PATH = "models/lid.176.ftz"
SUPPORTED_LANGUAGES = {
    'kn': 'Kannada',
    'ta': 'Tamil', 
    'te': 'Telugu',
    'hi': 'Hindi',
    'en': 'English'
}
SCRIPT_RANGES = {
    'kn': [(0x0C80, 0x0CFF)],
    'ta': [(0x0B80, 0x0BFF)],
    'te': [(0x0C00, 0x0C7F)],
    'hi': [(0x0900, 0x097F)],
    'en': [(0x0041, 0x005A), (0x0061, 0x007A)],
}

def detect_script_language(text: str, ft_model, audio_lang_hint=None, audio_conf_hint=0.0) -> dict:
    """Hybrid detection: Script + FastText + Audio Consensus"""
    start = time.time()
    
    # 1. Script Detection
    char_counts = {lang: 0 for lang in SCRIPT_RANGES}
    for char in text:
        code_point = ord(char)
        for lang, ranges in SCRIPT_RANGES.items():
            for range_start, range_end in ranges:
                if range_start <= code_point <= range_end:
                    char_counts[lang] += 1
                    break
    
    total = sum(char_counts.values())
    if total == 0:
        script_detected = 'en'
        script_conf = 0.5
    else:
        script_detected = max(char_counts, key=char_counts.get)
        script_conf = char_counts[script_detected] / total

    # 2. FastText Detection
    ft_detected = None
    ft_conf = 0.0
    if ft_model:
        try:
            clean_text = text.replace('\n', ' ').strip()
            labels, scores = ft_model.predict(clean_text)
            if labels:
                ft_lang = labels[0].replace("__label__", "")
                ft_score = float(scores[0])
                if ft_lang in SUPPORTED_LANGUAGES:
                    ft_detected = ft_lang
                    ft_conf = ft_score
        except Exception:
            pass

    # 3. Decision Logic
    final_lang = script_detected
    final_conf = script_conf
    
    # FastText Refinement
    if ft_detected and ft_conf > 0.4:
        if script_detected == 'en' and ft_detected != 'en':
             final_lang = ft_detected
             final_conf = ft_conf
        elif script_conf < 0.6:
             final_lang = ft_detected
             final_conf = ft_conf

    # Audio Signal Consensus (The Fix)
    is_translation = False
    if audio_lang_hint in SUPPORTED_LANGUAGES:
        # If Audio is confident and Native, but Text is English
        if audio_conf_hint > 0.6 and audio_lang_hint != 'en':
            if final_lang == 'en':
                final_lang = audio_lang_hint
                final_conf = audio_conf_hint
                is_translation = True
             
    latency = (time.time() - start) * 1000
    
    return {
        "final": final_lang,
        "name": SUPPORTED_LANGUAGES.get(final_lang, "Unknown"),
        "script": {"lang": script_detected, "conf": round(script_conf, 2)},
        "fasttext": {"lang": ft_detected, "conf": round(ft_conf, 2)},
        "audio": {"lang": audio_lang_hint, "conf": round(audio_conf_hint, 2)},
        "is_translation": is_translation,
        "latency_ms": round(latency, 2)
    }

def main(audio_path):
    print(f"Processing: {audio_path}")
    if not os.path.exists(audio_path):
        print("Error: File not found")
        sys.exit(1)

    # Load FastText
    print("Loading FastText...")
    if not os.path.exists(MODEL_PATH):
        os.system(f"curl -o {MODEL_PATH} https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
    ft_model = fasttext.load_model(MODEL_PATH)

    # Load Whisper
    print("Loading Whisper...")
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel("medium", device=device, compute_type=compute_type)

    print("Transcribing (task='transcribe')...")
    start_transcribe = time.time()
    # Force task="transcribe"
    segments, info = model.transcribe(audio_path, beam_size=5, task="transcribe")
    
    text = " ".join([segment.text for segment in segments]).strip()
    transcribe_latency = (time.time() - start_transcribe) * 1000
    
    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    print(f"Transcribed Text: \"{text}\"")
    print(f"Whisper Audio Detect: \033[96m{info.language}\033[0m (Confidence: {info.language_probability:.2f})")
    
    if not text:
        print("No text transcribed.")
        return

    # Run Hybrid Detection with Audio Hint
    result = detect_script_language(text, ft_model, audio_lang_hint=info.language, audio_conf_hint=info.language_probability)
    
    print("\n--- Consensus Logic ---")
    print(f"Details:")
    print(f"  Script Layer:   {result['script']['lang']} (conf {result['script']['conf']})")
    print(f"  FastText Layer: {result['fasttext']['lang']} (conf {result['fasttext']['conf']})")
    print(f"  Audio Layer:    {result['audio']['lang']} (conf {result['audio']['conf']})")
    
    if result['is_translation']:
         print(f"  \033[93mOVERRIDE: Audio signal overruled Text (Translation Detected)\033[0m")
         
    print(f"\nFinal Detected: \033[92m{result['name']} ({result['final']})\033[0m")
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model_direct.py <audio_file>")
        sys.exit(1)
    main(sys.argv[1])
