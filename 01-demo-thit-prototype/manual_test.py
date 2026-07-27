
import os
import fasttext
import time

# Mocking the logic from app.py for standalone testing
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

MODEL_PATH = "models/lid.176.ftz"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please ensure it is downloaded.")
        return None
    return fasttext.load_model(MODEL_PATH)

def detect_language(text, model):
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
    
    try:
        clean_text = text.replace('\n', ' ').strip()
        labels, scores = model.predict(clean_text)
        if labels:
            ft_lang = labels[0].replace("__label__", "")
            ft_score = float(scores[0])
            if ft_lang in SUPPORTED_LANGUAGES:
                ft_detected = ft_lang
                ft_conf = ft_score
    except Exception as e:
        pass

    # 3. Decision Logic (Same as app.py)
    final_lang = script_detected
    final_conf = script_conf
    
    if ft_detected and ft_conf > 0.4:
        if script_detected == 'en' and ft_detected != 'en':
             final_lang = ft_detected
             final_conf = ft_conf
        elif script_conf < 0.6:
             final_lang = ft_detected
             final_conf = ft_conf
             
    latency = (time.time() - start) * 1000
    
    return {
        "final": final_lang,
        "name": SUPPORTED_LANGUAGES.get(final_lang, "Unknown"),
        "script_layer": {"lang": script_detected, "conf": round(script_conf, 2)},
        "fasttext_layer": {"lang": ft_detected, "conf": round(ft_conf, 2)},
        "latency_ms": round(latency, 2)
    }

def main():
    print("Loading FastText model...")
    model = load_model()
    if not model:
        return

    print("\n" + "="*50)
    print(" MANUAL LANGUAGE DETECTION TESTER")
    print(" Type a sentence/word and hit Enter.")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            text = input(">> Enter text: ").strip()
            if text.lower() in ['exit', 'quit']:
                break
            if not text:
                continue
                
            result = detect_language(text, model)
            
            print(f"\n   Detected: \033[92m{result['name']} ({result['final']})\033[0m")
            print(f"   Layers:   Script={result['script_layer']['lang']} | FastText={result['fasttext_layer']['lang']} ({result['fasttext_layer']['conf']})")
            print(f"   Latency:  {result['latency_ms']:.2f}ms\n")
            
        except KeyboardInterrupt:
            break
    print("\nBye!")

if __name__ == "__main__":
    main()
