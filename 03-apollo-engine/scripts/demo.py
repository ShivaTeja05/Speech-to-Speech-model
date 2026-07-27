"""
Apollo Voice Engine Demo

Interactive demonstration of the unified speech-to-speech system.
Supports recording audio input and generating spoken responses.
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the Apollo Voice Engine banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           🏥 Apollo Omni-Indic Voice Engine 🎤                ║
║     Unified Speech-to-Speech for Indian Languages             ║
╠═══════════════════════════════════════════════════════════════╣
║  Supported Languages: Hindi (hi), Tamil (ta),                 ║
║                       Telugu (te), Kannada (kn)               ║
║  Target Latency: <300ms | Cost: <₹2/min                       ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are installed."""
    issues = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        issues.append("Transformers not installed")
    
    try:
        import snac
        logger.info("✓ SNAC installed")
    except ImportError:
        issues.append("SNAC not installed (pip install snac)")
    
    try:
        import torchaudio
        logger.info(f"✓ Torchaudio {torchaudio.__version__}")
    except ImportError:
        issues.append("Torchaudio not installed")
    
    if issues:
        logger.warning("Missing dependencies:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("\nRun: pip install -r requirements.txt")
        return False
    
    return True


def demo_text_to_speech(model, snac, text: str, language: str):
    """Demo: Generate speech from text."""
    logger.info(f"Generating speech for: '{text}' ({language})")
    
    # Prepare input (text only, model generates audio response)
    input_ids = model.text_to_tokens(text)
    
    # Add prompt for audio generation
    audio_prompt = f"<|audio_start|>"
    prompt_ids = model.text_to_tokens(audio_prompt)
    input_ids = torch.cat([input_ids, prompt_ids[:, 1:]], dim=1)  # Skip BOS
    
    # Generate
    output_ids = model.generate_response(
        input_ids,
        max_new_tokens=300,
        temperature=0.8
    )
    
    # Extract audio tokens
    audio_tokens = model.extract_audio_tokens(output_ids)
    
    if audio_tokens is not None:
        # Decode to audio
        audio = snac.decode(audio_tokens)
        logger.info(f"Generated {audio.shape[-1] / snac.sample_rate:.2f}s of audio")
        return audio
    else:
        logger.warning("No audio generated")
        return None


def demo_speech_to_speech(model, snac, audio_path: str):
    """Demo: End-to-end speech-to-speech."""
    import torch
    import torchaudio
    
    logger.info(f"Processing audio: {audio_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != snac.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, snac.sample_rate)
        waveform = resampler(waveform)
    
    # Encode input to tokens
    input_tokens = snac.encode(waveform)
    logger.info(f"Input encoded to {input_tokens.shape[1]} tokens")
    
    # Prepare model input
    input_ids = model.prepare_audio_input(input_tokens)
    
    # Generate response
    output_ids = model.generate_response(
        input_ids,
        max_new_tokens=500,
        temperature=0.7
    )
    
    # Extract response audio
    response_tokens = model.extract_audio_tokens(output_ids)
    
    if response_tokens is not None:
        response_audio = snac.decode(response_tokens)
        logger.info(f"Response: {response_audio.shape[-1] / snac.sample_rate:.2f}s")
        return response_audio
    
    return None


def run_interactive_demo():
    """Run interactive voice demo (requires microphone)."""
    print("\n🎤 Interactive Demo Mode")
    print("=" * 50)
    print("This mode requires a microphone for input.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        sample_rate = 24000
        duration = 5  # seconds
        
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Recording complete!")
        
        # Process would go here
        print("(Processing would be done here with loaded models)")
        
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"Error: {e}")


def run_demo(args):
    """Run the demo based on arguments."""
    print_banner()
    
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies first.")
        return 1
    
    if args.interactive:
        run_interactive_demo()
        return 0
    
    if args.check_only:
        print("\n✓ All dependencies available!")
        print("Run with --interactive for voice demo")
        print("Run without flags to test model loading")
        return 0
    
    # Full demo requires model loading
    try:
        import torch
        from apollo_voice_engine.models import AudioLLM, SNACWrapper
        
        print("\n📥 Loading models...")
        print("(This requires downloading ~4GB of model weights)")
        print("Run 'python scripts/download_models.py' first if not downloaded.\n")
        
        # Initialize SNAC
        snac = SNACWrapper()
        snac.load_model()
        print("✓ SNAC loaded")
        
        # Initialize Audio-LLM
        model = AudioLLM()
        model.load_model()
        print("✓ Audio-LLM loaded")
        
        # Run text-to-speech demo
        print("\n" + "=" * 50)
        print("Demo: Text to Speech")
        print("=" * 50)
        
        demo_texts = {
            "hi": "नमस्ते, मैं अपोलो वॉयस असिस्टेंट हूँ।",
            "ta": "வணக்கம், நான் அப்பல்லோ குரல் உதவியாளர்.",
            "te": "నమస్కారం, నేను అపోలో వాయిస్ అసిస్టెంట్.",
            "kn": "ನಮಸ್ಕಾರ, ನಾನು ಅಪೊಲೊ ಧ್ವನಿ ಸಹಾಯಕ."
        }
        
        text = demo_texts.get(args.lang, demo_texts["hi"])
        audio = demo_text_to_speech(model, snac, text, args.lang)
        
        if audio is not None and args.output:
            import torchaudio
            torchaudio.save(args.output, audio.unsqueeze(0), snac.sample_rate)
            print(f"Audio saved to: {args.output}")
        
        print("\n✓ Demo complete!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Apollo Voice Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --check-only        # Check dependencies only
  python demo.py --interactive       # Run interactive voice demo
  python demo.py --lang tamil        # Run demo in Tamil
  python demo.py --input audio.wav   # Process audio file
        """
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        choices=["hi", "ta", "te", "kn", "hindi", "tamil", "telugu", "kannada"],
        help="Language for demo (default: hindi)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input audio file to process"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive microphone demo"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't load models"
    )
    
    args = parser.parse_args()
    
    # Normalize language codes
    lang_map = {
        "hindi": "hi",
        "tamil": "ta", 
        "telugu": "te",
        "kannada": "kn"
    }
    args.lang = lang_map.get(args.lang, args.lang)
    
    sys.exit(run_demo(args))


if __name__ == "__main__":
    main()
