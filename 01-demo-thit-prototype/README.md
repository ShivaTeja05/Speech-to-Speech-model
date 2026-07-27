# Demo THIT

Real-time Voice AI Call system for Indian languages (Tamil, Telugu, Kannada, Hindi).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Hugging Face token:
```
HUGGINGFACE_TOKEN=your_token_here
```

3. Run the application:
```bash
bash run.sh
```

## Features

- Real-time speech-to-text using Faster-Whisper
- LLM-powered responses using Llama
- Text-to-speech using Edge TTS
- Web interface for interaction
- Support for multiple Indian languages

## Configuration

Make sure to set your Hugging Face token in the `.env` file before running the application.
