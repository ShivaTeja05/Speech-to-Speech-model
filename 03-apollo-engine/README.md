# Apollo Omni-Indic Voice Engine

> **⚠️ Status: research scaffold — architecture only, not trained.**
> This layer implements the *architecture* of a unified speech-to-speech
> transformer, but **no trained weights exist**, so it does not yet run as a
> working system (untrained, it produces noise). The features, latency, and
> cost figures below describe the **intended design/target**, not measured
> results. For a working, benchmarked pipeline see `../02-voice-model-ai/`.

A unified, de-novo speech-to-speech transformer for real-time conversational AI in Indian regional languages (Tamil, Telugu, Kannada, Hindi).

## Features

- 🎯 **Unified Architecture**: Single transformer combining STT + LLM + TTS
- ⚡ **Real-time**: <300ms latency for natural conversation
- 💰 **Cost-effective**: <₹2/min operational cost
- 🏥 **Healthcare-optimized**: Built for Apollo patient interactions

## Architecture

```
Audio In → SNAC Encoder → [Extended Sarvam-1 2B] → SNAC Decoder → Audio Out
```

## Supported Languages

| Language | Code | Status |
|----------|------|--------|
| Tamil | `ta` | 🔄 In Progress |
| Telugu | `te` | 🔄 In Progress |
| Kannada | `kn` | 🔄 In Progress |
| Hindi | `hi` | 🔄 In Progress |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py

# Run inference demo
python scripts/demo.py --lang tamil
```

## Project Structure

```
apollo_voice_engine/
├── config/          # Model configurations
├── data/            # Data processing pipelines
├── models/          # Core model implementations
├── training/        # Training scripts
├── inference/       # Optimized inference engine
└── integration/     # Kiosk & WebRTC deployment
```

## License

Proprietary - Apollo Hospitals
