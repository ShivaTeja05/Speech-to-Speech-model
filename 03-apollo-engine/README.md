# Apollo Omni-Indic Voice Engine

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
