"""
Apollo Omni-Indic Voice Engine

A unified speech-to-speech transformer for real-time conversational AI
in Indian regional languages.
"""

__version__ = "0.1.0"
__author__ = "Apollo AI Team"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "apollo_voice_engine" / "config" / "model_config.yaml"
