"""
Model Download Script

Downloads required models from HuggingFace:
- Sarvam-1 2B (Indic-optimized LLM)
- SNAC 24kHz (Neural Audio Codec)
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_sarvam(cache_dir: str = None):
    """Download Sarvam-1 2B model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "sarvamai/sarvam-1"
        logger.info(f"Downloading {model_name}...")
        
        # Download tokenizer first (smaller)
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir=cache_dir
        )
        logger.info(f"Model downloaded: {model.num_parameters() / 1e9:.2f}B parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Sarvam-1: {e}")
        return False


def download_snac(cache_dir: str = None):
    """Download SNAC audio codec."""
    try:
        logger.info("Downloading SNAC 24kHz model...")
        
        from snac import SNAC
        
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        logger.info("SNAC model downloaded successfully")
        
        return True
        
    except ImportError:
        logger.error("SNAC package not installed. Run: pip install snac")
        return False
    except Exception as e:
        logger.error(f"Failed to download SNAC: {e}")
        return False


def validate_downloads():
    """Validate that models are properly downloaded."""
    logger.info("Validating downloads...")
    
    issues = []
    
    # Check transformers can find Sarvam-1
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "sarvamai/sarvam-1",
            trust_remote_code=True
        )
        logger.info(f"✓ Sarvam-1 config loaded: {config.hidden_size} hidden, {config.num_hidden_layers} layers")
    except Exception as e:
        issues.append(f"Sarvam-1 validation failed: {e}")
    
    # Check SNAC
    try:
        from snac import SNAC
        logger.info("✓ SNAC package available")
    except ImportError:
        issues.append("SNAC package not installed")
    
    if issues:
        logger.warning("Validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("✓ All models validated successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Apollo Voice Engine models")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--skip-sarvam",
        action="store_true",
        help="Skip downloading Sarvam-1 model"
    )
    parser.add_argument(
        "--skip-snac",
        action="store_true",
        help="Skip downloading SNAC model"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing downloads"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        success = validate_downloads()
        sys.exit(0 if success else 1)
    
    success = True
    
    if not args.skip_sarvam:
        logger.info("=" * 50)
        logger.info("STEP 1: Downloading Sarvam-1 2B")
        logger.info("=" * 50)
        if not download_sarvam(args.cache_dir):
            success = False
    
    if not args.skip_snac:
        logger.info("=" * 50)
        logger.info("STEP 2: Downloading SNAC")
        logger.info("=" * 50)
        if not download_snac(args.cache_dir):
            success = False
    
    logger.info("=" * 50)
    logger.info("STEP 3: Validating Downloads")
    logger.info("=" * 50)
    if not validate_downloads():
        success = False
    
    if success:
        logger.info("")
        logger.info("🎉 All models downloaded successfully!")
        logger.info("You can now run the demo: python scripts/demo.py")
    else:
        logger.error("")
        logger.error("⚠ Some downloads failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
