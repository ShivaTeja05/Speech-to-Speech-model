"""
Apollo Voice Engine - AudioLLM Training Script

Trains the AudioLLM (Extended Sarvam-1) on speech-to-speech tasks using:
- SNAC for audio tokenization
- IndicVoices dataset for Indian languages

Training Phases:
1. Phase 1: Audio Understanding - Learn to transcribe (audio tokens → text)
2. Phase 2: Audio Generation - Learn to synthesize (text → audio tokens)
3. Phase 3: End-to-End - Full speech-to-speech dialogue

Usage:
    python scripts/train_audio_llm.py --phase 1 --epochs 5
    python scripts/train_audio_llm.py --phase all --epochs 10
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apollo_voice_engine.models.audio_llm import AudioLLM
from apollo_voice_engine.models.snac_wrapper import SNACWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Dataset
    dataset_path: str = "indic_voices_dataset"
    metadata_file: str = "metadata.json"
    
    # Model
    model_name: str = "sarvamai/sarvam-1"
    snac_model: str = "hubertsiuzdak/snac_24khz"
    
    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    max_audio_length: int = 10  # seconds
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 500
    
    # Languages
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["hi", "ta", "te", "kn"]

# ============================================================================
# Dataset
# ============================================================================

class IndicVoicesDataset(Dataset):
    """
    Dataset for IndicVoices data.
    
    Loads audio files and transcriptions for training AudioLLM.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        snac: SNACWrapper,
        phase: str = "understanding",  # understanding, generation, end2end
        split: str = "train",
        split_ratio: float = 0.9
    ):
        self.config = config
        self.snac = snac
        self.phase = phase
        self.split = split
        
        # Load metadata
        metadata_path = Path(config.dataset_path) / config.metadata_file
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.all_samples = json.load(f)
        
        # Filter by languages
        self.samples = [
            s for s in self.all_samples 
            if s['language'] in config.languages
        ]
        
        # Split train/val
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split, phase={phase}")
        
        # Filter out very short/long samples
        self.samples = [
            s for s in self.samples 
            if 0.5 <= s['duration'] <= config.max_audio_length
        ]
        logger.info(f"After filtering: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio_path = Path(self.config.dataset_path) / sample['audio_path']
        audio = self._load_audio(audio_path, sample['sampling_rate'])
        
        # Encode audio with SNAC
        with torch.no_grad():
            audio_tokens = self.snac.encode(audio)
        
        return {
            'audio_tokens': audio_tokens.squeeze(0),
            'text': sample['text'],
            'language': sample['language'],
            'audio_path': str(audio_path)
        }
    
    def _load_audio(self, path: Path, original_sr: int) -> torch.Tensor:
        """Load and resample audio to 24kHz for SNAC."""
        import librosa
        
        try:
            # Load audio
            audio, sr = librosa.load(path, sr=original_sr)
            
            # Resample to 24kHz (SNAC requirement)
            if sr != 24000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            
            # Convert to tensor and move to SNAC device
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_tensor = audio_tensor.to(self.snac.device)
            
            return audio_tensor
        except Exception as e:
            logger.warning(f"Error loading audio {path}: {e}")
            # Return silent audio as fallback (1 second at 24kHz)
            return torch.zeros(1, 24000, device=self.snac.device)

def collate_fn(batch, tokenizer, audio_llm, phase="understanding"):
    """
    Collate function for DataLoader.
    
    Prepares training inputs based on the training phase.
    """
    # Get max audio token length in batch
    max_audio_len = max(item['audio_tokens'].shape[0] for item in batch)
    
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for item in batch:
        audio_tokens = item['audio_tokens']
        text = item['text']
        language = item['language']
        
        if phase == "understanding":
            # Audio → Text (Speech Recognition)
            # Input: <audio_start> audio_tokens <audio_end>
            # Target: text
            
            # Pad audio tokens
            pad_len = max_audio_len - audio_tokens.shape[0]
            if pad_len > 0:
                audio_tokens = torch.cat([
                    audio_tokens,
                    torch.zeros(pad_len, dtype=audio_tokens.dtype)
                ])
            
            # Create input sequence with audio tokens
            input_ids = audio_llm.prepare_audio_input(
                audio_tokens.unsqueeze(0),
                text_prompt=f"Transcribe the following audio in {language}:"
            )
            
            # Tokenize target text
            target_tokens = tokenizer(
                text,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=256
            )['input_ids'].squeeze(0)
            
            # Combine input and target for language modeling
            full_input = torch.cat([input_ids.squeeze(0), target_tokens])
            
            # Labels: -100 for input positions, actual tokens for target
            labels = torch.cat([
                torch.full((input_ids.shape[1],), -100),
                target_tokens
            ])
            
            input_ids_list.append(full_input)
            labels_list.append(labels)
            
        elif phase == "generation":
            # Text → Audio (Speech Synthesis)
            # Input: text + <audio_start>
            # Target: audio_tokens + <audio_end>
            
            # Tokenize text
            text_tokens = tokenizer(
                f"Synthesize the following text in {language}: {text}",
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=256
            )['input_ids'].squeeze(0)
            
            # Pad audio tokens
            pad_len = max_audio_len - audio_tokens.shape[0]
            if pad_len > 0:
                audio_tokens = torch.cat([
                    audio_tokens,
                    torch.zeros(pad_len, dtype=audio_tokens.dtype)
                ])
            
            # Create input with text + audio start token
            audio_start_token = torch.tensor([audio_llm.tokenizer.convert_tokens_to_ids(audio_llm.AUDIO_START_TOKEN)])
            audio_end_token = torch.tensor([audio_llm.tokenizer.convert_tokens_to_ids(audio_llm.AUDIO_END_TOKEN)])
            
            # Full sequence: text + <audio_start> + audio_tokens + <audio_end>
            full_input = torch.cat([
                text_tokens,
                audio_start_token,
                audio_tokens.long() + audio_llm.AUDIO_TOKEN_OFFSET,
                audio_end_token
            ])
            
            # Labels: -100 for text, actual tokens for audio
            labels = torch.cat([
                torch.full((text_tokens.shape[0],), -100),
                audio_start_token,
                audio_tokens.long() + audio_llm.AUDIO_TOKEN_OFFSET,
                audio_end_token
            ])
            
            input_ids_list.append(full_input)
            labels_list.append(labels)
            
        else:  # end2end
            # Audio → Audio (Full speech-to-speech)
            # For this we need paired audio, using self-reconstruction for now
            pass
    
    # Pad to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.shape[0]
        
        padded_input_ids.append(
            torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        )
        padded_labels.append(
            torch.cat([labs, torch.full((pad_len,), -100)])
        )
        padded_attention_mask.append(
            torch.cat([torch.ones(ids.shape[0]), torch.zeros(pad_len)])
        )
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_mask': torch.stack(padded_attention_mask)
    }

# ============================================================================
# Trainer
# ============================================================================

class AudioLLMTrainer:
    """Trainer for the AudioLLM model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        logger.info("Initializing SNAC...")
        self.snac = SNACWrapper(
            model_name=config.snac_model,
            device=config.device
        )
        self.snac.load_model()
        
        logger.info("Initializing AudioLLM...")
        self.audio_llm = AudioLLM(
            model_name=config.model_name,
            device=config.device,
            load_in_8bit=False,  # Full precision for training
            load_in_4bit=False
        )
        self.audio_llm.load_model()
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_phase(
        self,
        phase: str,
        num_epochs: int,
        resume_from: Optional[str] = None
    ):
        """Train a specific phase."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Phase: {phase.upper()}")
        logger.info(f"{'='*60}\n")
        
        # Create datasets
        train_dataset = IndicVoicesDataset(
            self.config, self.snac, phase=phase, split="train"
        )
        val_dataset = IndicVoicesDataset(
            self.config, self.snac, phase=phase, split="val"
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=lambda batch: collate_fn(
                batch, 
                self.audio_llm.tokenizer, 
                self.audio_llm,
                phase
            )
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(
                batch,
                self.audio_llm.tokenizer,
                self.audio_llm,
                phase
            )
        )
        
        # Optimizer
        optimizer = AdamW(
            self.audio_llm.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, scaler, phase
            )
            
            # Validate
            val_loss = self._validate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(f"best_{phase}.pt")
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch+1}_{phase}.pt")
    
    def _train_epoch(self, loader, optimizer, scheduler, scaler, phase):
        """Train for one epoch."""
        self.audio_llm.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Training {phase}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device).long()
            labels = batch['labels'].to(self.device).long()
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.audio_llm.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = self.audio_llm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1
                
                # Save checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'step': self.global_step
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self, loader):
        """Validate the model."""
        self.audio_llm.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device).long()
            labels = batch['labels'].to(self.device).long()
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.audio_llm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, filename: str):
        """Save a checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.audio_llm.model.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.audio_llm.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AudioLLM on IndicVoices")
    parser.add_argument("--phase", type=str, default="understanding",
                       choices=["understanding", "generation", "end2end", "all"],
                       help="Training phase")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs per phase")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--dataset", type=str, default="indic_voices_dataset",
                       help="Path to dataset")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs
    )
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║              Apollo Voice Engine - AudioLLM Training                   ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Dataset:    {args.dataset:<53}║
║  Phase:      {args.phase:<53}║
║  Epochs:     {args.epochs:<53}║
║  Batch Size: {args.batch_size:<53}║
║  Device:     {config.device:<53}║
║                                                                        ║
║  Training phases:                                                      ║
║    1. Understanding: Audio → Text (ASR)                               ║
║    2. Generation: Text → Audio (TTS)                                  ║
║    3. End-to-End: Audio → Audio (Full S2S)                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize trainer
    trainer = AudioLLMTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    if args.phase == "all":
        # Train all phases sequentially
        for phase in ["understanding", "generation", "end2end"]:
            trainer.train_phase(phase, args.epochs)
    else:
        trainer.train_phase(args.phase, args.epochs)
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}/")

if __name__ == "__main__":
    main()
