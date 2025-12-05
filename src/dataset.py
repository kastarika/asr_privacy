"""
LibriSpeech dataset loader and preprocessing for adversarial training.
"""
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset for adversarial audio training."""

    def __init__(self, split='dev-clean', target_length=None, target_sr=24000, cache_dir=None):
        """
        Initialize LibriSpeech dataset.

        Args:
            split: Dataset split ('dev-clean', 'test-clean', 'train-clean-100', etc.)
            target_length: Target audio length in samples (None for variable length)
            target_sr: Target sample rate
            cache_dir: Directory to cache downloaded data
        """
        self.split = split
        self.target_length = target_length
        self.target_sr = target_sr

        # Load dataset from HuggingFace
        print(f"Loading LibriSpeech {split}...")
        self.dataset = load_dataset(
            "librispeech_asr",
            split=split,
            cache_dir=cache_dir
        )
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single audio sample.

        Returns:
            dict with keys:
                - audio: Audio tensor [channels, time]
                - text: Transcription text
                - sample_rate: Original sample rate
                - speaker_id: Speaker ID
                - chapter_id: Chapter ID
        """
        sample = self.dataset[idx]

        # Extract audio
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']

        # Convert to tensor
        audio = torch.from_numpy(audio_array).float()

        # Add channel dimension if mono
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            audio = resampler(audio)

        # Pad or crop to target length if specified
        if self.target_length is not None:
            current_length = audio.shape[1]
            if current_length < self.target_length:
                # Pad with zeros
                padding = self.target_length - current_length
                audio = torch.nn.functional.pad(audio, (0, padding))
            elif current_length > self.target_length:
                # Crop
                audio = audio[:, :self.target_length]

        return {
            'audio': audio,
            'text': sample['text'],
            'sample_rate': self.target_sr,
            'speaker_id': sample['speaker_id'],
            'chapter_id': sample['chapter_id'],
            'id': sample['id']
        }


def collate_fn_variable_length(batch):
    """
    Collate function for variable-length audio samples.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict with padded audio
    """
    # Find max length in batch
    max_length = max(item['audio'].shape[1] for item in batch)

    # Pad all audio to max length
    padded_audio = []
    lengths = []

    for item in batch:
        audio = item['audio']
        length = audio.shape[1]
        lengths.append(length)

        if length < max_length:
            padding = max_length - length
            audio = torch.nn.functional.pad(audio, (0, padding))

        padded_audio.append(audio)

    # Stack into batch
    audio_batch = torch.stack(padded_audio)
    lengths = torch.tensor(lengths)

    return {
        'audio': audio_batch,
        'lengths': lengths,
        'text': [item['text'] for item in batch],
        'sample_rate': batch[0]['sample_rate'],
        'speaker_ids': torch.tensor([item['speaker_id'] for item in batch]),
        'chapter_ids': torch.tensor([item['chapter_id'] for item in batch]),
    }


def collate_fn_fixed_length(batch):
    """
    Collate function for fixed-length audio samples.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dict
    """
    audio_batch = torch.stack([item['audio'] for item in batch])

    return {
        'audio': audio_batch,
        'text': [item['text'] for item in batch],
        'sample_rate': batch[0]['sample_rate'],
        'speaker_ids': torch.tensor([item['speaker_id'] for item in batch]),
        'chapter_ids': torch.tensor([item['chapter_id'] for item in batch]),
    }


def get_dataloader(split='dev-clean', batch_size=8, num_workers=4,
                   target_length=None, target_sr=24000, cache_dir=None,
                   shuffle=True):
    """
    Create a DataLoader for LibriSpeech.

    Args:
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        target_length: Target audio length in samples (None for variable)
        target_sr: Target sample rate
        cache_dir: Cache directory
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = LibriSpeechDataset(
        split=split,
        target_length=target_length,
        target_sr=target_sr,
        cache_dir=cache_dir
    )

    collate_fn = collate_fn_fixed_length if target_length else collate_fn_variable_length

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


class AudioChunker:
    """Utility for chunking audio into fixed-size segments for real-time processing."""

    def __init__(self, chunk_size, hop_size=None, sample_rate=24000):
        """
        Initialize audio chunker.

        Args:
            chunk_size: Size of each chunk in samples
            hop_size: Hop size between chunks (defaults to chunk_size for no overlap)
            sample_rate: Audio sample rate
        """
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size
        self.sample_rate = sample_rate

    def chunk_audio(self, audio):
        """
        Split audio into chunks.

        Args:
            audio: Audio tensor [channels, time]

        Returns:
            List of audio chunks, each [channels, chunk_size]
        """
        channels, length = audio.shape
        chunks = []

        for start in range(0, length - self.chunk_size + 1, self.hop_size):
            end = start + self.chunk_size
            chunk = audio[:, start:end]
            chunks.append(chunk)

        # Handle remaining audio
        if length % self.hop_size != 0:
            last_chunk = audio[:, -self.chunk_size:]
            chunks.append(last_chunk)

        return chunks

    def reconstruct_audio(self, chunks, original_length=None):
        """
        Reconstruct audio from chunks using overlap-add.

        Args:
            chunks: List of audio chunks
            original_length: Original audio length (for cropping)

        Returns:
            Reconstructed audio tensor [channels, time]
        """
        if not chunks:
            return torch.zeros(1, 0)

        channels = chunks[0].shape[0]

        # Calculate output length
        if original_length is None:
            output_length = (len(chunks) - 1) * self.hop_size + self.chunk_size
        else:
            output_length = original_length

        # Initialize output
        output = torch.zeros(channels, output_length)
        overlap_count = torch.zeros(output_length)

        # Overlap-add
        for i, chunk in enumerate(chunks):
            start = i * self.hop_size
            end = min(start + self.chunk_size, output_length)
            chunk_length = end - start

            output[:, start:end] += chunk[:, :chunk_length]
            overlap_count[start:end] += 1

        # Average overlapping regions
        overlap_count = overlap_count.clamp(min=1)
        output = output / overlap_count.unsqueeze(0)

        return output
