"""
EnCodec utilities for encoding and decoding audio.
"""
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio


class EnCodecProcessor:
    """Wrapper for EnCodec model with encoding/decoding capabilities."""

    def __init__(self, model_name="encodec_24khz", device="cuda"):
        """
        Initialize EnCodec processor.

        Args:
            model_name: EnCodec model variant (encodec_24khz or encodec_48khz)
            device: Device to run model on
        """
        self.device = device
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(6.0)  # 6 kbps
        self.model.to(device)

        # Keep in training mode to allow gradients through RNN layers
        # Note: EnCodec parameters remain trainable (requires_grad=True) to allow
        # gradient flow, but they won't be updated since they're excluded from optimizer
        self.model.train()

        self.sample_rate = self.model.sample_rate
        self.channels = self.model.channels

    def encode(self, audio):
        """
        Encode audio to latent codes.

        Args:
            audio: Audio tensor [batch, channels, time] or [channels, time]

        Returns:
            Encoded frames with shape [batch, n_quantizers, frames]
        """
        # Handle batched input
        if audio.dim() == 3:
            # Process each sample in the batch individually
            batch_size = audio.shape[0]
            encoded_batch = []

            for i in range(batch_size):
                single_audio = audio[i:i+1]  # Keep batch dim [1, channels, time]

                # Ensure correct sample rate and channels
                single_audio = convert_audio(
                    single_audio,
                    self.sample_rate,
                    self.model.sample_rate,
                    self.model.channels
                )

                # Encode
                encoded = self.model.encode(single_audio.to(self.device))
                encoded_batch.append(encoded)

            # Combine batch results
            # EnCodec returns a list of tuples, need to combine them properly
            return encoded_batch
        else:
            # Single sample (add batch dimension if needed)
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)

            # Ensure correct sample rate and channels
            audio = convert_audio(
                audio,
                self.sample_rate,
                self.model.sample_rate,
                self.model.channels
            )

            # Encode
            encoded_frames = self.model.encode(audio.to(self.device))

            return encoded_frames

    def decode(self, encoded_frames):
        """
        Decode latent codes back to audio.

        Args:
            encoded_frames: Encoded frames from encode() - can be a list for batches

        Returns:
            Reconstructed audio tensor [batch, channels, time]
        """
        # Handle batched encoded frames (list of encoded frames)
        if isinstance(encoded_frames, list):
            decoded_batch = []
            for encoded in encoded_frames:
                audio = self.model.decode(encoded)
                decoded_batch.append(audio)
            return torch.cat(decoded_batch, dim=0)
        else:
            # Single encoded frame
            audio = self.model.decode(encoded_frames)
            return audio

    def encode_decode(self, audio):
        """
        Full encode-decode cycle.

        Args:
            audio: Audio tensor [batch, channels, time]

        Returns:
            Reconstructed audio tensor [batch, channels, time]
        """
        # Handle batched input by processing each sample
        if audio.dim() == 3:
            batch_size = audio.shape[0]
            reconstructed_batch = []

            for i in range(batch_size):
                single_audio = audio[i:i+1]  # Keep batch dim [1, channels, time]
                encoded = self.encode(single_audio)
                decoded = self.decode(encoded)
                reconstructed_batch.append(decoded)

            return torch.cat(reconstructed_batch, dim=0)
        else:
            # Single sample
            encoded = self.encode(audio)
            decoded = self.decode(encoded)
            return decoded

    def get_latent_shape(self, audio_length):
        """
        Calculate latent shape for given audio length.

        Args:
            audio_length: Length of audio in samples

        Returns:
            Tuple of (n_frames, n_quantizers)
        """
        # EnCodec has a hop length that determines frame rate
        hop_length = self.model.encoder.hop_length
        n_frames = audio_length // hop_length
        n_quantizers = self.model.quantizer.n_q
        return n_frames, n_quantizers


def load_audio(file_path, target_sr=24000):
    """
    Load audio file and convert to target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Audio tensor [channels, time] and sample rate
    """
    audio, sr = torchaudio.load(file_path)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr

    return audio, sr


def save_audio(audio, file_path, sample_rate=24000):
    """
    Save audio tensor to file.

    Args:
        audio: Audio tensor [channels, time]
        file_path: Output file path
        sample_rate: Audio sample rate
    """
    # Ensure audio is on CPU
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu()

    # Normalize to prevent clipping
    audio = audio / (audio.abs().max() + 1e-8) * 0.95

    torchaudio.save(file_path, audio, sample_rate)
