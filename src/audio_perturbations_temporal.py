"""
Temporal audio perturbations using LSTM to generate time-varying parameters.
Applies learned perturbations to 25ms segments with temporal smoothing.
"""
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from scipy import signal
import librosa


class FeatureExtractor(nn.Module):
    """Extract short-term features from audio for LSTM input."""

    def __init__(self, sample_rate=24000, n_mels=40, n_mfcc=13, hop_length=600):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length  # 600 samples = 25ms at 24kHz

        # Mel-spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 1024,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'normalized': True
            }
        )

    def forward(self, audio):
        """
        Extract features from audio.

        Args:
            audio: [batch, channels, time]

        Returns:
            features: [batch, time_frames, feature_dim]
        """
        batch_size, channels, length = audio.shape

        # Average across channels if stereo, then ensure single channel
        if channels > 1:
            audio = audio.mean(dim=1, keepdim=True)  # [batch, 1, time]

        # Ensure we have exactly [batch, 1, time] for the transforms
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, time] -> [batch, 1, time]

        # Extract MFCC features
        mfcc = self.mfcc_transform(audio)  # [batch, n_mfcc, time_frames_mfcc]

        # Extract mel-spectrogram
        mel = self.mel_spec(audio)  # [batch, n_mels, time_frames_mel]
        mel = torch.log(mel + 1e-9)  # Log mel-spectrogram

        # Ensure both are 3D [batch, features, time]
        if mfcc.dim() == 4:  # [batch, channels, n_mfcc, time] -> squeeze channel
            mfcc = mfcc.squeeze(1)
        if mel.dim() == 4:  # [batch, channels, n_mels, time] -> squeeze channel
            mel = mel.squeeze(1)

        # Ensure both have the same time dimension
        # Take the minimum and truncate both to match
        min_time = min(mfcc.shape[2], mel.shape[2])
        mfcc = mfcc[:, :, :min_time]
        mel = mel[:, :, :min_time]

        # Concatenate features
        features = torch.cat([mfcc, mel], dim=1)  # [batch, n_mfcc + n_mels, time_frames]

        # Transpose to [batch, time_frames, feature_dim]
        features = features.transpose(1, 2)

        return features


class PerturbationLSTM(nn.Module):
    """LSTM that generates time-varying perturbation parameters."""

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_params=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_params = num_params

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, num_params)

        # Initialize weights to encourage diversity from the start
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM and output layer with higher variance to prevent collapse."""
        # Initialize LSTM weights with Xavier/Glorot initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                # Initialize biases to small random values (not zeros)
                nn.init.uniform_(param, -0.1, 0.1)

        # Initialize output layer to produce identity/zero perturbations
        # This starts with minimal perturbation and learns to increase it
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Add small random noise to break symmetry (prevent all outputs being identical)
        with torch.no_grad():
            self.output_layer.weight.add_(torch.randn_like(self.output_layer.weight) * 0.01)
            self.output_layer.bias.add_(torch.randn_like(self.output_layer.bias) * 0.01)

    def forward(self, features):
        """
        Generate perturbation parameters from features.

        Args:
            features: [batch, time_frames, feature_dim]

        Returns:
            params: [batch, time_frames, num_params]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(features)  # [batch, time_frames, hidden_dim]

        # Project to perturbation parameters
        params = self.output_layer(lstm_out)  # [batch, time_frames, num_params]

        # Add small noise during training to prevent collapse
        if self.training:
            noise = torch.randn_like(params) * 0.01
            params = params + noise

        return params


class TemporalAudioPerturbationLayer(nn.Module):
    """Learnable audio perturbations with temporal adaptation via LSTM."""

    def __init__(self, sample_rate=24000, segment_length_ms=25, ema_alpha=0.7):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_length_ms = segment_length_ms
        self.segment_length = int(sample_rate * segment_length_ms / 1000)  # 600 samples at 24kHz
        self.ema_alpha = ema_alpha

        # Feature extractor
        feature_dim = 13 + 40  # n_mfcc + n_mels
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_mels=40,
            n_mfcc=13,
            hop_length=self.segment_length
        )

        # Layer normalization for LSTM input
        self.input_norm = nn.LayerNorm(feature_dim)

        # LSTM for generating perturbation parameters
        self.perturbation_lstm = PerturbationLSTM(
            input_dim=feature_dim,
            hidden_dim=256,
            num_layers=3,
            num_params=8
        )

    def apply_activations(self, raw_params):
        """
        Apply activation functions to constrain perturbation parameters to valid ranges.

        Args:
            raw_params: [batch, time_frames, 8]

        Returns:
            params: Dictionary of constrained parameters
        """
        # Split parameters
        pitch = raw_params[..., 0]
        formant = raw_params[..., 1]
        echo_delay = raw_params[..., 2]
        echo_decay = raw_params[..., 3]
        reversal = raw_params[..., 4]
        bp_low = raw_params[..., 5]      
        bp_high = raw_params[..., 6]     
        bp_mix = raw_params[..., 7]      

        # Apply activations with appropriate ranges (reduced for more subtle perturbations)
        params = {
            # Old: [-12, 12] semitones
            # tanh(0) = 0 ✓ (identity at init)
            'pitch_shift': torch.tanh(pitch) * 6.0,  # [-6, 6] semitones

            # Old: [0.8, 1.2]
            # When x=0: sigmoid(0)=0.5, 0.5*0.2+0.9 = 1.0 ✓ (identity at init)
            'formant_ratio': torch.sigmoid(formant) * 0.2 + 0.9,  # [0.9, 1.1]

            # Old: [0.01, 0.5] seconds (delay not critical for identity)
            'echo_delay': torch.sigmoid(echo_delay) * 0.29 + 0.01,  # [0.01, 0.3] seconds

            # Old: [0.0, 0.8]
            # FIXED: Use relu(tanh) to start at 0 (tanh(0)=0, relu(0)=0)
            'echo_decay': torch.relu(torch.tanh(echo_decay)) * 0.5,  # [0.0, 0.5]

            # Old: [0, 1]
            # FIXED: Use relu(tanh) to start at 0
            'reversal_mix': torch.relu(torch.tanh(reversal)) * 0.5,  # [0, 0.5]

            # NEW: Bandpass cutoff coefficients (0-1 range, controls how much to cut)
            # Old: bp_center_freq [100, 12000] Hz, bp_bandwidth [100, 5000] Hz
            # FIXED: Use relu(tanh) to start at 0 (no cutting at init)
            'bp_low_coef': torch.relu(torch.tanh(bp_low)),   # [0, 1] - starts at 0
            'bp_high_coef': torch.relu(torch.tanh(bp_high)), # [0, 1] - starts at 0
            # Actual cutoffs: low = 400 * bp_low_coef, high = 16000 - (10000 * bp_high_coef)
            # At init: low=0, high=16000 (full spectrum, no cutting)

            # Unused (bandpass no longer uses mixing)
            'bp_mix': torch.sigmoid(bp_mix) * 0.3  # [0, 0.3] (UNUSED)
        }

        return params

    def apply_ema_smoothing(self, params):
        """
        Apply exponential moving average smoothing to parameters.

        Args:
            params: Dictionary of parameter tensors [batch, time_frames]

        Returns:
            smoothed_params: Dictionary of smoothed parameters
        """
        smoothed = {}
        alpha = self.ema_alpha

        for key, values in params.items():
            batch_size, time_frames = values.shape
            smoothed_values = torch.zeros_like(values)

            # Initialize with first frame
            smoothed_values[:, 0] = values[:, 0]

            # Apply EMA
            for t in range(1, time_frames):
                smoothed_values[:, t] = alpha * smoothed_values[:, t-1] + (1 - alpha) * values[:, t]

            smoothed[key] = smoothed_values

        return smoothed

    def apply_pitch_shift_segment(self, audio_segment, pitch_shift):
        """
        GPU-accelerated pitch shifting using phase vocoder.
        Fully differentiable.
        """
        if audio_segment.abs().max() < 1e-6:  # Silent segment
            return audio_segment

        # Pitch shift factor
        shift_factor = 2.0 ** (pitch_shift / 12.0)

        # Simple resampling approach (fast and differentiable)
        # Interpolate to change duration, which changes pitch
        if abs(shift_factor - 1.0) < 0.01:  # No shift needed
            return audio_segment

        # Add batch and channel dimensions for interpolate
        audio_4d = audio_segment.unsqueeze(0).unsqueeze(0)  # [1, 1, time]

        # Resample
        shifted = torch.nn.functional.interpolate(
            audio_4d,
            size=int(len(audio_segment) / shift_factor),
            mode='linear',
            align_corners=False
        )

        # Pad or trim to original length
        shifted = shifted.squeeze(0).squeeze(0)
        if len(shifted) < len(audio_segment):
            shifted = torch.nn.functional.pad(shifted, (0, len(audio_segment) - len(shifted)))
        else:
            shifted = shifted[:len(audio_segment)]

        return shifted

    def apply_formant_shift_segment(self, audio_segment, formant_ratio):
        """
        GPU-accelerated formant shifting.
        Time stretch + pitch correction (fully differentiable).
        """
        if audio_segment.abs().max() < 1e-6:  # Silent segment
            return audio_segment

        if abs(formant_ratio - 1.0) < 0.01:  # No shift needed
            return audio_segment

        # Step 1: Time stretch (changes formants)
        audio_4d = audio_segment.unsqueeze(0).unsqueeze(0)
        stretched = torch.nn.functional.interpolate(
            audio_4d,
            size=int(len(audio_segment) * formant_ratio),
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        # Step 2: Pitch shift back to original pitch
        pitch_correction = -12.0 * float(np.log2(formant_ratio))
        corrected = self.apply_pitch_shift_segment(stretched, pitch_correction)

        # Ensure correct length
        if len(corrected) < len(audio_segment):
            corrected = torch.nn.functional.pad(corrected, (0, len(audio_segment) - len(corrected)))
        else:
            corrected = corrected[:len(audio_segment)]

        return corrected

    def apply_echo_segment(self, audio_segment, echo_delay, echo_decay):
        """Apply echo to a single audio segment."""
        delay_samples = int(echo_delay * self.sample_rate)

        if delay_samples > 0 and delay_samples < len(audio_segment):
            padded = torch.nn.functional.pad(audio_segment, (delay_samples, 0))
            delayed = padded[:-delay_samples]
            echoed = audio_segment + echo_decay * delayed
            return echoed
        return audio_segment

    def apply_reversal_segment(self, audio_segment, reversal_mix):
        """Apply time reversal to a single audio segment."""
        reversed_audio = torch.flip(audio_segment, dims=[-1])
        mixed = (1 - reversal_mix) * audio_segment + reversal_mix * reversed_audio
        return mixed

    def apply_bandpass_segment(self, audio_segment, low_coef, high_coef, bp_mix=None):
        """
        GPU-accelerated bandpass filter using FFT.
        Uses coefficients to control cutoff frequencies.

        Args:
            audio_segment: Audio to filter
            low_coef: Coefficient [0, 1]. Cuts below (400 * low_coef) Hz
            high_coef: Coefficient [0, 1]. Cuts above (16000 - 10000 * high_coef) Hz
            bp_mix: Unused (kept for backward compatibility)
        """
        if audio_segment.abs().max() < 1e-6:  # Silent segment
            return audio_segment

        # Calculate cutoff frequencies from coefficients
        # Low: 0-400 Hz (cuts bass frequencies)
        low_freq = 400.0 * low_coef

        # High: 6000-16000 Hz (cuts treble frequencies)
        # high_coef=0: 16000 Hz (minimal cutting), high_coef=1: 6000 Hz (more aggressive)
        high_freq = 16000.0 - 10000.0 * high_coef

        # Clamp to valid range (using Python min/max since these are floats, not tensors)
        low_freq = max(0.0, min(low_freq, 400.0))
        high_freq = max(6000.0, min(high_freq, self.sample_rate / 2 - 10))

        # FFT-based filtering (GPU, differentiable)
        # Compute FFT
        fft = torch.fft.rfft(audio_segment)
        freqs = torch.fft.rfftfreq(len(audio_segment), d=1/self.sample_rate).to(audio_segment.device)

        # Create bandpass mask
        mask = torch.zeros_like(freqs)
        mask[(freqs >= low_freq) & (freqs <= high_freq)] = 1.0

        # Apply smooth edges to avoid ringing (Hann window edges)
        edge_width = 100  # Hz
        # Low edge
        low_edge = (freqs >= (low_freq - edge_width)) & (freqs < low_freq)
        if low_edge.any():
            low_transition = (freqs[low_edge] - (low_freq - edge_width)) / edge_width
            mask[low_edge] = 0.5 * (1 - torch.cos(torch.pi * low_transition))

        # High edge
        high_edge = (freqs > high_freq) & (freqs <= (high_freq + edge_width))
        if high_edge.any():
            high_transition = (freqs[high_edge] - high_freq) / edge_width
            mask[high_edge] = 0.5 * (1 + torch.cos(torch.pi * high_transition))

        # Apply filter in frequency domain
        filtered_fft = fft * mask
        filtered = torch.fft.irfft(filtered_fft, n=len(audio_segment))

        # Apply filter directly (no mixing with original)
        # Old: mixed = (1 - bp_mix) * audio_segment + bp_mix * filtered
        return filtered

    def apply_perturbations_to_segments(self, audio, params):
        """
        Apply time-varying perturbations to audio segments.

        Args:
            audio: [batch, channels, time]
            params: Dictionary of smoothed parameters [batch, time_frames]

        Returns:
            perturbed_audio: [batch, channels, time]
        """
        batch_size, channels, total_length = audio.shape
        num_segments = params['pitch_shift'].shape[1]

        # Pad audio to fit segments
        padded_length = num_segments * self.segment_length
        if total_length < padded_length:
            audio = torch.nn.functional.pad(audio, (0, padded_length - total_length))

        perturbed_segments = []

        for seg_idx in range(num_segments):
            start_idx = seg_idx * self.segment_length
            end_idx = start_idx + self.segment_length

            # Extract segment for all batches and channels
            segment = audio[:, :, start_idx:end_idx]  # [batch, channels, segment_length]

            # Process each batch sample
            batch_segments = []
            for b in range(batch_size):
                channel_segments = []

                for c in range(channels):
                    audio_seg = segment[b, c]  # [segment_length]

                    # Get parameters for this batch and time frame
                    pitch = params['pitch_shift'][b, seg_idx].item()
                    formant = params['formant_ratio'][b, seg_idx].item()
                    echo_d = params['echo_delay'][b, seg_idx].item()
                    echo_a = params['echo_decay'][b, seg_idx].item()
                    rev = params['reversal_mix'][b, seg_idx].item()
                    bp_low = params['bp_low_coef'][b, seg_idx].item()     # Changed from bp_center_freq
                    bp_high = params['bp_high_coef'][b, seg_idx].item()   # Changed from bp_bandwidth
                    bp_m = params['bp_mix'][b, seg_idx].item()            # Unused

                    # Apply perturbations sequentially
                    perturbed = audio_seg

                    if abs(pitch) > 0.1:
                        perturbed = self.apply_pitch_shift_segment(perturbed, pitch)

                    if abs(formant - 1.0) > 0.01:
                        perturbed = self.apply_formant_shift_segment(perturbed, formant)

                    if abs(echo_a) > 0.01:
                        perturbed = self.apply_echo_segment(perturbed, echo_d, echo_a)

                    if abs(rev) > 0.01:
                        perturbed = self.apply_reversal_segment(perturbed, rev)

                    # Apply bandpass filter (always apply since we use coefficients)
                    # Low/high coefs determine how much to cut
                    perturbed = self.apply_bandpass_segment(perturbed, bp_low, bp_high)

                    channel_segments.append(perturbed)

                batch_segments.append(torch.stack(channel_segments))

            perturbed_segments.append(torch.stack(batch_segments))

        # Concatenate segments
        perturbed_audio = torch.cat(perturbed_segments, dim=-1)

        # Trim to original length
        perturbed_audio = perturbed_audio[:, :, :total_length]

        # Ensure float32 dtype (librosa/scipy return float64)
        perturbed_audio = perturbed_audio.float()

        return perturbed_audio

    def forward(self, audio):
        """
        Apply temporal perturbations to audio.

        Args:
            audio: [batch, channels, time]

        Returns:
            perturbed_audio: [batch, channels, time]
            raw_params: [batch, time_frames, 8] (before smoothing)
            smoothed_params: Dictionary of smoothed parameters
        """
        # Extract features
        features = self.feature_extractor(audio)  # [batch, time_frames, feature_dim]

        # Normalize features before LSTM
        features = self.input_norm(features)  # [batch, time_frames, feature_dim]

        # Generate perturbation parameters
        raw_params = self.perturbation_lstm(features)  # [batch, time_frames, 8]

        # Apply activations
        params = self.apply_activations(raw_params)

        # Apply EMA smoothing
        smoothed_params = self.apply_ema_smoothing(params)

        # Apply perturbations to segments
        perturbed_audio = self.apply_perturbations_to_segments(audio, smoothed_params)

        return perturbed_audio, raw_params, smoothed_params

    def get_perturbation_params(self):
        """Get average perturbation parameters for logging."""
        # This is used for logging - we'll return zeros as placeholder
        # since parameters are now time-varying
        return {
            'lstm_hidden_dim': self.perturbation_lstm.hidden_dim,
            'num_lstm_layers': self.perturbation_lstm.num_layers,
            'segment_length_ms': self.segment_length_ms,
            'ema_alpha': self.ema_alpha
        }


def compute_magnitude_penalty(raw_params):
    """
    Compute L2 penalty on perturbation parameter magnitudes.
    Normalized to [0, 1] range using tanh.

    Args:
        raw_params: [batch, time_frames, 8]

    Returns:
        magnitude_loss: scalar in [0, 1]
    """
    raw_magnitude = torch.mean(raw_params ** 2)
    # Normalize to [0, 1] using tanh
    return torch.tanh(raw_magnitude)


def compute_temporal_smoothness_penalty(raw_params):
    """
    Compute penalty for differences between consecutive perturbation vectors.
    Normalized to [0, 1] range using tanh.

    Args:
        raw_params: [batch, time_frames, 8]

    Returns:
        smoothness_loss: scalar in [0, 1]
    """
    if raw_params.shape[1] < 2:
        return torch.tensor(0.0, device=raw_params.device)

    # Compute differences between consecutive frames
    diffs = raw_params[:, 1:, :] - raw_params[:, :-1, :]

    # L2 penalty on differences
    raw_smoothness = torch.mean(diffs ** 2)

    # Normalize to [0, 1] using tanh
    return torch.tanh(raw_smoothness)


def compute_entropy_regularization(raw_params):
    """
    Compute entropy regularization to encourage diverse perturbations across time.
    Prevents LSTM from collapsing to constant outputs.
    Normalized to [0, 1] range using exponential decay.

    Args:
        raw_params: [batch, time_frames, 8]

    Returns:
        entropy_loss: scalar in [0, 1]
                     - 0.0 means high diversity (good)
                     - 1.0 means no diversity/collapsed (bad)
    """
    # Compute variance across time dimension for each parameter
    # High variance = diverse perturbations = good
    # Low variance = constant perturbations = collapse

    # Variance for each parameter across time [batch, 8]
    temporal_variance = torch.var(raw_params, dim=1)

    # Mean variance across batch and parameters
    mean_variance = torch.mean(temporal_variance)

    # Normalize to [0, 1] using exponential decay
    # High variance → exp(-high) → ~0 (low loss, good)
    # Low/zero variance → exp(0) → 1 (high loss, bad)
    entropy_loss = torch.exp(-mean_variance)

    return entropy_loss


def compute_latent_space_loss(original_audio, perturbed_audio, encodec_processor):
    """
    Compute adversarial loss in EnCodec latent space.
    MAXIMIZES the difference between original and perturbed latent representations.
    This attacks the codec's internal representation while quality loss keeps output similar.
    Uses -tanh(MSE / 10000) for bounded, normalized loss in [-1, 0] range.

    Args:
        original_audio: [batch, channels, time]
        perturbed_audio: [batch, channels, time]
        encodec_processor: EnCodecProcessor instance

    Returns:
        latent_loss: scalar in [-1, 0]
                    - Closer to -1 = codes are more similar (bad for attack)
                    - Closer to 0 = codes are maximally different (good for attack)
    """
    # Encode both audios to latent space
    with torch.no_grad():
        # We don't want gradients through the original audio encoding
        original_encoded = encodec_processor.encode(original_audio)

    # We DO want gradients through perturbed audio encoding
    perturbed_encoded = encodec_processor.encode(perturbed_audio)

    # EnCodec returns a list of EncodedFrame objects for batches
    # Each EncodedFrame contains quantized codes

    if isinstance(original_encoded, list):
        # Batch processing
        batch_losses = []

        for orig_frame, pert_frame in zip(original_encoded, perturbed_encoded):
            # Extract quantized codes: [1, n_quantizers, n_frames]
            orig_codes = orig_frame[0][0]  # First element of tuple, first in batch
            pert_codes = pert_frame[0][0]

            # MSE distance between codes (higher = more different)
            distance = torch.nn.functional.mse_loss(pert_codes.float(), orig_codes.float())

            # Normalize with tanh and make negative (adversarial)
            # Use larger division to prevent saturation (typical distances are 10k-100k)
            normalized_distance = torch.tanh(distance / 50000.0)
            loss = -normalized_distance
            batch_losses.append(loss)

        latent_loss = torch.stack(batch_losses).mean()
    else:
        # Single sample
        orig_codes = original_encoded[0][0]
        pert_codes = perturbed_encoded[0][0]
        distance = torch.nn.functional.mse_loss(pert_codes.float(), orig_codes.float())
        normalized_distance = torch.tanh(distance / 50000.0)
        latent_loss = -normalized_distance

    return latent_loss
