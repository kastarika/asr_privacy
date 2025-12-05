"""
Audio-domain perturbations for adversarial attacks on ASR systems.
Implements pitch shifting, formant shifting, echo, reversal, and bandpass filters.
"""
import torch
import torchaudio
import numpy as np
from scipy import signal
import librosa


class AudioPerturbationLayer(torch.nn.Module):
    """Learnable audio perturbations applied in the time/frequency domain."""

    def __init__(self, sample_rate=24000):
        super().__init__()
        self.sample_rate = sample_rate

        # Learnable parameters for perturbations
        self.pitch_shift_semitones = torch.nn.Parameter(torch.zeros(1))
        self.formant_shift_ratio = torch.nn.Parameter(torch.ones(1))
        self.echo_delay = torch.nn.Parameter(torch.tensor([0.1]))  # seconds
        self.echo_decay = torch.nn.Parameter(torch.tensor([0.3]))
        self.reversal_mix = torch.nn.Parameter(torch.zeros(1))  # 0 = no reversal, 1 = full reversal

        # Bandpass filter parameters (learnable center frequency and bandwidth)
        self.bp_center_freq = torch.nn.Parameter(torch.tensor([1000.0]))  # Hz
        self.bp_bandwidth = torch.nn.Parameter(torch.tensor([2000.0]))  # Hz
        self.bp_mix = torch.nn.Parameter(torch.zeros(1))  # 0 = no filter, 1 = full filter

    def apply_pitch_shift(self, audio):
        """
        Apply pitch shifting to audio.

        Args:
            audio: Tensor [batch, channels, time]

        Returns:
            Pitch-shifted audio
        """
        batch_size, channels, length = audio.shape
        shifted_audio = []

        # Clamp pitch shift to reasonable range (-12 to +12 semitones)
        pitch_shift = torch.clamp(self.pitch_shift_semitones, -12, 12)

        for b in range(batch_size):
            for c in range(channels):
                audio_np = audio[b, c].detach().cpu().numpy()

                # Use librosa for pitch shifting
                shifted = librosa.effects.pitch_shift(
                    audio_np,
                    sr=self.sample_rate,
                    n_steps=pitch_shift.item()
                )

                # Ensure same length
                if len(shifted) < length:
                    shifted = np.pad(shifted, (0, length - len(shifted)))
                else:
                    shifted = shifted[:length]

                shifted_audio.append(torch.from_numpy(shifted))

        shifted_audio = torch.stack(shifted_audio).reshape(batch_size, channels, length)
        return shifted_audio.to(audio.device)

    def apply_formant_shift(self, audio):
        """
        Apply formant shifting (vocal tract length normalization).

        Args:
            audio: Tensor [batch, channels, time]

        Returns:
            Formant-shifted audio
        """
        batch_size, channels, length = audio.shape
        shifted_audio = []

        # Clamp formant shift to reasonable range (0.8 to 1.2)
        formant_ratio = torch.clamp(self.formant_shift_ratio, 0.8, 1.2)

        for b in range(batch_size):
            for c in range(channels):
                audio_np = audio[b, c].detach().cpu().numpy()

                # Formant shifting via time stretching + pitch correction
                # Stretch time by formant_ratio
                stretched = librosa.effects.time_stretch(audio_np, rate=formant_ratio.item())

                # Pitch shift back to original pitch
                shifted = librosa.effects.pitch_shift(
                    stretched,
                    sr=self.sample_rate,
                    n_steps=-12 * np.log2(formant_ratio.item())
                )

                # Ensure same length
                if len(shifted) < length:
                    shifted = np.pad(shifted, (0, length - len(shifted)))
                else:
                    shifted = shifted[:length]

                shifted_audio.append(torch.from_numpy(shifted))

        shifted_audio = torch.stack(shifted_audio).reshape(batch_size, channels, length)
        return shifted_audio.to(audio.device)

    def apply_echo(self, audio):
        """
        Apply echo effect to audio.

        Args:
            audio: Tensor [batch, channels, time]

        Returns:
            Audio with echo
        """
        # Clamp parameters
        delay = torch.clamp(self.echo_delay, 0.01, 0.5)  # 10ms to 500ms
        decay = torch.clamp(self.echo_decay, 0.0, 0.8)

        delay_samples = int(delay.item() * self.sample_rate)

        # Create echo by adding delayed signal
        padded = torch.nn.functional.pad(audio, (delay_samples, 0))
        delayed = padded[..., :-delay_samples]
        echoed = audio + decay * delayed

        return echoed

    def apply_reversal(self, audio):
        """
        Apply time reversal to audio.

        Args:
            audio: Tensor [batch, channels, time]

        Returns:
            Mixed original and reversed audio
        """
        # Reverse audio in time dimension
        reversed_audio = torch.flip(audio, dims=[-1])

        # Mix with original based on reversal_mix parameter
        mix = torch.sigmoid(self.reversal_mix)  # 0 to 1
        mixed = (1 - mix) * audio + mix * reversed_audio

        return mixed

    def apply_bandpass_filter(self, audio):
        """
        Apply bandpass filter to audio.

        Args:
            audio: Tensor [batch, channels, time]

        Returns:
            Filtered audio
        """
        # Clamp parameters
        center_freq = torch.clamp(self.bp_center_freq, 100, self.sample_rate / 2 - 100)
        bandwidth = torch.clamp(self.bp_bandwidth, 100, 5000)

        low_freq = center_freq - bandwidth / 2
        high_freq = center_freq + bandwidth / 2

        # Ensure valid frequency range
        low_freq = torch.clamp(low_freq, 50, self.sample_rate / 2 - 50)
        high_freq = torch.clamp(high_freq, low_freq + 100, self.sample_rate / 2 - 10)

        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low = low_freq.item() / nyquist
        high = high_freq.item() / nyquist

        # Use butterworth filter
        sos = signal.butter(4, [low, high], btype='band', output='sos')

        # Apply filter to each channel
        filtered_audio = []
        for b in range(audio.shape[0]):
            for c in range(audio.shape[1]):
                audio_np = audio[b, c].detach().cpu().numpy()
                filtered = signal.sosfilt(sos, audio_np)
                filtered_audio.append(torch.from_numpy(filtered))

        filtered_audio = torch.stack(filtered_audio).reshape(audio.shape)
        filtered_audio = filtered_audio.to(audio.device)

        # Mix with original based on bp_mix parameter
        mix = torch.sigmoid(self.bp_mix)
        mixed = (1 - mix) * audio + mix * filtered_audio

        return mixed

    def forward(self, audio, apply_pitch=True, apply_formant=True, apply_echo=True,
                apply_reversal=True, apply_bandpass=True):
        """
        Apply all perturbations to audio.

        Args:
            audio: Tensor [batch, channels, time]
            apply_*: Boolean flags to enable/disable specific perturbations

        Returns:
            Perturbed audio
        """
        perturbed = audio

        if apply_pitch and self.pitch_shift_semitones.abs() > 0.1:
            perturbed = self.apply_pitch_shift(perturbed)

        if apply_formant and (self.formant_shift_ratio - 1.0).abs() > 0.01:
            perturbed = self.apply_formant_shift(perturbed)

        if apply_echo and self.echo_decay.abs() > 0.01:
            perturbed = self.apply_echo(perturbed)

        if apply_reversal and self.reversal_mix.abs() > 0.01:
            perturbed = self.apply_reversal(perturbed)

        if apply_bandpass and self.bp_mix.abs() > 0.01:
            perturbed = self.apply_bandpass_filter(perturbed)

        return perturbed

    def get_perturbation_params(self):
        """Get current perturbation parameters as dict."""
        return {
            'pitch_shift_semitones': self.pitch_shift_semitones.item(),
            'formant_shift_ratio': self.formant_shift_ratio.item(),
            'echo_delay': self.echo_delay.item(),
            'echo_decay': self.echo_decay.item(),
            'reversal_mix': self.reversal_mix.item(),
            'bp_center_freq': self.bp_center_freq.item(),
            'bp_bandwidth': self.bp_bandwidth.item(),
            'bp_mix': self.bp_mix.item(),
        }
