"""
Audio quality metrics for evaluating perturbed audio.
Implements PESQ, STOI, and other perceptual metrics.
"""
import torch
import numpy as np

# PESQ requires gcc to compile, so make it optional
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: PESQ not available (requires gcc for compilation)")

from pystoi import stoi
import torchaudio


def calculate_pesq(reference, degraded, sample_rate=16000, mode='wb'):
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality) score.

    Args:
        reference: Reference audio [time] or [channels, time]
        degraded: Degraded audio [time] or [channels, time]
        sample_rate: Sample rate (must be 8000 or 16000 for PESQ)
        mode: 'wb' (wideband, 16kHz) or 'nb' (narrowband, 8kHz)

    Returns:
        PESQ score (higher is better, -0.5 to 4.5) or 0.0 if PESQ not available
    """
    if not PESQ_AVAILABLE:
        return 0.0

    # Convert to numpy
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.cpu().numpy()

    # Convert to mono if stereo
    if reference.ndim == 2:
        reference = reference.mean(axis=0)
    if degraded.ndim == 2:
        degraded = degraded.mean(axis=0)

    # PESQ only supports 8kHz and 16kHz
    if sample_rate not in [8000, 16000]:
        # Resample to 16kHz
        import librosa
        reference = librosa.resample(reference, orig_sr=sample_rate, target_sr=16000)
        degraded = librosa.resample(degraded, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
        mode = 'wb'

    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    try:
        score = pesq(sample_rate, reference, degraded, mode)
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        score = 0.0

    return score


def calculate_stoi(reference, degraded, sample_rate=16000):
    """
    Calculate STOI (Short-Time Objective Intelligibility) score.

    Args:
        reference: Reference audio [time] or [channels, time]
        degraded: Degraded audio [time] or [channels, time]
        sample_rate: Sample rate (should be >= 10kHz)

    Returns:
        STOI score (0 to 1, higher is better)
    """
    # Convert to numpy
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.cpu().numpy()

    # Convert to mono if stereo
    if reference.ndim == 2:
        reference = reference.mean(axis=0)
    if degraded.ndim == 2:
        degraded = degraded.mean(axis=0)

    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    try:
        score = stoi(reference, degraded, sample_rate, extended=False)
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        score = 0.0

    return score


def calculate_snr(reference, degraded):
    """
    Calculate Signal-to-Noise Ratio.

    Args:
        reference: Reference audio
        degraded: Degraded audio

    Returns:
        SNR in dB
    """
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.cpu().numpy()

    # Ensure same shape
    min_len = min(len(reference.flatten()), len(degraded.flatten()))
    reference = reference.flatten()[:min_len]
    degraded = degraded.flatten()[:min_len]

    # Calculate noise
    noise = reference - degraded

    # Calculate SNR
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return 100.0  # Very high SNR

    snr = 10 * np.log10(signal_power / noise_power)

    return snr


def calculate_lsd(reference, degraded, sample_rate=24000):
    """
    Calculate Log-Spectral Distance.

    Args:
        reference: Reference audio
        degraded: Degraded audio
        sample_rate: Sample rate

    Returns:
        LSD value (lower is better)
    """
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    if isinstance(degraded, torch.Tensor):
        degraded = degraded.cpu().numpy()

    # Convert to mono if needed
    if reference.ndim == 2:
        reference = reference.mean(axis=0)
    if degraded.ndim == 2:
        degraded = degraded.mean(axis=0)

    # Ensure same length
    min_len = min(len(reference), len(degraded))
    reference = reference[:min_len]
    degraded = degraded[:min_len]

    # Compute spectrograms using STFT
    import librosa
    n_fft = 2048
    hop_length = 512

    ref_spec = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop_length))
    deg_spec = np.abs(librosa.stft(degraded, n_fft=n_fft, hop_length=hop_length))

    # Add small constant to avoid log(0)
    ref_spec = ref_spec + 1e-10
    deg_spec = deg_spec + 1e-10

    # Calculate LSD (average over frequency bins, then over time frames)
    lsd = np.mean(np.sqrt(np.mean((np.log10(ref_spec) - np.log10(deg_spec)) ** 2, axis=0)))

    return lsd


def evaluate_audio_quality(reference, degraded, sample_rate=24000):
    """
    Comprehensive audio quality evaluation.

    Args:
        reference: Reference audio tensor [batch, channels, time]
        degraded: Degraded audio tensor [batch, channels, time]
        sample_rate: Sample rate

    Returns:
        Dict with quality metrics
    """
    batch_size = reference.shape[0]

    pesq_scores = []
    stoi_scores = []
    snr_scores = []
    lsd_scores = []

    for i in range(batch_size):
        ref = reference[i]
        deg = degraded[i]

        # Calculate metrics
        # pesq_score = calculate_pesq(ref, deg, sample_rate)
        stoi_score = calculate_stoi(ref, deg, sample_rate)
        snr_score = calculate_snr(ref, deg)
        lsd_score = calculate_lsd(ref, deg, sample_rate)

        pesq_scores.append(0.0)  # PESQ disabled for simplicity
        stoi_scores.append(stoi_score)
        snr_scores.append(snr_score)
        lsd_scores.append(lsd_score)

    results = {
        'pesq': np.mean(pesq_scores),
        'pesq_std': np.std(pesq_scores),
        'stoi': np.mean(stoi_scores),
        'stoi_std': np.std(stoi_scores),
        'snr': np.mean(snr_scores),
        'snr_std': np.std(snr_scores),
        'lsd': np.mean(lsd_scores),
        'lsd_std': np.std(lsd_scores),
    }

    return results


def compute_quality_loss(reference, degraded):
    """
    Compute loss to maintain audio quality.
    Lower loss means higher quality (closer to reference).
    Normalized to [0, 1] range using tanh.

    Args:
        reference: Reference audio tensor [batch, channels, time]
        degraded: Degraded audio tensor [batch, channels, time]

    Returns:
        Quality loss tensor in [0, 1]
    """
    # L1 loss in time domain
    time_loss = torch.nn.functional.l1_loss(degraded, reference)

    # Spectral loss
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).to(reference.device)

    # Compute spectrograms
    ref_spec = torch.stft(
        reference.reshape(-1, reference.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    deg_spec = torch.stft(
        degraded.reshape(-1, degraded.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )

    # Magnitude loss
    ref_mag = torch.abs(ref_spec)
    deg_mag = torch.abs(deg_spec)
    spectral_loss = torch.nn.functional.l1_loss(deg_mag, ref_mag)

    # Combined loss
    raw_loss = time_loss + 0.5 * spectral_loss

    # Normalize to [0, 1] using tanh
    # Scale by 5 to make typical values (0.01-0.5) map to a good range
    normalized_loss = torch.tanh(raw_loss * 5.0)

    return normalized_loss
