"""
Test script to apply a single perturbation with custom parameters.
Allows testing individual perturbations (pitch, formant, echo, reversal, bandpass).
"""
import os
from datetime import datetime
import torch
import torchaudio
import argparse
import numpy as np
from pystoi import stoi

from dataset import LibriSpeechDataset
from asr_evaluation import ASREvaluator
from audio_perturbations_temporal import TemporalAudioPerturbationLayer


def apply_pitch_shift(audio_segment, pitch_shift, sample_rate=24000):
    """Apply pitch shift to audio segment."""
    layer = TemporalAudioPerturbationLayer(sample_rate=sample_rate)
    return layer.apply_pitch_shift_segment(audio_segment, pitch_shift)


def apply_formant_shift(audio_segment, formant_ratio, sample_rate=24000):
    """Apply formant shift to audio segment."""
    layer = TemporalAudioPerturbationLayer(sample_rate=sample_rate)
    return layer.apply_formant_shift_segment(audio_segment, formant_ratio)


def apply_echo(audio_segment, echo_delay, echo_decay, sample_rate=24000):
    """Apply echo to audio segment."""
    layer = TemporalAudioPerturbationLayer(sample_rate=sample_rate)
    return layer.apply_echo_segment(audio_segment, echo_delay, echo_decay)


def apply_reversal(audio_segment, reversal_mix, sample_rate=24000):
    """Apply time reversal to audio segment."""
    layer = TemporalAudioPerturbationLayer(sample_rate=sample_rate)
    return layer.apply_reversal_segment(audio_segment, reversal_mix)


def apply_bandpass(audio_segment, low_coef, high_coef, sample_rate=24000):
    """Apply bandpass filter to audio segment."""
    layer = TemporalAudioPerturbationLayer(sample_rate=sample_rate)
    return layer.apply_bandpass_segment(audio_segment, low_coef, high_coef)


def calculate_stoi_metric(original_audio, perturbed_audio, sample_rate=24000):
    """Calculate STOI metric between original and perturbed audio."""
    # Convert to numpy and ensure mono
    if isinstance(original_audio, torch.Tensor):
        original_audio = original_audio.cpu().numpy()
    if isinstance(perturbed_audio, torch.Tensor):
        perturbed_audio = perturbed_audio.cpu().numpy()

    # Convert to mono if needed
    if original_audio.ndim == 2:
        original_audio = original_audio.mean(axis=0)
    if perturbed_audio.ndim == 2:
        perturbed_audio = perturbed_audio.mean(axis=0)

    # Ensure same length
    min_length = min(len(original_audio), len(perturbed_audio))
    original_audio = original_audio[:min_length]
    perturbed_audio = perturbed_audio[:min_length]

    # Calculate STOI
    stoi_score = stoi(original_audio, perturbed_audio, sample_rate, extended=False)
    return stoi_score


def main():
    parser = argparse.ArgumentParser(description='Test single audio perturbation')

    # Dataset arguments
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to use')
    parser.add_argument('--split', type=str, default='test.clean', help='Dataset split to use')
    parser.add_argument('--cache-dir', type=str, default='./data', help='Data cache directory')
    parser.add_argument('--audio-length', type=int, default=96000, help='Fixed audio length in samples')
    parser.add_argument('--sample-rate', type=int, default=24000, help='Audio sample rate')

    # Perturbation selection
    parser.add_argument('--perturbation', type=str, required=True,
                        choices=['pitch', 'formant', 'echo', 'reversal', 'bandpass'],
                        help='Type of perturbation to apply')

    # Perturbation parameters
    parser.add_argument('--pitch-shift', type=float, default=0.0,
                        help='Pitch shift in semitones (e.g., -6 to 6)')
    parser.add_argument('--formant-ratio', type=float, default=1.0,
                        help='Formant ratio (e.g., 0.9 to 1.1)')
    parser.add_argument('--echo-delay', type=float, default=0.1,
                        help='Echo delay in seconds (e.g., 0.01 to 0.3)')
    parser.add_argument('--echo-decay', type=float, default=0.3,
                        help='Echo decay factor (e.g., 0.0 to 0.5)')
    parser.add_argument('--reversal-mix', type=float, default=0.3,
                        help='Reversal mix ratio (e.g., 0.0 to 0.5)')
    parser.add_argument('--bandpass-low-coef', type=float, default=0.0,
                        help='Low frequency cutoff coefficient 0-1 (cuts below 400*coef Hz)')
    parser.add_argument('--bandpass-high-coef', type=float, default=0.0,
                        help='High frequency cutoff coefficient 0-1 (cuts above 16000-10000*coef Hz)')

    # ASR and output
    parser.add_argument('--asr-model', type=str, default='openai/whisper-medium',
                        help='ASR model for evaluation')
    parser.add_argument('--output-dir', type=str, default='./output/single_perturbation',
                        help='Directory to save perturbed audio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"\n=== Loading LibriSpeech {args.split} dataset ===")
    dataset = LibriSpeechDataset(
        split=args.split,
        target_sr=args.sample_rate,
        target_length=args.audio_length,
        cache_dir=args.cache_dir
    )

    # Load a single audio sample
    print(f"\nLoading sample {args.sample_idx}...")
    sample = dataset[args.sample_idx]
    audio = sample['audio']  # [channels, time]
    text = sample['text']
    sample_id = sample['id']

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Ground truth text: {text}")
    print(f"Sample ID: {sample_id}")

    # Initialize ASR evaluator
    print("\n=== Initializing ASR Evaluator (Whisper) ===")
    asr_evaluator = ASREvaluator(model_name=args.asr_model, device=args.device)

    # Evaluate original audio
    print("\n=== Evaluating ORIGINAL audio ===")
    original_transcription = asr_evaluator.transcribe(audio, sample_rate=args.sample_rate)
    original_wer = asr_evaluator.calculate_wer(text, original_transcription)
    original_cer = asr_evaluator.calculate_cer(text, original_transcription)

    print(f"Original transcription: {original_transcription}")
    print(f"Original WER: {original_wer:.4f} ({original_wer*100:.2f}%)")
    print(f"Original CER: {original_cer:.4f} ({original_cer*100:.2f}%)")

    # Apply perturbation
    print(f"\n=== Applying {args.perturbation.upper()} perturbation ===")

    perturbed_audio = audio.clone()

    # Apply perturbation to each channel
    for c in range(perturbed_audio.shape[0]):
        audio_channel = perturbed_audio[c].to(args.device)

        if args.perturbation == 'pitch':
            print(f"Pitch shift: {args.pitch_shift} semitones")
            perturbed_channel = apply_pitch_shift(audio_channel, args.pitch_shift, args.sample_rate)

        elif args.perturbation == 'formant':
            print(f"Formant ratio: {args.formant_ratio}")
            perturbed_channel = apply_formant_shift(audio_channel, args.formant_ratio, args.sample_rate)

        elif args.perturbation == 'echo':
            print(f"Echo delay: {args.echo_delay}s, decay: {args.echo_decay}")
            perturbed_channel = apply_echo(audio_channel, args.echo_delay, args.echo_decay, args.sample_rate)

        elif args.perturbation == 'reversal':
            print(f"Reversal mix: {args.reversal_mix}")
            perturbed_channel = apply_reversal(audio_channel, args.reversal_mix, args.sample_rate)

        elif args.perturbation == 'bandpass':
            print(f"Bandpass low_coef: {args.bandpass_low_coef}, high_coef: {args.bandpass_high_coef}")
            low_freq = 400.0 * args.bandpass_low_coef
            high_freq = 16000.0 - 10000.0 * args.bandpass_high_coef
            print(f"  -> Cuts below {low_freq:.1f} Hz, above {high_freq:.1f} Hz")
            perturbed_channel = apply_bandpass(audio_channel, args.bandpass_low_coef,
                                               args.bandpass_high_coef, args.sample_rate)

        perturbed_audio[c] = perturbed_channel.cpu()

    # Calculate STOI metric
    print("\n=== Calculating Audio Quality Metrics ===")
    stoi_score = calculate_stoi_metric(audio, perturbed_audio, args.sample_rate)
    print(f"STOI: {stoi_score:.4f} (1.0 = perfect intelligibility)")

    # Save audio files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n=== Saving Audio Files ===")

    # Save original
    original_filename = f"original_{sample_id}_{timestamp}.wav"
    original_path = os.path.join(args.output_dir, original_filename)
    torchaudio.save(original_path, audio.cpu(), args.sample_rate)
    print(f"Original: {original_path}")

    # Save perturbed
    # Create descriptive filename based on perturbation type
    if args.perturbation == 'pitch':
        perturb_desc = f"pitch{args.pitch_shift:+.1f}st"
    elif args.perturbation == 'formant':
        perturb_desc = f"formant{args.formant_ratio:.2f}"
    elif args.perturbation == 'echo':
        perturb_desc = f"echo_{args.echo_delay:.2f}s_{args.echo_decay:.2f}"
    elif args.perturbation == 'reversal':
        perturb_desc = f"reversal{args.reversal_mix:.2f}"
    elif args.perturbation == 'bandpass':
        perturb_desc = f"bandpass_l{args.bandpass_low_coef:.2f}_h{args.bandpass_high_coef:.2f}"

    perturbed_filename = f"perturbed_{sample_id}_{perturb_desc}_{timestamp}.wav"
    perturbed_path = os.path.join(args.output_dir, perturbed_filename)
    torchaudio.save(perturbed_path, perturbed_audio.cpu(), args.sample_rate)
    print(f"Perturbed: {perturbed_path}")

    # Evaluate perturbed audio
    print("\n=== Evaluating PERTURBED audio ===")
    perturbed_transcription = asr_evaluator.transcribe(perturbed_audio, sample_rate=args.sample_rate)
    perturbed_wer = asr_evaluator.calculate_wer(text, perturbed_transcription)
    perturbed_cer = asr_evaluator.calculate_cer(text, perturbed_transcription)

    print(f"Perturbed transcription: {perturbed_transcription}")
    print(f"Perturbed WER: {perturbed_wer:.4f} ({perturbed_wer*100:.2f}%)")
    print(f"Perturbed CER: {perturbed_cer:.4f} ({perturbed_cer*100:.2f}%)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Sample ID: {sample_id}")
    print(f"Perturbation: {args.perturbation.upper()}")

    if args.perturbation == 'pitch':
        print(f"  Parameters: pitch_shift={args.pitch_shift} semitones")
    elif args.perturbation == 'formant':
        print(f"  Parameters: formant_ratio={args.formant_ratio}")
    elif args.perturbation == 'echo':
        print(f"  Parameters: delay={args.echo_delay}s, decay={args.echo_decay}")
    elif args.perturbation == 'reversal':
        print(f"  Parameters: mix={args.reversal_mix}")
    elif args.perturbation == 'bandpass':
        low_freq = 400.0 * args.bandpass_low_coef
        high_freq = 16000.0 - 10000.0 * args.bandpass_high_coef
        print(f"  Parameters: low_coef={args.bandpass_low_coef}, high_coef={args.bandpass_high_coef}")
        print(f"  Frequency range: {low_freq:.1f} Hz - {high_freq:.1f} Hz")

    print(f"\nGround truth: {text}")
    print(f"\nOriginal Audio:")
    print(f"  Transcription: {original_transcription}")
    print(f"  WER: {original_wer:.4f} ({original_wer*100:.2f}%)")
    print(f"  CER: {original_cer:.4f} ({original_cer*100:.2f}%)")

    print(f"\nPerturbed Audio:")
    print(f"  Transcription: {perturbed_transcription}")
    print(f"  WER: {perturbed_wer:.4f} ({perturbed_wer*100:.2f}%)")
    print(f"  CER: {perturbed_cer:.4f} ({perturbed_cer*100:.2f}%)")
    print(f"  STOI: {stoi_score:.4f}")

    print(f"\nChanges:")
    print(f"  WER increase: {perturbed_wer - original_wer:+.4f} ({(perturbed_wer - original_wer)*100:+.2f}%)")
    print(f"  CER increase: {perturbed_cer - original_cer:+.4f} ({(perturbed_cer - original_cer)*100:+.2f}%)")

    print(f"\nOutput files:")
    print(f"  Original: {original_path}")
    print(f"  Perturbed: {perturbed_path}")
    print("="*80)

    # Save summary to JSON
    summary_path = os.path.join(args.output_dir, f"summary_{sample_id}_{perturb_desc}_{timestamp}.json")
    import json

    # Build perturbation parameters dict
    perturbation_params = {}
    if args.perturbation == 'pitch':
        perturbation_params = {'pitch_shift': args.pitch_shift}
    elif args.perturbation == 'formant':
        perturbation_params = {'formant_ratio': args.formant_ratio}
    elif args.perturbation == 'echo':
        perturbation_params = {'echo_delay': args.echo_delay, 'echo_decay': args.echo_decay}
    elif args.perturbation == 'reversal':
        perturbation_params = {'reversal_mix': args.reversal_mix}
    elif args.perturbation == 'bandpass':
        perturbation_params = {
            'bandpass_low_coef': args.bandpass_low_coef,
            'bandpass_high_coef': args.bandpass_high_coef,
            'low_freq_hz': 400.0 * args.bandpass_low_coef,
            'high_freq_hz': 16000.0 - 10000.0 * args.bandpass_high_coef
        }

    summary = {
        'config': {
            'dataset_split': args.split,
            'asr_model': args.asr_model,
            'sample_rate': args.sample_rate,
            'audio_length': args.audio_length,
            'perturbation_type': args.perturbation,
            'perturbation_params': perturbation_params
        },
        'sample_id': sample_id,
        'ground_truth': text,
        'original': {
            'transcription': original_transcription,
            'wer': float(original_wer),
            'cer': float(original_cer),
            'audio_file': original_path
        },
        'perturbed': {
            'transcription': perturbed_transcription,
            'wer': float(perturbed_wer),
            'cer': float(perturbed_cer),
            'stoi': float(stoi_score),
            'audio_file': perturbed_path
        },
        'changes': {
            'wer_increase': float(perturbed_wer - original_wer),
            'cer_increase': float(perturbed_cer - original_cer)
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
