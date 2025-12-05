"""
Demo script for static (non-temporal) audio perturbations.
Uses AudioPerturbationLayer with learnable but constant parameters.
Compared to temporal model, this applies the same perturbation across entire audio.
"""
import os
import sys
from datetime import datetime
import torch
import torchaudio
import argparse

# Add src-asr-1 to path to import AudioPerturbationLayer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src-asr-1'))

from dataset import LibriSpeechDataset
from asr_evaluation import ASREvaluator
from audio_perturbations import AudioPerturbationLayer
from encodec_utils import EnCodecProcessor


def main():
    parser = argparse.ArgumentParser(description='Demo static audio perturbation and ASR evaluation')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to use')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model checkpoint (optional)')
    parser.add_argument('--output-dir', type=str, default='./output/static_perturbation', help='Directory to save perturbed audio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-rate', type=int, default=24000, help='Audio sample rate')
    parser.add_argument('--audio-length', type=int, default=96000, help='Fixed audio length in samples (4 sec at 24kHz)')
    parser.add_argument('--split', type=str, default='test.clean', help='Dataset split to use')
    parser.add_argument('--cache-dir', type=str, default='./data', help='Data cache directory')
    parser.add_argument('--asr-model', type=str, default='openai/whisper-medium', help='ASR model for evaluation')

    # Manual parameter setting (if no model provided)
    parser.add_argument('--pitch-shift', type=float, default=None, help='Manual pitch shift (semitones)')
    parser.add_argument('--formant-ratio', type=float, default=None, help='Manual formant ratio')
    parser.add_argument('--echo-delay', type=float, default=None, help='Manual echo delay (seconds)')
    parser.add_argument('--echo-decay', type=float, default=None, help='Manual echo decay')
    parser.add_argument('--reversal-mix', type=float, default=None, help='Manual reversal mix')
    parser.add_argument('--bp-center-freq', type=float, default=None, help='Manual bandpass center freq (Hz)')
    parser.add_argument('--bp-bandwidth', type=float, default=None, help='Manual bandpass bandwidth (Hz)')
    parser.add_argument('--bp-mix', type=float, default=None, help='Manual bandpass mix')

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

    # Add batch dimension
    audio_batch = audio.unsqueeze(0).to(args.device)  # [1, channels, time]

    # Initialize ASR evaluator
    print("\n=== Initializing ASR Evaluator (Whisper) ===")
    asr_evaluator = ASREvaluator(model_name=args.asr_model, device=args.device)

    # Evaluate original audio
    print("\n=== Evaluating ORIGINAL audio ===")
    original_transcription = asr_evaluator.transcribe(audio, sample_rate=args.sample_rate)
    original_wer = asr_evaluator.calculate_wer(text, original_transcription)
    original_cer = asr_evaluator.calculate_cer(text, original_transcription)

    print(f"Original transcription: {original_transcription}")
    print(f"Original WER: {original_wer:.4f}")
    print(f"Original CER: {original_cer:.4f}")

    # Initialize EnCodec
    print("\n=== Loading EnCodec ===")
    encodec = EnCodecProcessor(device=args.device)

    # Initialize static perturbation model
    print("\n=== Initializing Static Perturbation Model ===")
    perturbation_layer = AudioPerturbationLayer(sample_rate=args.sample_rate).to(args.device)

    # Load trained weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            # Try to extract perturbation_layer weights
            model_state = checkpoint['model_state_dict']
            perturbation_state = {
                k.replace('perturbation_layer.', ''): v
                for k, v in model_state.items()
                if k.startswith('perturbation_layer.')
            }
            if perturbation_state:
                perturbation_layer.load_state_dict(perturbation_state)
                print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                print("Warning: No perturbation_layer found in checkpoint")
        elif 'perturbation_layer' in checkpoint:
            perturbation_layer.load_state_dict(checkpoint['perturbation_layer'])
            print("Model loaded successfully")
        else:
            perturbation_layer.load_state_dict(checkpoint)
            print("Model loaded successfully")
    else:
        print("No trained model provided")

        # Apply manual parameters if provided
        if args.pitch_shift is not None:
            perturbation_layer.pitch_shift_semitones.data = torch.tensor([args.pitch_shift]).to(args.device)
            print(f"Set pitch shift to {args.pitch_shift} semitones")
        if args.formant_ratio is not None:
            perturbation_layer.formant_shift_ratio.data = torch.tensor([args.formant_ratio]).to(args.device)
            print(f"Set formant ratio to {args.formant_ratio}")
        if args.echo_delay is not None:
            perturbation_layer.echo_delay.data = torch.tensor([args.echo_delay]).to(args.device)
            print(f"Set echo delay to {args.echo_delay}s")
        if args.echo_decay is not None:
            perturbation_layer.echo_decay.data = torch.tensor([args.echo_decay]).to(args.device)
            print(f"Set echo decay to {args.echo_decay}")
        if args.reversal_mix is not None:
            perturbation_layer.reversal_mix.data = torch.tensor([args.reversal_mix]).to(args.device)
            print(f"Set reversal mix to {args.reversal_mix}")
        if args.bp_center_freq is not None:
            perturbation_layer.bp_center_freq.data = torch.tensor([args.bp_center_freq]).to(args.device)
            print(f"Set bandpass center freq to {args.bp_center_freq} Hz")
        if args.bp_bandwidth is not None:
            perturbation_layer.bp_bandwidth.data = torch.tensor([args.bp_bandwidth]).to(args.device)
            print(f"Set bandpass bandwidth to {args.bp_bandwidth} Hz")
        if args.bp_mix is not None:
            perturbation_layer.bp_mix.data = torch.tensor([args.bp_mix]).to(args.device)
            print(f"Set bandpass mix to {args.bp_mix}")

    # Display current parameters
    print("\n=== Current Perturbation Parameters ===")
    params = perturbation_layer.get_perturbation_params()
    for param_name, param_value in params.items():
        print(f"  {param_name:25s}: {param_value:8.4f}")

    # Apply perturbations
    print("\n=== Applying Static Perturbations ===")
    perturbation_layer.eval()

    with torch.no_grad():
        # Apply perturbations
        perturbed_audio_before_codec = perturbation_layer(audio_batch)

        # Apply EnCodec encode-decode (same as training pipeline)
        print("\n=== Applying EnCodec encode-decode ===")
        perturbed_audio = encodec.encode_decode(perturbed_audio_before_codec)

    # Move back to CPU for saving
    perturbed_audio_cpu = perturbed_audio.cpu()
    perturbed_audio_before_codec_cpu = perturbed_audio_before_codec.cpu()

    # Save audio files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perturbation_name = "static_multi"

    print(f"\n=== Saving Audio Files ===")

    # Save original audio
    original_output_filename = f"original_{sample_id}_{timestamp}.wav"
    original_output_path = os.path.join(args.output_dir, original_output_filename)
    torchaudio.save(
        original_output_path,
        audio.cpu(),
        args.sample_rate
    )
    print(f"Original: {original_output_path}")

    # Save perturbed audio (before EnCodec)
    before_codec_filename = f"perturbed_before_codec_{sample_id}_{perturbation_name}_{timestamp}.wav"
    before_codec_path = os.path.join(args.output_dir, before_codec_filename)
    torchaudio.save(
        before_codec_path,
        perturbed_audio_before_codec_cpu[0],
        args.sample_rate
    )
    print(f"Perturbed (before EnCodec): {before_codec_path}")

    # Save perturbed audio (after EnCodec)
    output_filename = f"perturbed_after_codec_{sample_id}_{perturbation_name}_{timestamp}.wav"
    output_path = os.path.join(args.output_dir, output_filename)
    torchaudio.save(
        output_path,
        perturbed_audio_cpu[0],
        args.sample_rate
    )
    print(f"Perturbed (after EnCodec): {output_path}")

    # Evaluate perturbed audio
    print("\n=== Evaluating PERTURBED audio ===")
    perturbed_transcription = asr_evaluator.transcribe(
        perturbed_audio_cpu[0],
        sample_rate=args.sample_rate
    )
    perturbed_wer = asr_evaluator.calculate_wer(text, perturbed_transcription)
    perturbed_cer = asr_evaluator.calculate_cer(text, perturbed_transcription)

    print(f"Perturbed transcription: {perturbed_transcription}")
    print(f"Perturbed WER: {perturbed_wer:.4f}")
    print(f"Perturbed CER: {perturbed_cer:.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - STATIC PERTURBATION MODEL")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset split: {args.split}")
    print(f"  ASR model: {args.asr_model}")
    print(f"  Model type: Static (constant parameters)")
    print(f"  Model checkpoint: {args.model_path if args.model_path else 'Manual/Random'}")

    print(f"\nPerturbation Parameters (applied uniformly):")
    for param_name, param_value in params.items():
        print(f"  {param_name:25s}: {param_value:8.4f}")

    print(f"\nSample ID: {sample_id}")
    print(f"Ground truth: {text}")
    print(f"\nOriginal:")
    print(f"  Transcription: {original_transcription}")
    print(f"  WER: {original_wer:.4f} ({original_wer*100:.2f}%)")
    print(f"  CER: {original_cer:.4f} ({original_cer*100:.2f}%)")
    print(f"\nPerturbed (after EnCodec):")
    print(f"  Transcription: {perturbed_transcription}")
    print(f"  WER: {perturbed_wer:.4f} ({perturbed_wer*100:.2f}%)")
    print(f"  CER: {perturbed_cer:.4f} ({perturbed_cer*100:.2f}%)")
    print(f"\nChange:")
    print(f"  WER increase: {perturbed_wer - original_wer:.4f} ({(perturbed_wer - original_wer)*100:.2f}%)")
    print(f"  CER increase: {perturbed_cer - original_cer:.4f} ({(perturbed_cer - original_cer)*100:.2f}%)")
    print(f"\nOutput files:")
    print(f"  Original: {original_output_path}")
    print(f"  Perturbed (before EnCodec): {before_codec_path}")
    print(f"  Perturbed (after EnCodec): {output_path}")
    print("="*80)

    # Save summary to JSON
    summary_path = os.path.join(args.output_dir, f"summary_{sample_id}_{timestamp}.json")
    import json
    summary = {
        'config': {
            'dataset_split': args.split,
            'asr_model': args.asr_model,
            'model_type': 'static',
            'audio_length': args.audio_length,
            'sample_rate': args.sample_rate,
            'model_checkpoint': args.model_path if args.model_path else 'manual_random'
        },
        'perturbation_params': params,
        'sample_id': sample_id,
        'ground_truth': text,
        'original': {
            'transcription': original_transcription,
            'wer': float(original_wer),
            'cer': float(original_cer),
            'audio_file': original_output_path
        },
        'perturbed_before_codec': {
            'audio_file': before_codec_path
        },
        'perturbed_after_codec': {
            'transcription': perturbed_transcription,
            'wer': float(perturbed_wer),
            'cer': float(perturbed_cer),
            'audio_file': output_path,
            'perturbation_name': perturbation_name
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
