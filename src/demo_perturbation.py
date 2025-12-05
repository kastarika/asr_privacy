"""
Demo script to load audio from test.clean, apply perturbations, and evaluate ASR.
Uses the same dataloader and settings as the training script.
"""
import os
import time
from datetime import datetime
import torch
import torchaudio
import argparse

from dataset import LibriSpeechDataset
from asr_evaluation import ASREvaluator
from audio_perturbations_temporal import TemporalAudioPerturbationLayer
from audio_perturbations import AudioPerturbationLayer
from encodec_utils import EnCodecProcessor


def main():
    parser = argparse.ArgumentParser(description='Demo audio perturbation and ASR evaluation')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to use')
    parser.add_argument('--model-path', type=str, default=None, help='Path to trained model checkpoint (optional)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save perturbed audio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-rate', type=int, default=24000, help='Audio sample rate')
    parser.add_argument('--audio-length', type=int, default=96000, help='Fixed audio length in samples (4 sec at 24kHz)')
    parser.add_argument('--split', type=str, default='test.clean', help='Dataset split to use')
    parser.add_argument('--cache-dir', type=str, default='./data', help='Data cache directory')
    parser.add_argument('--asr-model', type=str, default='openai/whisper-medium', help='ASR model for evaluation')
    parser.add_argument('--segment-length-ms', type=int, default=25, help='Segment length in ms (from training)')
    parser.add_argument('--ema-alpha', type=float, default=0.3, help='EMA alpha (from training)')
    parser.add_argument('--strength', type=float, default=1.0, help='Perturbation strength multiplier (0.0-2.0)')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset using the same method as training
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

    # Initialize EnCodec (needed for proper model structure)
    print("\n=== Loading EnCodec ===")
    encodec = EnCodecProcessor(device=args.device)

    # Initialize perturbation model with same settings as training
    print("\n=== Initializing Perturbation Model ===")
    print(f"Segment length: {args.segment_length_ms}ms")
    print(f"EMA alpha: {args.ema_alpha}")
    perturbation_layer = TemporalAudioPerturbationLayer(
        sample_rate=args.sample_rate,
        segment_length_ms=args.segment_length_ms,
        ema_alpha=args.ema_alpha
    ).to(args.device)
    

    # Load trained weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)

        # Extract perturbation layer state dict from model
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Extract only perturbation_layer weights
            perturbation_state = {
                k.replace('perturbation_layer.', ''): v
                for k, v in model_state.items()
                if k.startswith('perturbation_layer.')
            }
            perturbation_layer.load_state_dict(perturbation_state)
            print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        elif 'perturbation_layer' in checkpoint:
            perturbation_layer.load_state_dict(checkpoint['perturbation_layer'])
            print("Model loaded successfully")
        else:
            perturbation_layer.load_state_dict(checkpoint)
            print("Model loaded successfully")
    else:
        print("No trained model provided - using randomly initialized perturbations")
        print("(Results will be random/minimal - train a model for better perturbations)")

    # Apply perturbations
    print("\n=== Applying Perturbations ===")
    perturbation_layer.eval()

    # Calculate audio duration for real-time metrics
    audio_duration_seconds = args.audio_length / args.sample_rate
    print(f"Audio duration: {audio_duration_seconds:.2f} seconds ({audio_duration_seconds*1000:.0f} ms)")

    # Start timing the processing
    start_time = time.time()

    with torch.no_grad():
        # Apply perturbations (returns perturbed audio, raw params, smoothed params)
        perturbed_audio_before_codec, raw_params, smoothed_params = perturbation_layer(audio_batch)

        # Print perturbation statistics
        print("\nPerturbation statistics:")
        for param_name, param_values in smoothed_params.items():
            mean_val = param_values.mean().item()
            std_val = param_values.std().item()
            min_val = param_values.min().item()
            max_val = param_values.max().item()
            print(f"  {param_name:20s}: mean={mean_val:8.4f}, std={std_val:8.4f}, min={min_val:8.4f}, max={max_val:8.4f}")

        # Apply EnCodec encode-decode (same as training pipeline)
        print("\n=== Applying EnCodec encode-decode ===")
        perturbed_audio = encodec.encode_decode(perturbed_audio_before_codec)

    # End timing and calculate metrics
    end_time = time.time()
    processing_time = end_time - start_time

    # Calculate real-time factor and delay
    real_time_factor = processing_time / audio_duration_seconds
    real_time_delay = processing_time - audio_duration_seconds

    print(f"\n=== Processing Time Metrics ===")
    print(f"Total processing time: {processing_time:.4f} seconds ({processing_time*1000:.2f} ms)")
    print(f"Audio duration: {audio_duration_seconds:.4f} seconds ({audio_duration_seconds*1000:.2f} ms)")
    print(f"Real-time factor (RTF): {real_time_factor:.4f}x")
    if real_time_delay > 0:
        print(f"Real-time delay: {real_time_delay:.4f} seconds ({real_time_delay*1000:.2f} ms)")
        print(f"  → Processing is {real_time_factor:.2f}x slower than real-time")
        print(f"  → Would accumulate {real_time_delay:.2f}s delay if audio played in real-time")
    else:
        print(f"Real-time delay: 0 seconds (processing faster than real-time)")
        print(f"  → Processing is {1/real_time_factor:.2f}x faster than real-time")
        print(f"  → No delay would occur if audio played in real-time")

    # Move back to CPU for saving
    perturbed_audio_cpu = perturbed_audio.cpu()
    perturbed_audio_before_codec_cpu = perturbed_audio_before_codec.cpu()

    # Save perturbed audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perturbation_name = "temporal_multi"  # Multiple perturbations applied
    strength_str = f"strength{args.strength:.2f}".replace('.', 'p')

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
    before_codec_filename = f"perturbed_before_codec_{sample_id}_{perturbation_name}_{strength_str}_{timestamp}.wav"
    before_codec_path = os.path.join(args.output_dir, before_codec_filename)
    torchaudio.save(
        before_codec_path,
        perturbed_audio_before_codec_cpu[0],
        args.sample_rate
    )
    print(f"Perturbed (before EnCodec): {before_codec_path}")

    # Save perturbed audio (after EnCodec)
    output_filename = f"perturbed_after_codec_{sample_id}_{perturbation_name}_{strength_str}_{timestamp}.wav"
    output_path = os.path.join(args.output_dir, output_filename)
    torchaudio.save(
        output_path,
        perturbed_audio_cpu[0],  # Remove batch dimension [channels, time]
        args.sample_rate
    )
    print(f"Perturbed (after EnCodec): {output_path}")

    # Evaluate perturbed audio
    print("\n=== Evaluating PERTURBED audio ===")
    perturbed_transcription = asr_evaluator.transcribe(
        perturbed_audio_cpu[0],
        sample_rate=args.sample_rate
    )
    perturbed_wer = asr_evaluator.calculate_wer(original_transcription, perturbed_transcription)
    perturbed_cer = asr_evaluator.calculate_cer(original_transcription, perturbed_transcription)

    print(f"Perturbed transcription: {perturbed_transcription}")
    print(f"Perturbed WER: {perturbed_wer:.4f}")
    print(f"Perturbed CER: {perturbed_cer:.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset split: {args.split}")
    print(f"  ASR model: {args.asr_model}")
    print(f"  Segment length: {args.segment_length_ms}ms")
    print(f"  EMA alpha: {args.ema_alpha}")
    print(f"  Model checkpoint: {args.model_path if args.model_path else 'Random initialization'}")
    print(f"\nSample ID: {sample_id}")
    print(f"Ground truth: {text}")
    print(f"\nProcessing Performance:")
    print(f"  Audio duration: {audio_duration_seconds:.4f}s")
    print(f"  Processing time: {processing_time:.4f}s")
    print(f"  Real-time factor: {real_time_factor:.4f}x")
    print(f"  Real-time delay: {max(0, real_time_delay):.4f}s")
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
            'segment_length_ms': args.segment_length_ms,
            'ema_alpha': args.ema_alpha,
            'audio_length': args.audio_length,
            'sample_rate': args.sample_rate,
            'model_checkpoint': args.model_path if args.model_path else 'random_init'
        },
        'sample_id': sample_id,
        'ground_truth': text,
        'processing_performance': {
            'audio_duration_seconds': float(audio_duration_seconds),
            'processing_time_seconds': float(processing_time),
            'real_time_factor': float(real_time_factor),
            'real_time_delay_seconds': float(max(0, real_time_delay)),
            'faster_than_realtime': real_time_factor < 1.0
        },
        'original': {
            'transcription': original_transcription,
            'wer': original_wer,
            'cer': original_cer,
            'audio_file': original_output_path
        },
        'perturbed_before_codec': {
            'audio_file': before_codec_path
        },
        'perturbed_after_codec': {
            'transcription': perturbed_transcription,
            'wer': perturbed_wer,
            'cer': perturbed_cer,
            'audio_file': output_path,
            'perturbation_name': perturbation_name,
            'strength': args.strength
        },
        'changes': {
            'wer_increase': float(perturbed_wer - original_wer),
            'cer_increase': float(perturbed_cer - original_cer)
        },
        'perturbation_stats': {
            param_name: {
                'mean': float(param_values.mean().item()),
                'std': float(param_values.std().item()),
                'min': float(param_values.min().item()),
                'max': float(param_values.max().item())
            }
            for param_name, param_values in smoothed_params.items()
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
