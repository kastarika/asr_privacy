"""
Real-time audio perturbation demo.
Captures audio from microphone, applies perturbations, and plays back in real-time.
"""
import os
import argparse
import numpy as np
import torch
import sounddevice as sd
import queue
import threading
from datetime import datetime
import torchaudio

from audio_perturbations_temporal import TemporalAudioPerturbationLayer
from audio_perturbations import AudioPerturbationLayer
from encodec_utils import EnCodecProcessor


class RealTimeProcessor:
    """Real-time audio processor with perturbations."""

    def __init__(self, model, encodec, device, sample_rate=24000,
                 chunk_duration=1.0, apply_codec=False):
        """
        Initialize real-time processor.

        Args:
            model: Perturbation model (temporal or static)
            encodec: EnCodec processor (optional)
            device: torch device
            sample_rate: Audio sample rate
            chunk_duration: Duration of each processing chunk in seconds
            apply_codec: Whether to apply EnCodec (adds latency)
        """
        self.model = model
        self.encodec = encodec
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.apply_codec = apply_codec

        # Queues for audio chunks
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # Statistics
        self.total_chunks = 0
        self.total_latency = 0.0
        self.running = False

        # Set model to eval mode
        self.model.eval()

    def audio_callback(self, indata, outdata, frames, time, status):
        """
        Callback function for sounddevice stream.
        Called for each audio block.
        """
        if status:
            print(f"Audio callback status: {status}")

        # Put input audio in queue
        self.input_queue.put(indata.copy())

        # Get processed audio from queue (or zeros if not ready)
        try:
            processed = self.output_queue.get_nowait()
            outdata[:] = processed
        except queue.Empty:
            outdata[:] = np.zeros_like(outdata)

    def process_chunk(self, audio_chunk):
        """
        Process a single audio chunk through the model.

        Args:
            audio_chunk: numpy array [frames, channels]

        Returns:
            Processed audio chunk
        """
        import time
        start_time = time.time()

        # Convert to torch tensor [batch, channels, time]
        audio_tensor = torch.from_numpy(audio_chunk.T).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            # Apply perturbations
            if hasattr(self.model, 'perturbation_layer'):
                # Wrapped model (from training)
                perturbed, _, _ = self.model.perturbation_layer(audio_tensor)
            else:
                # Direct perturbation layer
                if isinstance(self.model, TemporalAudioPerturbationLayer):
                    perturbed, _, _ = self.model(audio_tensor)
                else:
                    # Static model
                    perturbed = self.model(audio_tensor)

            # Apply EnCodec if requested (adds latency)
            if self.apply_codec and self.encodec is not None:
                perturbed = self.encodec.encode_decode(perturbed)

        # Convert back to numpy [frames, channels]
        processed = perturbed.squeeze(0).T.cpu().numpy()

        # Update statistics
        latency = time.time() - start_time
        self.total_latency += latency
        self.total_chunks += 1

        return processed

    def processing_thread(self):
        """Thread that processes audio chunks."""
        print("Processing thread started")

        while self.running:
            try:
                # Get input chunk
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Process it
                processed_chunk = self.process_chunk(audio_chunk)

                # Put in output queue
                self.output_queue.put(processed_chunk)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()

    def start(self):
        """Start real-time processing."""
        print("\n" + "="*60)
        print("REAL-TIME AUDIO PERTURBATION")
        print("="*60)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_duration}s)")
        print(f"Apply EnCodec: {self.apply_codec}")
        print(f"Device: {self.device}")
        print("\nPress Ctrl+C to stop")
        print("="*60)

        self.running = True

        # Start processing thread
        self.thread = threading.Thread(target=self.processing_thread, daemon=True)
        self.thread.start()

        # Start audio stream
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=1,  # Mono
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback
            ):
                print("\n🎤 Recording... Speak into your microphone!")
                print("🔊 Playing back perturbed audio...\n")

                # Keep running until interrupted
                while self.running:
                    sd.sleep(1000)

                    # Print statistics
                    if self.total_chunks > 0:
                        avg_latency = self.total_latency / self.total_chunks
                        print(f"\rChunks: {self.total_chunks} | "
                              f"Avg latency: {avg_latency*1000:.1f}ms | "
                              f"Queue size: {self.input_queue.qsize()}", end='')

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop real-time processing."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total chunks processed: {self.total_chunks}")
        if self.total_chunks > 0:
            avg_latency = self.total_latency / self.total_chunks
            print(f"Average processing latency: {avg_latency*1000:.2f} ms")
            print(f"Theoretical max latency: {self.chunk_duration*1000:.2f} ms")
            print(f"Real-time factor: {self.chunk_duration / avg_latency:.2f}x")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Real-time audio perturbation demo')

    # Model parameters
    parser.add_argument('--model-type', type=str, choices=['temporal', 'static'],
                        default='temporal', help='Model type to use')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')

    # Audio parameters
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='Audio sample rate')
    parser.add_argument('--chunk-duration', type=float, default=1.0,
                        help='Duration of each processing chunk in seconds')
    parser.add_argument('--apply-codec', action='store_true',
                        help='Apply EnCodec (adds latency)')

    # Temporal model parameters
    parser.add_argument('--segment-length-ms', type=int, default=25,
                        help='Segment length in ms (temporal model only)')
    parser.add_argument('--ema-alpha', type=float, default=0.7,
                        help='EMA alpha (temporal model only)')

    # Static model parameters
    parser.add_argument('--pitch-shift', type=float, default=None,
                        help='Manual pitch shift (semitones, static model only)')
    parser.add_argument('--formant-ratio', type=float, default=None,
                        help='Manual formant ratio (static model only)')
    parser.add_argument('--echo-delay', type=float, default=None,
                        help='Manual echo delay (seconds, static model only)')
    parser.add_argument('--echo-decay', type=float, default=None,
                        help='Manual echo decay (static model only)')
    parser.add_argument('--reversal-mix', type=float, default=None,
                        help='Manual reversal mix (static model only)')

    # System parameters
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print(f"\n🚀 Initializing real-time audio perturbation demo...")
    print(f"Device: {args.device}")
    print(f"Model type: {args.model_type}")

    # Initialize EnCodec if needed
    encodec = None
    if args.apply_codec:
        print("Loading EnCodec...")
        encodec = EnCodecProcessor(device=args.device)

    # Initialize model
    print(f"Initializing {args.model_type} perturbation model...")

    if args.model_type == 'temporal':
        # Temporal LSTM model
        model = TemporalAudioPerturbationLayer(
            sample_rate=args.sample_rate,
            segment_length_ms=args.segment_length_ms,
            ema_alpha=args.ema_alpha
        ).to(args.device)

        # Load trained weights if provided
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading trained model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=args.device,
                                    weights_only=False)

            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                perturbation_state = {
                    k.replace('perturbation_layer.', ''): v
                    for k, v in model_state.items()
                    if k.startswith('perturbation_layer.')
                }
                if perturbation_state:
                    model.load_state_dict(perturbation_state)
                    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Model loaded")
        else:
            print("⚠️  No model path provided - using random initialization")

    else:
        # Static model
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src-asr-1'))
        from audio_perturbations import AudioPerturbationLayer

        model = AudioPerturbationLayer(sample_rate=args.sample_rate).to(args.device)

        # Load or set manual parameters
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading trained model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=args.device,
                                    weights_only=False)

            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                perturbation_state = {
                    k.replace('perturbation_layer.', ''): v
                    for k, v in model_state.items()
                    if k.startswith('perturbation_layer.')
                }
                if perturbation_state:
                    model.load_state_dict(perturbation_state)
                    print("✅ Model loaded")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Model loaded")
        else:
            print("Setting manual parameters...")
            if args.pitch_shift is not None:
                model.pitch_shift_semitones.data = torch.tensor([args.pitch_shift]).to(args.device)
                print(f"  Pitch shift: {args.pitch_shift} semitones")
            if args.formant_ratio is not None:
                model.formant_shift_ratio.data = torch.tensor([args.formant_ratio]).to(args.device)
                print(f"  Formant ratio: {args.formant_ratio}")
            if args.echo_delay is not None:
                model.echo_delay.data = torch.tensor([args.echo_delay]).to(args.device)
                print(f"  Echo delay: {args.echo_delay}s")
            if args.echo_decay is not None:
                model.echo_decay.data = torch.tensor([args.echo_decay]).to(args.device)
                print(f"  Echo decay: {args.echo_decay}")
            if args.reversal_mix is not None:
                model.reversal_mix.data = torch.tensor([args.reversal_mix]).to(args.device)
                print(f"  Reversal mix: {args.reversal_mix}")

    # Create processor
    processor = RealTimeProcessor(
        model=model,
        encodec=encodec,
        device=args.device,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        apply_codec=args.apply_codec
    )

    # Start processing
    try:
        processor.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
