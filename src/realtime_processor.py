"""
Real-time audio processing for adversarial perturbations.
Processes audio streams with minimal latency for online conversations.
"""
import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import time
from collections import deque

from encodec_utils import EnCodecProcessor
from audio_perturbations import AudioPerturbationLayer
from latent_perturbations import AdversarialLatentModel


class RealtimeAudioProcessor:
    """
    Real-time audio processor for adversarial perturbations.
    Processes audio in chunks with overlap-add for smooth output.
    """

    def __init__(self, model, device='cuda', chunk_duration=0.5,
                 sample_rate=24000, overlap_ratio=0.5):
        """
        Initialize real-time processor.

        Args:
            model: Trained perturbation model
            device: Device to run on
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate
            overlap_ratio: Overlap between chunks (0 to 1)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = sample_rate

        # Chunk parameters
        self.chunk_size = int(chunk_duration * sample_rate)
        self.hop_size = int(self.chunk_size * (1 - overlap_ratio))

        # Buffers
        self.input_buffer = deque(maxlen=self.chunk_size * 2)
        self.output_buffer = deque(maxlen=self.chunk_size * 2)

        # Threading
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False

        # Statistics
        self.processing_times = []
        self.latency_ms = 0.0

    def process_chunk(self, audio_chunk):
        """
        Process a single audio chunk through the model.

        Args:
            audio_chunk: Audio tensor [channels, samples]

        Returns:
            Perturbed audio chunk
        """
        with torch.no_grad():
            # Add batch dimension
            audio_batch = audio_chunk.unsqueeze(0).to(self.device)

            # Process through model
            start_time = time.time()
            perturbed = self.model(audio_batch)
            processing_time = (time.time() - start_time) * 1000  # ms

            # Track latency
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            self.latency_ms = np.mean(self.processing_times)

            # Remove batch dimension
            perturbed = perturbed.squeeze(0).cpu()

        return perturbed

    def processing_thread(self):
        """Background thread for processing audio chunks."""
        while self.running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.1)

                # Ensure correct shape [channels, samples]
                if audio_chunk.dim() == 1:
                    audio_chunk = audio_chunk.unsqueeze(0)

                # Pad if needed
                if audio_chunk.shape[1] < self.chunk_size:
                    padding = self.chunk_size - audio_chunk.shape[1]
                    audio_chunk = torch.nn.functional.pad(audio_chunk, (0, padding))
                elif audio_chunk.shape[1] > self.chunk_size:
                    audio_chunk = audio_chunk[:, :self.chunk_size]

                # Process chunk
                perturbed_chunk = self.process_chunk(audio_chunk)

                # Put in output queue
                self.output_queue.put(perturbed_chunk)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """
        Callback for sounddevice stream.
        Called for each audio block.

        Args:
            indata: Input audio data
            outdata: Output audio data buffer
            frames: Number of frames
            time_info: Timing information
            status: Status flags
        """
        if status:
            print(f"Status: {status}")

        # Convert input to torch tensor
        audio_in = torch.from_numpy(indata.copy()).float().T  # [channels, frames]

        # Add to input queue for processing
        try:
            self.input_queue.put_nowait(audio_in)
        except queue.Full:
            print("Warning: Input queue full, dropping frame")

        # Get processed audio from output queue
        try:
            audio_out = self.output_queue.get_nowait()

            # Convert to numpy and copy to output
            audio_out_np = audio_out.T.numpy()  # [frames, channels]

            # Ensure correct size
            if audio_out_np.shape[0] >= frames:
                outdata[:] = audio_out_np[:frames]
            else:
                outdata[:audio_out_np.shape[0]] = audio_out_np
                outdata[audio_out_np.shape[0]:] = 0

        except queue.Empty:
            # No processed audio available, output silence
            outdata[:] = 0

    def start_stream(self, input_device=None, output_device=None,
                     channels=1, blocksize=2048):
        """
        Start real-time audio stream.

        Args:
            input_device: Input audio device (None for default)
            output_device: Output audio device (None for default)
            channels: Number of audio channels
            blocksize: Audio block size
        """
        print(f"Starting real-time processor...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate:.3f}s)")
        print(f"Block size: {blocksize} samples")

        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self.processing_thread)
        self.process_thread.start()

        # Start audio stream
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=blocksize,
            device=(input_device, output_device),
            channels=channels,
            dtype='float32',
            callback=self.audio_callback
        )

        self.stream.start()
        print("Stream started! Press Ctrl+C to stop.")

        # Monitor latency
        try:
            while True:
                time.sleep(1)
                print(f"\rLatency: {self.latency_ms:.1f} ms | "
                      f"Input queue: {self.input_queue.qsize()} | "
                      f"Output queue: {self.output_queue.qsize()}",
                      end='', flush=True)
        except KeyboardInterrupt:
            print("\n\nStopping stream...")

        self.stop_stream()

    def stop_stream(self):
        """Stop real-time audio stream."""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        print("Stream stopped.")

    def process_file(self, input_file, output_file):
        """
        Process an audio file (non-realtime).

        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
        """
        import torchaudio
        from dataset import AudioChunker

        print(f"Processing file: {input_file}")

        # Load audio
        audio, sr = torchaudio.load(input_file)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Chunk audio
        chunker = AudioChunker(
            chunk_size=self.chunk_size,
            hop_size=self.hop_size,
            sample_rate=self.sample_rate
        )
        chunks = chunker.chunk_audio(audio)

        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = self.process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        # Reconstruct audio
        processed_audio = chunker.reconstruct_audio(
            processed_chunks,
            original_length=audio.shape[1]
        )

        # Save
        torchaudio.save(output_file, processed_audio, self.sample_rate)
        print(f"Saved to: {output_file}")


def load_trained_model(checkpoint_path, model_type='audio', device='cuda'):
    """
    Load a trained perturbation model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: 'audio' or 'latent'
        device: Device to load on

    Returns:
        Loaded model
    """
    from encodec_utils import EnCodecProcessor

    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize EnCodec
    encodec = EnCodecProcessor(device=device)

    # Initialize model based on type
    if model_type == 'audio':
        from train_audio_perturbations import AudioPerturbationModel
        model = AudioPerturbationModel(encodec, sample_rate=encodec.sample_rate)
    elif model_type == 'latent':
        from latent_perturbations import AdversarialLatentModel
        model = AdversarialLatentModel(encodec, perturbation_type='additive')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-time audio perturbation processor')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='audio',
                        choices=['audio', 'latent'],
                        help='Type of model')
    parser.add_argument('--mode', type=str, default='stream',
                        choices=['stream', 'file'],
                        help='Processing mode')
    parser.add_argument('--input_file', type=str,
                        help='Input audio file (for file mode)')
    parser.add_argument('--output_file', type=str,
                        help='Output audio file (for file mode)')
    parser.add_argument('--chunk_duration', type=float, default=0.5,
                        help='Chunk duration in seconds')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                        help='Overlap ratio between chunks')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load model
    model = load_trained_model(args.checkpoint, args.model_type, args.device)

    # Create processor
    processor = RealtimeAudioProcessor(
        model=model,
        device=args.device,
        chunk_duration=args.chunk_duration,
        overlap_ratio=args.overlap_ratio
    )

    # Run in selected mode
    if args.mode == 'stream':
        processor.start_stream()
    elif args.mode == 'file':
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file required for file mode")
        else:
            processor.process_file(args.input_file, args.output_file)
