"""
ASR evaluation using Whisper for measuring adversarial attack effectiveness.
"""
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
from typing import List, Dict


class ASREvaluator:
    """Evaluator for measuring ASR performance on perturbed audio."""

    def __init__(self, model_name="openai/whisper-small", device="cuda"):
        """
        Initialize ASR evaluator with Whisper model.

        Args:
            model_name: Whisper model variant to use
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # Expected sample rate for Whisper
        self.target_sr = 16000

    def transcribe(self, audio, sample_rate=24000):
        """
        Transcribe audio using Whisper.

        Args:
            audio: Audio tensor [channels, time] or [time]
            sample_rate: Audio sample rate

        Returns:
            Transcription text
        """
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Resample if needed
        if sample_rate != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sr)

        # Process with Whisper processor
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )

        # Move to device
        input_features = inputs.input_features.to(self.device)

        # Generate transcription
        # Set language='en' for English transcription (LibriSpeech is English)
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="en",
                task="transcribe"
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()

    def transcribe_batch(self, audio_batch, sample_rate=24000):
        """
        Transcribe a batch of audio samples.

        Args:
            audio_batch: Batch of audio tensors [batch, channels, time]
            sample_rate: Audio sample rate

        Returns:
            List of transcription texts
        """
        transcriptions = []

        for i in range(audio_batch.shape[0]):
            audio = audio_batch[i]
            transcription = self.transcribe(audio, sample_rate)
            transcriptions.append(transcription)

        return transcriptions

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis.

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            WER as a float (0.0 to 1.0+)
        """
        # Normalize text
        reference = reference.lower().strip()
        hypothesis = hypothesis.lower().strip()

        # Calculate WER using jiwer
        try:
            wer = jiwer.wer(reference, hypothesis)
        except:
            # If one string is empty, return max error
            wer = 1.0

        return wer

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate between reference and hypothesis.

        Args:
            reference: Ground truth text
            hypothesis: Predicted text

        Returns:
            CER as a float (0.0 to 1.0+)
        """
        reference = reference.lower().strip()
        hypothesis = hypothesis.lower().strip()

        try:
            cer = jiwer.cer(reference, hypothesis)
        except:
            cer = 1.0

        return cer

    def evaluate_batch(self, audio_batch, reference_texts, sample_rate=24000):
        """
        Evaluate a batch of perturbed audio.

        Args:
            audio_batch: Batch of audio tensors [batch, channels, time]
            reference_texts: List of ground truth transcriptions
            sample_rate: Audio sample rate

        Returns:
            Dict with evaluation metrics
        """
        # Transcribe all samples
        hypotheses = self.transcribe_batch(audio_batch, sample_rate)

        # Calculate WER and CER for each sample
        wers = []
        cers = []

        for ref, hyp in zip(reference_texts, hypotheses):
            wer = self.calculate_wer(ref, hyp)
            cer = self.calculate_cer(ref, hyp)
            wers.append(wer)
            cers.append(cer)

        results = {
            'wer': np.mean(wers),
            'cer': np.mean(cers),
            'wer_std': np.std(wers),
            'cer_std': np.std(cers),
            'transcriptions': hypotheses,
            'individual_wers': wers,
            'individual_cers': cers,
        }

        return results

    def evaluate_attack_success(self, original_audio, perturbed_audio,
                                reference_texts, sample_rate=24000):
        """
        Evaluate the success of adversarial attack.

        Args:
            original_audio: Original audio batch [batch, channels, time]
            perturbed_audio: Perturbed audio batch [batch, channels, time]
            reference_texts: Ground truth transcriptions
            sample_rate: Audio sample rate

        Returns:
            Dict with attack success metrics
        """
        # Evaluate original audio
        original_results = self.evaluate_batch(original_audio, reference_texts, sample_rate)

        # Evaluate perturbed audio
        perturbed_results = self.evaluate_batch(perturbed_audio, reference_texts, sample_rate)

        # Calculate attack success metrics
        wer_increase = perturbed_results['wer'] - original_results['wer']
        cer_increase = perturbed_results['cer'] - original_results['cer']

        # Success if WER increases significantly
        attack_success_rate = np.mean([
            1.0 if pwer > ower * 1.5 else 0.0
            for ower, pwer in zip(original_results['individual_wers'],
                                 perturbed_results['individual_wers'])
        ])

        results = {
            'original_wer': original_results['wer'],
            'perturbed_wer': perturbed_results['wer'],
            'wer_increase': wer_increase,
            'original_cer': original_results['cer'],
            'perturbed_cer': perturbed_results['cer'],
            'cer_increase': cer_increase,
            'attack_success_rate': attack_success_rate,
            'original_transcriptions': original_results['transcriptions'],
            'perturbed_transcriptions': perturbed_results['transcriptions'],
        }

        return results


def compute_adversarial_loss(asr_model, audio, target_texts, sample_rate=24000):
    """
    Compute adversarial loss to maximize ASR error.

    Args:
        asr_model: ASR model (Whisper)
        audio: Audio tensor [batch, channels, time]
        target_texts: Original correct transcriptions
        sample_rate: Audio sample rate

    Returns:
        Adversarial loss (higher WER = lower loss for gradient descent)
    """
    # This is a simplified version - in practice you'd want to use
    # the actual logits from the ASR model
    evaluator = ASREvaluator(device=audio.device)

    # Transcribe
    transcriptions = evaluator.transcribe_batch(audio, sample_rate)

    # Calculate WER
    wers = []
    for ref, hyp in zip(target_texts, transcriptions):
        wer = evaluator.calculate_wer(ref, hyp)
        wers.append(wer)

    # We want to maximize WER, so minimize negative WER
    avg_wer = np.mean(wers)
    adversarial_loss = -avg_wer  # Negative because we want to maximize WER

    return torch.tensor(adversarial_loss, requires_grad=True)
