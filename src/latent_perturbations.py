"""
Latent-space perturbations for adversarial attacks on ASR systems.
Applies perturbations directly in EnCodec's encoded latent space.
"""
import torch
import torch.nn as nn


class LatentPerturbationLayer(nn.Module):
    """Learnable perturbations applied in EnCodec's latent space."""

    def __init__(self, n_quantizers=32, codebook_size=1024, perturbation_type='additive'):
        """
        Initialize latent perturbation layer.

        Args:
            n_quantizers: Number of quantizers in EnCodec
            codebook_size: Size of the codebook
            perturbation_type: Type of perturbation ('additive', 'substitution', 'masking')
        """
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.perturbation_type = perturbation_type

        if perturbation_type == 'additive':
            # Learn additive noise in latent space
            # This will be added to the continuous embeddings before quantization
            self.noise_scale = nn.Parameter(torch.tensor(0.1))

        elif perturbation_type == 'substitution':
            # Learn a substitution matrix for codes
            # Maps each code to a probability distribution over other codes
            self.substitution_logits = nn.Parameter(
                torch.zeros(n_quantizers, codebook_size, codebook_size)
            )
            # Initialize to identity (no substitution initially)
            for q in range(n_quantizers):
                self.substitution_logits.data[q] = torch.eye(codebook_size) * 10

        elif perturbation_type == 'masking':
            # Learn which codes to mask/zero out
            self.masking_logits = nn.Parameter(torch.zeros(n_quantizers, codebook_size))

        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        # Additional learned noise parameters
        self.temporal_noise_scale = nn.Parameter(torch.tensor(0.05))
        self.quantizer_noise_scale = nn.Parameter(torch.tensor(0.05))

    def apply_additive_noise(self, encoded_frames):
        """
        Apply additive noise to encoded frames.

        Args:
            encoded_frames: EnCodec encoded frames

        Returns:
            Perturbed encoded frames
        """
        # Extract the codes from EncodedFrame
        codes = encoded_frames[0][0]  # Shape: [batch, n_q, frames]

        # Generate learnable noise
        noise = torch.randn_like(codes.float()) * self.noise_scale

        # Add temporal correlation to noise
        temporal_noise = torch.randn(
            codes.shape[0], codes.shape[1], 1, device=codes.device
        ) * self.temporal_noise_scale
        noise = noise + temporal_noise

        # Add quantizer correlation to noise
        quantizer_noise = torch.randn(
            codes.shape[0], 1, codes.shape[2], device=codes.device
        ) * self.quantizer_noise_scale
        noise = noise + quantizer_noise

        # Add noise to codes (will be re-quantized)
        perturbed_codes = codes.float() + noise

        # Clip to valid code range
        perturbed_codes = torch.clamp(perturbed_codes, 0, self.codebook_size - 1).long()

        # Reconstruct encoded frames with perturbed codes
        encoded_frames[0] = (perturbed_codes,) + encoded_frames[0][1:]

        return encoded_frames

    def apply_substitution(self, encoded_frames):
        """
        Apply code substitution to encoded frames.

        Args:
            encoded_frames: EnCodec encoded frames

        Returns:
            Perturbed encoded frames
        """
        codes = encoded_frames[0][0]  # Shape: [batch, n_q, frames]
        batch_size, n_q, n_frames = codes.shape

        perturbed_codes = codes.clone()

        # Apply substitution for each quantizer
        for q in range(min(n_q, self.n_quantizers)):
            # Get substitution probabilities for this quantizer
            sub_probs = torch.softmax(self.substitution_logits[q], dim=-1)

            # For each code in this quantizer, sample a substitution
            original_codes = codes[:, q, :].flatten()  # [batch * frames]

            # Get probability distribution for each original code
            probs = sub_probs[original_codes]  # [batch * frames, codebook_size]

            # Sample new codes from the distribution
            if self.training:
                # Use Gumbel-Softmax for differentiability during training
                temperature = 0.5
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
                logits = torch.log(probs + 1e-10) + gumbel_noise
                new_codes = torch.argmax(logits / temperature, dim=-1)
            else:
                # Use argmax during inference
                new_codes = torch.argmax(probs, dim=-1)

            perturbed_codes[:, q, :] = new_codes.reshape(batch_size, n_frames)

        # Reconstruct encoded frames with perturbed codes
        encoded_frames[0] = (perturbed_codes,) + encoded_frames[0][1:]

        return encoded_frames

    def apply_masking(self, encoded_frames):
        """
        Apply selective masking to encoded frames.

        Args:
            encoded_frames: EnCodec encoded frames

        Returns:
            Perturbed encoded frames
        """
        codes = encoded_frames[0][0]  # Shape: [batch, n_q, frames]
        batch_size, n_q, n_frames = codes.shape

        perturbed_codes = codes.clone()

        # Apply masking for each quantizer
        for q in range(min(n_q, self.n_quantizers)):
            # Get masking probabilities for this quantizer
            mask_probs = torch.sigmoid(self.masking_logits[q])

            # For each code in this quantizer, decide whether to mask
            original_codes = codes[:, q, :].flatten()  # [batch * frames]
            probs = mask_probs[original_codes]  # [batch * frames]

            # Sample masking decisions
            if self.training:
                # Use Gumbel trick for differentiability
                mask = (probs + torch.rand_like(probs)) > 1.0
            else:
                mask = probs > 0.5

            # Apply mask (set masked codes to 0)
            masked_codes = original_codes.clone()
            masked_codes[mask] = 0

            perturbed_codes[:, q, :] = masked_codes.reshape(batch_size, n_frames)

        # Reconstruct encoded frames with perturbed codes
        encoded_frames[0] = (perturbed_codes,) + encoded_frames[0][1:]

        return encoded_frames

    def forward(self, encoded_frames):
        """
        Apply perturbations to encoded frames.

        Args:
            encoded_frames: EnCodec encoded frames (single or list of batches)

        Returns:
            Perturbed encoded frames
        """
        # Handle batched encoded frames (list)
        if isinstance(encoded_frames, list):
            perturbed_batch = []
            for encoded in encoded_frames:
                if self.perturbation_type == 'additive':
                    perturbed = self.apply_additive_noise(encoded)
                elif self.perturbation_type == 'substitution':
                    perturbed = self.apply_substitution(encoded)
                elif self.perturbation_type == 'masking':
                    perturbed = self.apply_masking(encoded)
                else:
                    perturbed = encoded
                perturbed_batch.append(perturbed)
            return perturbed_batch
        else:
            # Single encoded frame
            if self.perturbation_type == 'additive':
                return self.apply_additive_noise(encoded_frames)
            elif self.perturbation_type == 'substitution':
                return self.apply_substitution(encoded_frames)
            elif self.perturbation_type == 'masking':
                return self.apply_masking(encoded_frames)
            else:
                return encoded_frames

    def get_perturbation_strength(self):
        """Get a measure of perturbation strength."""
        if self.perturbation_type == 'additive':
            return {
                'noise_scale': self.noise_scale.item(),
                'temporal_noise_scale': self.temporal_noise_scale.item(),
                'quantizer_noise_scale': self.quantizer_noise_scale.item(),
            }
        elif self.perturbation_type == 'substitution':
            # Measure how much substitution matrix deviates from identity
            identity_bonus = 0.0
            for q in range(self.n_quantizers):
                probs = torch.softmax(self.substitution_logits[q], dim=-1)
                identity_bonus += torch.diag(probs).mean().item()
            return {
                'avg_identity_prob': identity_bonus / self.n_quantizers,
            }
        elif self.perturbation_type == 'masking':
            return {
                'avg_masking_prob': torch.sigmoid(self.masking_logits).mean().item(),
            }
        else:
            return {}


class AdversarialLatentModel(nn.Module):
    """
    End-to-end model that applies latent perturbations for adversarial attacks.
    """

    def __init__(self, encodec_processor, perturbation_type='additive'):
        """
        Initialize adversarial latent model.

        Args:
            encodec_processor: EnCodecProcessor instance
            perturbation_type: Type of perturbation to apply
        """
        super().__init__()
        self.encodec = encodec_processor

        # Get number of quantizers from EnCodec model
        n_q = self.encodec.model.quantizer.n_q
        codebook_size = self.encodec.model.quantizer.bins

        self.perturbation_layer = LatentPerturbationLayer(
            n_quantizers=n_q,
            codebook_size=codebook_size,
            perturbation_type=perturbation_type
        )

    def forward(self, audio):
        """
        Encode audio, perturb in latent space, and decode.

        Args:
            audio: Input audio tensor [batch, channels, time]

        Returns:
            Perturbed audio tensor [batch, channels, time]
        """
        # Encode to latent space
        encoded_frames = self.encodec.encode(audio)

        # Apply perturbations in latent space
        perturbed_frames = self.perturbation_layer(encoded_frames)

        # Decode back to audio
        perturbed_audio = self.encodec.decode(perturbed_frames)

        return perturbed_audio

    def get_perturbation_strength(self):
        """Get perturbation strength metrics."""
        return self.perturbation_layer.get_perturbation_strength()
