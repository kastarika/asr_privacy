"""
Training script for latent-space perturbations.
Learns perturbations in EnCodec's encoded latent space.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

from encodec_utils import EnCodecProcessor
from latent_perturbations import AdversarialLatentModel
from dataset import get_dataloader
from asr_evaluation import ASREvaluator
from audio_metrics import evaluate_audio_quality, compute_quality_loss


def train_epoch(model, dataloader, optimizer, asr_evaluator, device, epoch, writer, args):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_quality_loss = 0.0
    total_asr_wer = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        texts = batch['text']

        # Forward pass
        perturbed_audio = model(audio)

        # Calculate quality loss (we want to minimize change)
        quality_loss = compute_quality_loss(audio, perturbed_audio)

        # Calculate ASR performance (we want to maximize WER)
        # Note: This is expensive, so we do it every N batches
        if batch_idx % args.asr_eval_freq == 0:
            with torch.no_grad():
                asr_results = asr_evaluator.evaluate_batch(
                    perturbed_audio,
                    texts,
                    sample_rate=model.encodec.sample_rate
                )
                current_wer = asr_results['wer']

                # Adversarial loss: we want high WER
                # Convert to loss: negative WER (minimize -WER = maximize WER)
                asr_loss = -current_wer
        else:
            asr_loss = 0.0
            current_wer = 0.0

        # Combined loss
        # Balance between maintaining quality and fooling ASR
        loss = args.quality_weight * quality_loss + args.adversarial_weight * asr_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        total_quality_loss += quality_loss.item()
        if batch_idx % args.asr_eval_freq == 0:
            total_asr_wer += current_wer

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'quality': quality_loss.item(),
            'wer': current_wer if batch_idx % args.asr_eval_freq == 0 else 0.0
        })

        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/QualityLoss', quality_loss.item(), global_step)
        if batch_idx % args.asr_eval_freq == 0:
            writer.add_scalar('Train/WER', current_wer, global_step)

        # Log perturbation strength
        if batch_idx % 100 == 0:
            strength = model.get_perturbation_strength()
            for param_name, param_value in strength.items():
                writer.add_scalar(f'Perturbations/{param_name}', param_value, global_step)

    avg_loss = total_loss / num_batches
    avg_quality_loss = total_quality_loss / num_batches
    avg_wer = total_asr_wer / (num_batches / args.asr_eval_freq + 1)

    return {
        'loss': avg_loss,
        'quality_loss': avg_quality_loss,
        'wer': avg_wer
    }


def evaluate(model, dataloader, asr_evaluator, device, epoch, writer):
    """Evaluate on validation set."""
    model.eval()

    all_original_audio = []
    all_perturbed_audio = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            texts = batch['text']

            perturbed_audio = model(audio)

            all_original_audio.append(audio.cpu())
            all_perturbed_audio.append(perturbed_audio.cpu())
            all_texts.extend(texts)

    # Concatenate all batches
    original_audio = torch.cat(all_original_audio, dim=0)
    perturbed_audio = torch.cat(all_perturbed_audio, dim=0)

    # Evaluate ASR performance
    asr_results = asr_evaluator.evaluate_attack_success(
        original_audio,
        perturbed_audio,
        all_texts,
        sample_rate=model.encodec.sample_rate
    )

    # Evaluate audio quality
    quality_results = evaluate_audio_quality(
        original_audio,
        perturbed_audio,
        sample_rate=model.encodec.sample_rate
    )

    # Combine results
    results = {**asr_results, **quality_results}

    # Log to tensorboard
    writer.add_scalar('Eval/Original_WER', results['original_wer'], epoch)
    writer.add_scalar('Eval/Perturbed_WER', results['perturbed_wer'], epoch)
    writer.add_scalar('Eval/WER_Increase', results['wer_increase'], epoch)
    writer.add_scalar('Eval/Attack_Success_Rate', results['attack_success_rate'], epoch)
    writer.add_scalar('Eval/PESQ', results['pesq'], epoch)
    writer.add_scalar('Eval/STOI', results['stoi'], epoch)
    writer.add_scalar('Eval/SNR', results['snr'], epoch)

    return results


def main(args):
    """Main training function."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))

    # Initialize EnCodec
    print("Loading EnCodec...")
    encodec = EnCodecProcessor(device=device)

    # Initialize model
    print("Initializing model...")
    model = AdversarialLatentModel(encodec, perturbation_type=args.perturbation_type)
    model.to(device)

    # Initialize ASR evaluator
    print("Loading ASR model...")
    asr_evaluator = ASREvaluator(model_name=args.asr_model, device=device)

    # Create dataloaders
    print("Loading datasets...")
    train_loader = get_dataloader(
        split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=args.audio_length,
        target_sr=encodec.sample_rate,
        cache_dir=args.cache_dir,
        shuffle=True
    )

    val_loader = get_dataloader(
        split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_length=args.audio_length,
        target_sr=encodec.sample_rate,
        cache_dir=args.cache_dir,
        shuffle=False
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.perturbation_layer.parameters(),
        lr=args.learning_rate
    )

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    # Training loop
    best_wer_increase = 0.0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, asr_evaluator,
            device, epoch, writer, args
        )

        print(f"Train Loss: {train_results['loss']:.4f}, "
              f"Quality Loss: {train_results['quality_loss']:.4f}, "
              f"WER: {train_results['wer']:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            eval_results = evaluate(
                model, val_loader, asr_evaluator,
                device, epoch, writer
            )

            print(f"Eval - Original WER: {eval_results['original_wer']:.4f}, "
                  f"Perturbed WER: {eval_results['perturbed_wer']:.4f}, "
                  f"WER Increase: {eval_results['wer_increase']:.4f}")
            print(f"Audio Quality - PESQ: {eval_results['pesq']:.4f}, "
                  f"STOI: {eval_results['stoi']:.4f}, "
                  f"SNR: {eval_results['snr']:.2f} dB")

            # Save best model
            if eval_results['wer_increase'] > best_wer_increase:
                best_wer_increase = eval_results['wer_increase']
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_results': eval_results,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

        # Update learning rate
        scheduler.step()

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    print(f"\nTraining complete! Final model saved to {final_path}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train latent-space perturbations')

    # Data parameters
    parser.add_argument('--train_split', type=str, default='train.clean.100',
                        help='Training data split')
    parser.add_argument('--val_split', type=str, default='dev.clean',
                        help='Validation data split')
    parser.add_argument('--cache_dir', type=str, default='./data',
                        help='Data cache directory')
    parser.add_argument('--audio_length', type=int, default=96000,
                        help='Fixed audio length in samples (4 sec at 24kHz)')

    # Model parameters
    parser.add_argument('--asr_model', type=str, default='openai/whisper-small',
                        help='ASR model for evaluation')
    parser.add_argument('--perturbation_type', type=str, default='additive',
                        choices=['additive', 'substitution', 'masking'],
                        help='Type of latent perturbation')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--quality_weight', type=float, default=1.0,
                        help='Weight for quality loss')
    parser.add_argument('--adversarial_weight', type=float, default=10.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--asr_eval_freq', type=int, default=10,
                        help='Frequency of ASR evaluation during training')

    # Evaluation parameters
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Validation frequency (epochs)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Checkpoint save frequency (epochs)')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs/latent_perturbations',
                        help='Output directory')

    args = parser.parse_args()

    main(args)
