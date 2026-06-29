import argparse
import os
import random
import sys
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add workspace root to sys.path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.datasets.scoring_dataset import ScoringDataset, unpack_score_label
from flow.models.scoring_network import ScoringMLP, logits_to_expected_score


def set_seed(seed):
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(device):
    """Enable backend settings that improve CUDA training throughput."""
    if device.type != 'cuda':
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def move_batch_to_device(batch, device):
    """Move scoring tensors in a collated batch to the selected device."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_resume_checkpoint(args):
    """Resolve the checkpoint path used to resume scoring training."""
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint
    if args.resume_latest:
        return os.path.join(args.checkpoint_dir, "scoring_latest.tar")
    return None


def load_resume_checkpoint(resume_path, device):
    """Load a scoring-training checkpoint from disk."""
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    return torch.load(resume_path, map_location=device, weights_only=False)


def count_score_classes(dataset):
    """Count score labels over the valid seeds in the selected dataset files."""
    counts = torch.zeros(11, dtype=torch.float64)
    for filename in tqdm(dataset.files, desc="Counting score classes"):
        sample_path = os.path.join(dataset.dataset_dir, filename)
        sample = torch.load(sample_path, map_location='cpu', weights_only=True)
        valid_mask = sample['seed_valid_mask'].squeeze(0).bool()  # [N]
        labels = unpack_score_label(sample).squeeze(0).long()  # [N, 300]
        valid_labels = labels[valid_mask].reshape(-1)  # [Nv * 300]
        counts += torch.bincount(valid_labels, minlength=11).to(torch.float64)
    return counts


def build_class_weights(dataset, cache_path=None, max_class_weight=5.0, force_recompute=False):
    """Build normalized inverse-sqrt class weights for 11-class CE loss."""
    if cache_path is not None and os.path.exists(cache_path) and not force_recompute:
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        weights = cached['class_weights'] if isinstance(cached, dict) else cached
        return weights.float()

    counts = count_score_classes(dataset)
    frequencies = counts / counts.sum().clamp_min(1.0)
    weights = torch.rsqrt(frequencies + 1e-8)
    weights = weights / weights.mean().clamp_min(1e-8)
    weights = torch.clamp(weights, max=max_class_weight).float()

    if cache_path is not None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                'class_counts': counts,
                'class_weights': weights,
                'max_class_weight': max_class_weight,
            },
            cache_path,
        )
    return weights


def estimate_sampled_class_balance(dataset, class_weights):
    """Estimate class balance after per-frame sampling and CE class weighting."""
    sampled_counts = torch.zeros(11, dtype=torch.float64)
    for idx in range(len(dataset)):
        item = dataset[idx]
        sampled_counts += torch.bincount(item['score_label'], minlength=11).to(torch.float64)

    sampled_frequencies = sampled_counts / sampled_counts.sum().clamp_min(1.0)
    weighted_counts = sampled_counts * class_weights.to(torch.float64)
    weighted_frequencies = weighted_counts / weighted_counts.sum().clamp_min(1.0)
    return sampled_frequencies, weighted_frequencies


def update_metrics(metrics, logits, labels, loss):
    """Accumulate loss and score-class metrics for one batch."""
    with torch.no_grad():
        # logits: [B, K, 11], labels: [B, K]
        pred_class = torch.argmax(logits, dim=-1)  # [B, K]
        expected_score = logits_to_expected_score(logits)  # [B, K]
        target_score = labels.float() / 10.0  # [B, K]

        batch_count = labels.numel()
        metrics['loss_sum'] += float(loss.item()) * batch_count
        metrics['sample_count'] += batch_count
        metrics['correct'] += int((pred_class == labels).sum().item())
        metrics['abs_error_sum'] += float(torch.abs(expected_score - target_score).sum().item())

        positive_mask = labels > 0
        metrics['positive_total'] += int(positive_mask.sum().item())
        if positive_mask.any():
            metrics['positive_hit'] += int((pred_class[positive_mask] > 0).sum().item())

        high_mask = labels >= 8
        metrics['high_total'] += int(high_mask.sum().item())
        if high_mask.any():
            metrics['high_hit'] += int((pred_class[high_mask] >= 8).sum().item())


def finalize_metrics(metrics):
    """Convert accumulated metric sums into scalar averages."""
    sample_count = max(metrics['sample_count'], 1)
    positive_total = metrics['positive_total']
    high_total = metrics['high_total']
    return {
        'loss': metrics['loss_sum'] / sample_count,
        'accuracy': metrics['correct'] / sample_count,
        'mae': metrics['abs_error_sum'] / sample_count,
        'positive_recall': metrics['positive_hit'] / positive_total if positive_total > 0 else 0.0,
        'high_score_recall': metrics['high_hit'] / high_total if high_total > 0 else 0.0,
        'positive_total': positive_total,
        'high_total': high_total,
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    epochs,
    grad_clip_norm,
    scaler,
    use_amp,
):
    """Train the scoring classifier for one epoch."""
    model.train()
    metrics = {
        'loss_sum': 0.0,
        'sample_count': 0,
        'correct': 0,
        'abs_error_sum': 0.0,
        'positive_hit': 0,
        'positive_total': 0,
        'high_hit': 0,
        'high_total': 0,
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [Train]")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        grasp_config = batch['grasp_config']  # [B, K, 133]
        labels = batch['score_label']  # [B, K]

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            if use_amp else nullcontext()
        )
        with autocast_context:
            logits = model(grasp_config)  # [B, K, 11]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        update_metrics(metrics, logits.detach(), labels, loss.detach())
        current = finalize_metrics(metrics)
        pbar.set_postfix(
            {
                'loss': f"{current['loss']:.4f}",
                'acc': f"{current['accuracy']:.3f}",
                'pos_rec': f"{current['positive_recall']:.3f}",
            }
        )

    return finalize_metrics(metrics)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, class_weights, args):
    """Save only the latest scoring-training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'train_metrics': metrics,
        'class_weights': class_weights.detach().cpu(),
        'args': vars(args),
    }
    latest_path = os.path.join(args.checkpoint_dir, "scoring_latest.tar")
    for filename in os.listdir(args.checkpoint_dir):
        if filename.startswith("scoring_epoch_") and filename.endswith(".tar"):
            os.remove(os.path.join(args.checkpoint_dir, filename))
    torch.save(checkpoint, latest_path)


def train_scoring(args):
    """Train a seed-conditioned 11-class grasp scoring network."""
    if args.num_classes != 11:
        raise ValueError("Scoring training expects exactly 11 score classes for labels 0..10.")

    set_seed(args.seed)
    device_name = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    configure_torch_runtime(device)

    if args.run_subdir:
        run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = resolve_resume_checkpoint(args)
    resume_checkpoint = load_resume_checkpoint(resume_path, device) if resume_path is not None else None

    dataset = ScoringDataset(
        args.dataset_dir,
        configs_per_frame=args.configs_per_frame,
        positive_fraction=args.positive_fraction,
        sampling=args.sampling,
        limit=args.limit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    weights_cache_path = args.class_weights_path
    if weights_cache_path is None:
        weights_cache_path = os.path.join(args.checkpoint_dir, "scoring_class_weights.pt")
    class_weights = build_class_weights(
        dataset,
        cache_path=weights_cache_path,
        max_class_weight=args.max_class_weight,
        force_recompute=args.recompute_class_weights,
    )

    model = ScoringMLP(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    if args.compile_model and device.type == 'cuda' and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint['model_state_dict'])

    checkpoint_class_weights = None
    if resume_checkpoint is not None and 'class_weights' in resume_checkpoint:
        checkpoint_class_weights = resume_checkpoint['class_weights'].float()
        class_weights = checkpoint_class_weights

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])

    use_amp = bool(args.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if device.type == 'cuda' else None
    if resume_checkpoint is not None and scaler is not None and resume_checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])

    start_epoch = int(resume_checkpoint['epoch']) if resume_checkpoint is not None else 0
    total_epochs = start_epoch + args.additional_epochs if args.additional_epochs > 0 else args.epochs
    if total_epochs <= start_epoch:
        raise ValueError(
            f"Total epochs ({total_epochs}) must be larger than resumed epoch ({start_epoch})."
        )

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=args.min_lr,
            last_epoch=start_epoch - 1,
        )
    else:
        scheduler = None
    if (
        resume_checkpoint is not None
        and scheduler is not None
        and args.additional_epochs <= 0
        and resume_checkpoint.get('scheduler_state_dict') is not None
    ):
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

    print(f"Using device: {device}")
    print(f"Loaded scoring dataset from: {args.dataset_dir}")
    print(f"Training frames: {len(dataset)} | batches: {len(dataloader)}")
    print(
        f"Sampling: {args.sampling} | configs_per_frame={args.configs_per_frame} | "
        f"positive_fraction={args.positive_fraction}"
    )
    print(f"num_workers={args.num_workers} | prefetch_factor={args.prefetch_factor} | amp={use_amp}")
    print(f"Optimizer: AdamW | lr={args.lr} | scheduler={args.scheduler} | min_lr={args.min_lr}")
    print(f"Epoch plan: start_epoch={start_epoch} | total_epochs={total_epochs}")
    if args.compile_model and device.type == 'cuda' and hasattr(torch, "compile"):
        print(f"torch.compile enabled with mode={args.compile_mode}")
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
    print(f"Class weights: {class_weights.tolist()}")
    if args.report_sampled_balance:
        sampled_freq, weighted_freq = estimate_sampled_class_balance(dataset, class_weights)
        print(
            "Estimated sampled balance: "
            f"zero={sampled_freq[0].item():.4f}, positive={sampled_freq[1:].sum().item():.4f}"
        )
        print(
            "Estimated weighted loss balance: "
            f"zero={weighted_freq[0].item():.4f}, positive={weighted_freq[1:].sum().item():.4f}"
        )
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_lr = optimizer.param_groups[0]['lr']
        metrics = train_one_epoch(
            model,
            dataloader,
            criterion,
            optimizer,
            device,
            epoch,
            total_epochs,
            args.grad_clip_norm,
            scaler,
            use_amp,
        )
        print(
            f"Epoch {epoch} | "
            f"lr={epoch_lr:.8f} | "
            f"loss={metrics['loss']:.6f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"mae={metrics['mae']:.4f} | "
            f"positive_recall={metrics['positive_recall']:.4f} | "
            f"high_score_recall={metrics['high_score_recall']:.4f} "
            f"(positive_total={metrics['positive_total']}, high_total={metrics['high_total']})"
        )
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, class_weights, args)


def parse_args():
    """Parse command-line arguments for scoring-network training."""
    parser = argparse.ArgumentParser(description="Train an 11-class grasp scoring network.")
    parser.add_argument('--dataset_dir', type=str,
                        default='/media/dsp520/Grasp_2T/graspnet/scoring_dataset_top8_neighbor/train',
                        help='Directory containing compact scoring .pt samples.')
    parser.add_argument('--checkpoint_dir', type=str, default='flow/results/scoring_network',
                        help='Directory for scoring network checkpoints.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size in frames.')
    parser.add_argument('--configs_per_frame', type=int, default=8192,
                        help='Number of grasp configs sampled from each frame.')
    parser.add_argument('--positive_fraction', type=float, default=0.2,
                        help='Fraction of sampled configs with score > 0 when using balanced sampling.')
    parser.add_argument('--sampling', type=str, default='balanced', choices=['balanced', 'random'],
                        help='Config sampling strategy inside each frame.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--additional_epochs', type=int, default=0,
                        help='Extra epochs to train after the resumed checkpoint epoch.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduling.')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'],
                        help='Learning-rate scheduler. Use none to keep a fixed learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay.')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of prefetched batches per worker.')
    parser.add_argument('--limit', type=int, default=None, help='Optional number of frame files to use.')
    parser.add_argument('--device', type=str, default=None, help='Device override, e.g. cuda:0 or cpu.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--input_dim', type=int, default=133, help='Input dimension for each grasp config.')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of discrete score classes.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for ScoringMLP.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability in ScoringMLP.')
    parser.add_argument('--max_class_weight', type=float, default=5.0,
                        help='Maximum value after inverse-sqrt class-weight normalization.')
    parser.add_argument('--class_weights_path', type=str, default=None,
                        help='Optional cache path for computed class weights.')
    parser.add_argument('--recompute_class_weights', action='store_true',
                        help='Recompute class weights even when the cache path exists.')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Gradient clipping max norm. Use <=0 to disable.')
    parser.add_argument('--run_subdir', action='store_true',
                        help='Save checkpoints under a timestamped checkpoint_dir subfolder.')
    parser.add_argument('--report_sampled_balance', action='store_true',
                        help='Estimate sampled and weighted zero/positive balance before training.')
    parser.add_argument('--amp', action='store_true',
                        help='Enable CUDA automatic mixed precision training.')
    parser.add_argument('--compile_model', action='store_true',
                        help='Enable torch.compile for the scoring model when CUDA is available.')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode used when --compile_model is enabled.')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Checkpoint path used to resume model, optimizer, and scaler states.')
    parser.add_argument('--resume_latest', action='store_true',
                        help='Resume from checkpoint_dir/scoring_latest.tar.')
    return parser.parse_args()


if __name__ == '__main__':
    train_scoring(parse_args())
