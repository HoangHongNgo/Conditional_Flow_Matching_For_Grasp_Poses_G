import sys
import os
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.datasets.cfm_dataset import CFMDataset
from flow.models.grasp_cfm import GraspVelocityMLP
from flow.models.modules_flow import Sphere_Grouping_Global_Interaction
from flow.utils.cfm_norm import build_norm_metadata
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)


def sample_masked_flow(FM, x0, x1, target_valid_mask):
    """Sample CFM locations and velocities while ignoring invalid seed targets."""
    B, M, _ = x1.shape
    xt = torch.zeros_like(x1)  # [B, 1024, 5]
    ut = torch.zeros_like(x1)  # [B, 1024, 5]
    t_seed = torch.zeros(B, M, device=x1.device, dtype=x1.dtype)  # [B, 1024]

    for b in range(B):
        valid_idx = torch.where(target_valid_mask[b])[0]  # [num_valid]
        if valid_idx.numel() == 0:
            continue
        t_val, xt_b, ut_b = FM.sample_location_and_conditional_flow(x0[b, valid_idx], x1[b, valid_idx])
        t_seed[b, valid_idx] = t_val
        xt[b, valid_idx] = xt_b
        ut[b, valid_idx] = ut_b

    return t_seed, xt, ut


def build_flow_matcher(fm_type, sigma):
    """Build a torchcfm flow matcher from a short experiment-friendly name."""
    matcher_by_type = {
        'target': TargetConditionalFlowMatcher,
        'independent': ConditionalFlowMatcher,
        'ot': ExactOptimalTransportConditionalFlowMatcher,
    }
    return matcher_by_type[fm_type](sigma=sigma)


def masked_weighted_mse_loss(
    v_pred,
    ut,
    target_valid_mask,
    rot_weight=1.0,
    width_weight=1.0,
    depth_weight=1.0,
):
    """Compute weighted group MSE over valid seed targets only."""
    sq_error = (v_pred - ut) ** 2  # [B, 1024, 5]
    rot_loss = sq_error[..., :3].mean(dim=-1)  # [B, 1024]
    width_loss = sq_error[..., 3]  # [B, 1024]
    depth_loss = sq_error[..., 4]  # [B, 1024]
    weight_sum = rot_weight + width_weight + depth_weight
    per_seed_loss = (
        rot_weight * rot_loss +
        width_weight * width_loss +
        depth_weight * depth_loss
    ) / max(weight_sum, 1e-8)
    mask = target_valid_mask.float()  # [B, 1024]
    return (per_seed_loss * mask).sum() / mask.sum().clamp_min(1.0)


def train_cfm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using training device: {device}")

    resume_checkpoint_path = args.resume_checkpoint
    resume_mode = bool(resume_checkpoint_path)
    start_epoch = 1
    final_epoch = args.epochs
    resume_payload = None

    if resume_mode:
        args.checkpoint_dir = os.path.dirname(os.path.abspath(resume_checkpoint_path))
        print(f"Resuming CFM training from: {resume_checkpoint_path}")
        print(f"Logs and checkpoints will continue in: {args.checkpoint_dir}")
    else:
        # Subfolder named with date-month-year and hour-minute-second of execution
        run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
        print(f"Logs and checkpoints will be saved to: {args.checkpoint_dir}")
    
    # 1. Ensure fixed normalization metadata exists
    if not os.path.exists(args.stats_path):
        os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
        norm_metadata = build_norm_metadata(args.dataset_dir)
        torch.save(norm_metadata, args.stats_path)
        print(f"Validated and saved normalization metadata to: {args.stats_path}")
        
    # 2. Dataset splits
    train_dataset = CFMDataset(
        args.dataset_dir,
        stats_path=args.stats_path,
        limit=args.limit,
        target_sampling=args.target_sampling,
    )
    val_dataset = CFMDataset(
        args.eval_dataset_dir,
        stats_path=args.stats_path,
        limit=args.limit,
        target_sampling=args.target_sampling,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Loaded training dataset from: {args.dataset_dir}")
    print(f"Loaded evaluation dataset from: {args.eval_dataset_dir}")
    print(f"  Training samples: {len(train_dataset)} | Batches: {len(train_loader)}")
    print(f"  Evaluation samples: {len(val_dataset)} | Batches: {len(val_loader)}")
    print(f"Using target sampling strategy: {args.target_sampling}")
    
    # 3. Initialize Models
    seed_conditioner = Sphere_Grouping_Global_Interaction(
        nsample=args.nsample,
        seed_feature_dim=512,
        sphere_radius=args.sphere_radius,
    ).to(device)
    mlp = GraspVelocityMLP(grasp_dim=5, cond_dim=128).to(device)
    
    # Optimizer & Scheduler
    params = list(seed_conditioner.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if resume_mode:
        resume_payload = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        seed_conditioner.load_state_dict(resume_payload['seed_conditioner_state_dict'])
        mlp.load_state_dict(resume_payload['mlp_state_dict'])
        optimizer.load_state_dict(resume_payload['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_payload:
            scheduler.load_state_dict(resume_payload['scheduler_state_dict'])
        start_epoch = int(resume_payload['epoch']) + 1
        final_epoch = int(resume_payload['epoch']) + args.epochs
        print(
            f"Resume state loaded at epoch {resume_payload['epoch']}. "
            f"Training will continue through epoch {final_epoch}."
        )
    
    # Target CFM keeps each seed-conditioned target paired with its own seed.
    FM = build_flow_matcher(args.fm_type, args.fm_sigma)
    if min(args.rot_loss_weight, args.width_loss_weight, args.depth_loss_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.rot_loss_weight + args.width_loss_weight + args.depth_loss_weight <= 0:
        raise ValueError("At least one loss weight must be positive.")
    print(f"Using torchcfm matcher: {args.fm_type} (sigma={args.fm_sigma})")
    print(
        "Using loss weights: "
        f"rot={args.rot_loss_weight}, width={args.width_loss_weight}, depth={args.depth_loss_weight}"
    )
    
    # Create checkpoints directory & log file
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(args.checkpoint_dir, "cfm_training_log.txt")
    
    # If starting fresh, write log header
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("Epoch,Train_Loss,Eval_Loss\n")
            
    # 4. Training Loop
    print("\nStarting CFM training split...")
    for epoch in range(start_epoch, final_epoch + 1):
        # ------------------ TRAINING STEP ------------------
        seed_conditioner.train()
        mlp.train()
        
        train_epoch_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{final_epoch} [Train]")
        for batch in pbar:
            x1 = batch['x1'].to(device)  # [B, 1024, 5]
            target_valid_mask = batch['target_valid_mask'].to(device)  # [B, 1024]
            seed_xyz = batch['seed_xyz'].to(device)  # [B, 1024, 3]
            seed_feats = batch['seed_feats'].to(device)  # [B, 512, 1024]
            
            if not target_valid_mask.any():
                continue
            
            # Sample prior x0 ~ N(0, I)
            x0 = torch.randn_like(x1)  # [B, 1024, 5]
            t_batch, xt, ut = sample_masked_flow(FM, x0, x1, target_valid_mask)
            
            # Encode local seed geometry and gather per-seed condition.
            seed_cond = seed_conditioner(seed_xyz, seed_feats).transpose(1, 2).contiguous()  # [B, 1024, 128]
            
            # Predict velocity
            v_pred = mlp(xt, t_batch, seed_cond)  # [B, 1024, 5]
            
            # Masked weighted MSE ignores seeds with no valid grasp config.
            loss = masked_weighted_mse_loss(
                v_pred,
                ut,
                target_valid_mask,
                rot_weight=args.rot_loss_weight,
                width_weight=args.width_loss_weight,
                depth_weight=args.depth_loss_weight,
            )
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_epoch_loss / max(train_batches, 1)
        
        # ------------------ EVALUATION STEP ------------------
        seed_conditioner.eval()
        mlp.eval()
        
        val_epoch_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{final_epoch} [Eval]"):
                x1 = batch['x1'].to(device)  # [B, 1024, 5]
                target_valid_mask = batch['target_valid_mask'].to(device)  # [B, 1024]
                seed_xyz = batch['seed_xyz'].to(device)  # [B, 1024, 3]
                seed_feats = batch['seed_feats'].to(device)  # [B, 512, 1024]
                if not target_valid_mask.any():
                    continue
                
                x0 = torch.randn_like(x1)
                t_batch, xt, ut = sample_masked_flow(FM, x0, x1, target_valid_mask)
                
                seed_cond = seed_conditioner(seed_xyz, seed_feats).transpose(1, 2).contiguous()
                v_pred = mlp(xt, t_batch, seed_cond)
                loss = masked_weighted_mse_loss(
                    v_pred,
                    ut,
                    target_valid_mask,
                    rot_weight=args.rot_loss_weight,
                    width_weight=args.width_loss_weight,
                    depth_weight=args.depth_loss_weight,
                )
                val_epoch_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_epoch_loss / val_batches if val_batches > 0 else 0.0
        
        # Log to console
        print(f"Epoch {epoch} finished.")
        print(f"  Avg Train Loss: {avg_train_loss:.5f}")
        print(f"  Avg Eval Loss:  {avg_val_loss:.5f}")
        
        # ------------------ WRITE TO LOG FILE ------------------
        with open(log_file_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.5f},{avg_val_loss:.5f}\n")
        print(f"  Saved epoch logs to: {log_file_path}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'seed_conditioner_state_dict': seed_conditioner.state_dict(),
            'mlp_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_val_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"flowgrasp_epoch_{epoch}.tar"))
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "flowgrasp_latest.tar"))
        
    print(f"\nTraining completed! Checkpoints and logs saved to: {args.checkpoint_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching Grasp Pose Generator")
    parser.add_argument('--dataset_dir', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_dataset_seed5d/train',
                        help='Directory of cached .pt training split files')
    parser.add_argument('--eval_dataset_dir', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_dataset_seed5d/eval',
                        help='Directory of cached .pt evaluation split files')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_seed5d_norm_stats.pt',
                        help='Path to fixed normalization metadata')
    parser.add_argument('--checkpoint_dir', type=str, default='flow/results',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='Optional path to flowgrasp checkpoint to continue training from')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size')
    parser.add_argument('--nsample', type=int, default=32, help='Number of neighbor seeds for spherical grouping')
    parser.add_argument('--sphere_radius', type=float, default=0.005, help='Seed grouping radius in meters')
    parser.add_argument(
        '--fm_type',
        type=str,
        default='target',
        choices=['target', 'independent', 'ot'],
        help='torchcfm matcher: target is recommended for seed-conditioned grasp CFM'
    )
    parser.add_argument('--fm_sigma', type=float, default=0.0, help='Conditional flow matcher sigma')
    parser.add_argument(
        '--target_sampling',
        type=str,
        default='top8_rot_weighted',
        choices=['top8_rot_weighted', 'best_score', 'score_weighted', 'uniform'],
        help='How to select one 5D target from each valid seed grasp pool'
    )
    parser.add_argument('--rot_loss_weight', type=float, default=1.0,
                        help='Group loss weight for the 3D Lie rotation velocity')
    parser.add_argument('--width_loss_weight', type=float, default=1.0,
                        help='Group loss weight for the gripper width velocity')
    parser.add_argument('--depth_loss_weight', type=float, default=1.0,
                        help='Group loss weight for the grasp depth velocity')
    
    args = parser.parse_args()
    
    train_cfm(args)
