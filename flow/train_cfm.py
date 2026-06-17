import sys
import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add workspace root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.datasets.cfm_dataset import CFMDataset
from flow.models.grasp_cfm import SceneMinkEncoder, GraspVelocityMLP
from flow.utils.cfm_norm import compute_norm_stats
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
def train_cfm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using training device: {device}")
    
    # Subfolder named with date-month-year and hour-minute-second of execution
    run_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_id)
    print(f"Logs and checkpoints will be saved to: {args.checkpoint_dir}")
    
    # 1. Ensure Norm Stats exist
    if not os.path.exists(args.stats_path):
        os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)
        stats = compute_norm_stats(args.dataset_dir)
        torch.save(stats, args.stats_path)
        print(f"Computed and saved normalization stats to: {args.stats_path}")
        
    # 2. Dataset and Train/Eval Split
    full_dataset = CFMDataset(args.dataset_dir, stats_path=args.stats_path, limit=args.limit)
    
    # Deterministic split: 90% for training, 10% for validation/evaluation
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
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
    
    print(f"Loaded dataset with {len(full_dataset)} total samples.")
    print(f"  Training samples: {len(train_dataset)} | Batches: {len(train_loader)}")
    print(f"  Evaluation samples: {len(val_dataset)} | Batches: {len(val_loader)}")
    
    # 3. Initialize Models
    encoder = SceneMinkEncoder(in_channels=512, out_channels=256).to(device)
    mlp = GraspVelocityMLP(grasp_dim=8, cond_dim=256).to(device)
    
    # Optimizer & Scheduler
    params = list(encoder.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize torchcfm Exact Optimal Transport Flow Matcher
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    
    # Create checkpoints directory & log file
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(args.checkpoint_dir, "cfm_training_log.txt")
    
    # If starting fresh, write log header
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("Epoch,Train_Loss,Eval_Loss\n")
            
    # 4. Training Loop
    print("\nStarting CFM training split...")
    for epoch in range(1, args.epochs + 1):
        # ------------------ TRAINING STEP ------------------
        encoder.train()
        mlp.train()
        
        train_epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            x1 = batch['x1'].to(device)                 # [B, 50, 8] (GT)
            seed_xyz = batch['seed_xyz'].to(device)     # [B, 1024, 3]
            seed_feats = batch['seed_feats'].to(device) # [B, 1024, 512]
            
            B, M, D = x1.shape
            
            # Sample prior x0 ~ N(0, I)
            x0 = torch.randn_like(x1) # [B, 50, 8]
            
            # Prepare target variables
            xt = torch.zeros_like(x1) # [B, 50, 8]
            ut = torch.zeros_like(x1) # [B, 50, 8]
            t_batch = torch.zeros(B, device=device) # [B]
            
            # Solve batch-wise optimal transport scene by scene
            for b in range(B):
                t_val, xt_b, ut_b = FM.sample_location_and_conditional_flow(x0[b], x1[b])
                xt[b] = xt_b
                ut[b] = ut_b
                t_batch[b] = t_val[0]
            
            # Encode geometry
            scene_cond = encoder(seed_xyz, seed_feats) # [B, 256]
            
            # Predict velocity
            v_pred = mlp(xt, t_batch, scene_cond) # [B, 50, 8]
            
            # Standard MSE Loss
            loss = ((v_pred - ut) ** 2).mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        avg_train_loss = train_epoch_loss / len(train_loader)
        
        # ------------------ EVALUATION STEP ------------------
        encoder.eval()
        mlp.eval()
        
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Eval]"):
                x1 = batch['x1'].to(device)                 # [B, 50, 8]
                seed_xyz = batch['seed_xyz'].to(device)     # [B, 1024, 3]
                seed_feats = batch['seed_feats'].to(device) # [B, 1024, 512]
                
                B, M, D = x1.shape
                
                x0 = torch.randn_like(x1)
                xt = torch.zeros_like(x1)
                ut = torch.zeros_like(x1)
                t_batch = torch.zeros(B, device=device)
                
                for b in range(B):
                    t_val, xt_b, ut_b = FM.sample_location_and_conditional_flow(x0[b], x1[b])
                    xt[b] = xt_b
                    ut[b] = ut_b
                    t_batch[b] = t_val[0]
                
                scene_cond = encoder(seed_xyz, seed_feats)
                v_pred = mlp(xt, t_batch, scene_cond)
                loss = ((v_pred - ut) ** 2).mean()
                val_epoch_loss += loss.item()
                
        avg_val_loss = val_epoch_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
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
            'encoder_state_dict': encoder.state_dict(),
            'mlp_state_dict': mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_val_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"flowgrasp_epoch_{epoch}.tar"))
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "flowgrasp_latest.tar"))
        
    print(f"\nTraining completed! Checkpoints and logs saved to: {args.checkpoint_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching Grasp Pose Generator")
    parser.add_argument('--dataset_dir', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_dataset_train',
                        help='Directory of cached pt training files')
    parser.add_argument('--stats_path', type=str, default='/media/dsp520/Grasp_2T/graspnet/cfm_norm_stats.pt',
                        help='Path to normalization stats')
    parser.add_argument('--checkpoint_dir', type=str, default='flow/results',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=str, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size')
    
    # Parse args (ensure batch_size is parsed as int)
    args = parser.parse_args()
    args.batch_size = int(args.batch_size)
    
    train_cfm(args)
