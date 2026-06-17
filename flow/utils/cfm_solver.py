import torch
from flow.utils.cfm_norm import denormalize_x

@torch.no_grad()
def euler_solve(encoder, mlp, x0, seed_xyz, seed_feats, stats, n_steps=20):
    """
    Solve the probability flow ODE using Euler method.
    x0: [B, 50, 8]  prior samples ~ N(0, I)
    seed_xyz: [B, 1024, 3] coordinates
    seed_feats: [B, 1024, 512] features
    stats: empirical normalization stats dict
    n_steps: number of integration steps (default: 20)
    
    Returns:
    x_pred: [B, 50, 8] denormalized predicted grasp poses (absolute camera coords)
    """
    B, M, D = x0.shape
    device = x0.device
    
    # 1. Compute global scene condition
    scene_cond = encoder(seed_xyz, seed_feats) # [B, 256]
    
    # 2. Integrate flow from t=0 to t=1
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.full((B,), t_val, device=device, dtype=torch.float32) # [B]
        
        # Predict velocity
        v = mlp(x, t, scene_cond) # [B, 50, 8]
        
        # Euler step
        x = x + v * dt
        
    # Compute median of seed points for each scene in batch
    # seed_median: [B, 1, 3]
    seed_median = torch.median(seed_xyz, dim=1)[0].unsqueeze(1)
        
    # 3. Denormalize output to physical units relative to seed medians
    x_denorm = denormalize_x(x, stats, seed_median)
    return x_denorm
