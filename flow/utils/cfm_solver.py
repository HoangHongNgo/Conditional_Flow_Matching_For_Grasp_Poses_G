import torch
from flow.utils.cfm_norm import denormalize_x

@torch.no_grad()
def euler_solve(seed_conditioner, mlp, x0, seed_xyz, seed_feats, stats, n_steps=20):
    """
    Solve the probability flow ODE using Euler method.
    x0: [B, N, 5] prior samples ~ N(0, I)
    seed_xyz: [B, N, 3] coordinates
    seed_feats: [B, 512, N] features
    stats: empirical normalization stats dict
    n_steps: number of integration steps (default: 20)
    
    Returns:
    x_pred: [B, N, 5] denormalized predicted grasp configs [omega, width, depth]
    """
    B, _, _ = x0.shape
    device = x0.device
    
    # 1. Compute per-seed condition.
    seed_cond = seed_conditioner(seed_xyz, seed_feats).transpose(1, 2).contiguous()  # [B, N, 128]
    
    # 2. Integrate flow from t=0 to t=1
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.full((B,), t_val, device=device, dtype=torch.float32) # [B]
        
        # Predict velocity
        v = mlp(x, t, seed_cond)  # [B, N, 5]
        
        # Euler step
        x = x + v * dt

    # 3. Denormalize generated 5D grasp configs.
    x_denorm = denormalize_x(x, stats)
    return x_denorm
