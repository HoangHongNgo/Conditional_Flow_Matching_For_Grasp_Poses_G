import torch
from flow.utils.cfm_norm import denormalize_x

@torch.no_grad()
def euler_solve(seed_conditioner, mlp, x0, seed_xyz, seed_feats, norm_metadata, n_steps=20):
    """
    Solve the probability flow ODE using Euler method.
    x0: [B, N, 5] prior samples ~ N(0, I)
    seed_xyz: [B, N, 3] coordinates
    seed_feats: [B, 512, N] features
    norm_metadata: fixed normalization metadata dict
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
    x_denorm = denormalize_x(x, norm_metadata)
    return x_denorm


@torch.no_grad()
def euler_solve_with_trajectory(seed_conditioner, mlp, x0, seed_xyz, seed_feats,
                                norm_metadata, n_steps=20, save_every=5):
    """
    Solve the probability flow ODE and save intermediate snapshots.

    Args:
        seed_conditioner: Seed conditioning module.
        mlp: Velocity prediction MLP.
        x0: [B, N, 5] prior samples ~ N(0, I).
        seed_xyz: [B, N, 3] seed point coordinates.
        seed_feats: [B, 512, N] seed features.
        norm_metadata: Fixed normalization metadata dict.
        n_steps: Number of Euler integration steps.
        save_every: Save a denormalized snapshot every this many steps.

    Returns:
        trajectory: List of (step_index, x_denorm) tuples. Each x_denorm is [B, N, 5].
                    Always includes step 0 (noise) and the final step.
    """
    B, _, _ = x0.shape
    device = x0.device

    # 1. Compute per-seed condition.
    seed_cond = seed_conditioner(seed_xyz, seed_feats).transpose(1, 2).contiguous()  # [B, N, 128]

    # 2. Integrate flow from t=0 to t=1, saving snapshots.
    x = x0.clone()
    dt = 1.0 / n_steps
    trajectory = [(0, denormalize_x(x.clone(), norm_metadata))]

    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.full((B,), t_val, device=device, dtype=torch.float32)  # [B]

        # Predict velocity.
        v = mlp(x, t, seed_cond)  # [B, N, 5]

        # Euler step.
        x = x + v * dt

        step_idx = i + 1
        is_save_step = (step_idx % save_every == 0)
        is_final_step = (step_idx == n_steps)
        if is_save_step or is_final_step:
            trajectory.append((step_idx, denormalize_x(x.clone(), norm_metadata)))

    return trajectory
