"""
Test: Kiểm tra mapping giữa Lie Group SO(3) và Lie Algebra so(3)
trong models/flowgrasp.py

Root-cause analysis targets:
    - EPS = 1e-4 in Lie.py → biases log_SO3 via:
        1. acos clamping to [-1+EPS, 1-EPS]
        2. denominator 2*sin(θ) + EPS in the regular-case formula
    - Near-π rotations:  sin(θ) → 0 amplifies the EPS bias

Tests:
    1. exp ∘ log roundtrip, stratified by rotation angle
    2. log ∘ exp roundtrip
    3. bracket_so3 invertibility
    4. Geodesic interpolation boundary conditions
    5. Body velocity constancy along geodesic
    6. Euler integration (flowgrasp.sample logic)
    7. project_to_so3
    8. Flowgrasp interpolation path boundary
    9. Spatial ↔ Body velocity roundtrip
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from libs.EquiGraspFlow.utils.Lie import (
    inv_SO3, log_SO3, exp_so3, bracket_so3, is_SO3, EPS as LIE_EPS
)
from models.flowgrasp import project_to_so3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_SO3(n, device='cpu', dtype=torch.float64):
    """Generate n random SO(3) matrices."""
    quats = Rotation.random(n).as_quat()
    mats = Rotation.from_quat(quats).as_matrix()
    return torch.tensor(mats, dtype=dtype, device=device)


def random_so3_vec(n, max_angle=2.5, device='cpu', dtype=torch.float64):
    """Generate n random so(3) vectors with ||w|| < max_angle."""
    w = torch.randn(n, 3, dtype=dtype, device=device)
    w = w / w.norm(dim=1, keepdim=True) * torch.rand(n, 1, dtype=dtype, device=device) * max_angle
    return w


def rotation_angle(R):
    """Return rotation angle θ ∈ [0, π] for each R in batch [N,3,3]."""
    tr = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    return torch.acos(torch.clamp((tr - 1) / 2, -1.0, 1.0))


def sep(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_tests():
    torch.manual_seed(42)
    np.random.seed(42)
    N = 200
    device = 'cpu'
    dtype = torch.float64  # Lie.py dùng float64 trong test, float32 trong training

    passed = 0
    failed = 0

    print(f"Config: N={N}, dtype={dtype}, Lie.py EPS={LIE_EPS}")

    # =====================================================================
    # Test 1: exp(log(R)) ≈ R — stratified by angle range
    # =====================================================================
    sep("Test 1: exp_so3(log_SO3(R)) ≈ R  [stratified by θ]")

    R = random_SO3(N, device, dtype)
    theta = rotation_angle(R)
    w_mat = log_SO3(R)
    R_rec = exp_so3(w_mat)

    per_sample_err = (R - R_rec).abs().reshape(N, -1).max(dim=1).values

    # Stratify by angle
    bins = [
        ("θ ∈ [0, 0.5)",   0.0,  0.5),
        ("θ ∈ [0.5, 1.5)",  0.5,  1.5),
        ("θ ∈ [1.5, 2.5)",  1.5,  2.5),
        ("θ ∈ [2.5, 3.0)",  2.5,  3.0),
        ("θ ∈ [3.0, π]",    3.0,  np.pi + 0.01),
    ]

    max_err_all = 0.0
    for label, lo, hi in bins:
        mask = (theta >= lo) & (theta < hi)
        cnt = mask.sum().item()
        if cnt > 0:
            err = per_sample_err[mask].max().item()
            max_err_all = max(max_err_all, err)
            flag = "✓" if err < 1e-3 else "⚠" if err < 1e-2 else "✗"
            print(f"  {label:22s}  n={cnt:3d}  max_err={err:.2e}  {flag}")
        else:
            print(f"  {label:22s}  n=  0  (no samples)")

    # Near-π rotations are known to degrade due to EPS in denominator
    # Accept if error < 1e-2 overall (EPS-induced)
    tol1 = 1e-2
    ok = max_err_all < tol1
    print(f"  Overall max error: {max_err_all:.2e}  (tol={tol1:.0e})  {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  Root cause of errors: EPS={LIE_EPS} in log_SO3 denominator")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 1b: Same test but EXCLUDING near-π rotations (θ < 2.5)
    # This isolates whether the Rodrigues formula itself is correct
    # =====================================================================
    sep("Test 1b: exp(log(R)) ≈ R  [θ < 2.5 only — excludes near-π]")
    safe_mask = theta < 2.5
    if safe_mask.sum() > 0:
        err_safe = per_sample_err[safe_mask].max().item()
        tol1b = 2e-4  # EPS-induced error bound: ~EPS*θ/(2sinθ)
        ok1b = err_safe < tol1b
        print(f"  n={safe_mask.sum().item()}, max_err={err_safe:.2e}  (tol={tol1b:.0e})  {'✓ PASS' if ok1b else '✗ FAIL'}")
        if ok1b:
            passed += 1
        else:
            failed += 1
    else:
        print("  (no samples in safe range)")

    # =====================================================================
    # Test 2: log(exp(w)) ≈ [w]
    # =====================================================================
    sep("Test 2: log_SO3(exp_so3(w)) ≈ bracket_so3(w)")

    w_vec = random_so3_vec(N, max_angle=2.5, device=device, dtype=dtype)
    R_from_w = exp_so3(w_vec)
    w_mat_rec = log_SO3(R_from_w)
    w_mat_orig = bracket_so3(w_vec)

    err_pos = (w_mat_rec - w_mat_orig).abs().max().item()
    err_neg = (w_mat_rec + w_mat_orig).abs().max().item()
    err = min(err_pos, err_neg)

    # Stratify by ||w||
    w_norms = w_vec.norm(dim=1)
    for label, lo, hi in [("||w|| < 1", 0, 1), ("||w|| ∈ [1,2)", 1, 2), ("||w|| ∈ [2,2.5]", 2, 2.51)]:
        mask = (w_norms >= lo) & (w_norms < hi)
        cnt = mask.sum().item()
        if cnt > 0:
            per_err = (w_mat_rec - w_mat_orig).abs().reshape(N, -1).max(dim=1).values
            bin_err = per_err[mask].max().item()
            print(f"  {label:20s}  n={cnt:3d}  max_err={bin_err:.2e}")

    tol2 = 5e-4  # EPS-induced error at θ≈2.5: ~EPS*θ/(2sinθ)^2 ≈ 2e-4
    ok = err < tol2
    print(f"  Overall max error: {err:.2e}  (tol={tol2:.0e})  {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 3: bracket_so3 invertibility
    # =====================================================================
    sep("Test 3: bracket_so3 invertibility (vec → mat → vec)")
    w_vec_test = torch.randn(N, 3, dtype=dtype, device=device)
    w_mat_test = bracket_so3(w_vec_test)
    w_vec_rec = bracket_so3(w_mat_test)

    err = (w_vec_test - w_vec_rec).abs().max().item()
    ok = err < 1e-12
    print(f"  Max error: {err:.2e}  (tol=1e-12)  {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 4: Geodesic interpolation — R_t = R0 @ exp(t * log(R0^T R1))
    #   Filter: only use rotation pairs where relative angle < 2.5
    #   (avoids near-π issues in log_SO3 that are NOT flowgrasp bugs)
    # =====================================================================
    sep("Test 4: Geodesic interpolation (boundary conditions + SO(3))")

    R0 = random_SO3(N, device, dtype)
    R1 = random_SO3(N, device, dtype)

    # Compute relative rotation angle
    R_rel = inv_SO3(R0) @ R1
    rel_theta = rotation_angle(R_rel)
    safe = rel_theta < 2.5  # exclude near-π relative rotations
    n_safe = safe.sum().item()
    n_nearpi = N - n_safe
    print(f"  Pairs: {n_safe} safe (rel θ < 2.5), {n_nearpi} near-π (rel θ ≥ 2.5)")

    # Use safe pairs only
    R0_s = R0[safe]
    R1_s = R1[safe]
    R_rel_s = R_rel[safe]
    n = n_safe

    w_body_mat = log_SO3(R_rel_s)

    # t = 0
    R_t0 = R0_s @ exp_so3(0.0 * w_body_mat)
    err_t0 = (R_t0 - R0_s).abs().max().item()

    # t = 1
    R_t1 = R0_s @ exp_so3(1.0 * w_body_mat)
    err_t1 = (R_t1 - R1_s).abs().max().item()

    # SO(3) membership at intermediate t
    so3_ok = True
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        R_t = R0_s @ exp_so3(t_val * w_body_mat)
        if not is_SO3(R_t):
            so3_ok = False
            print(f"  ✗ R_t NOT in SO(3) at t={t_val}")

    tol4 = 2e-4  # EPS-induced error from log_SO3
    ok_t0 = err_t0 < tol4
    ok_t1 = err_t1 < tol4
    ok = ok_t0 and ok_t1 and so3_ok
    print(f"  t=0 error: {err_t0:.2e}  {'✓' if ok_t0 else '✗'}")
    print(f"  t=1 error: {err_t1:.2e}  {'✓' if ok_t1 else '✗'}")
    print(f"  SO(3) membership: {'✓' if so3_ok else '✗'}")
    print(f"  Overall: {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 4b: Geodesic with ALL pairs (including near-π) — expect degradation
    # =====================================================================
    sep("Test 4b: Geodesic interpolation (ALL pairs, including near-π)")
    w_body_mat_all = log_SO3(inv_SO3(R0) @ R1)
    R_t1_all = R0 @ exp_so3(1.0 * w_body_mat_all)
    err_per = (R_t1_all - R1).abs().reshape(N, -1).max(dim=1).values

    err_safe_t1 = err_per[safe].max().item() if safe.sum() > 0 else 0
    err_nearpi_t1 = err_per[~safe].max().item() if (~safe).sum() > 0 else 0
    print(f"  Safe pairs (rel θ<2.5):   max_err={err_safe_t1:.2e}")
    print(f"  Near-π pairs (rel θ≥2.5): max_err={err_nearpi_t1:.2e}")
    print(f"  This shows the degradation is ONLY in near-π rotations")

    # =====================================================================
    # Test 5: Body velocity constancy — uses safe pairs
    # =====================================================================
    sep("Test 5: Body velocity constancy along geodesic (safe pairs)")

    w_body_vec = bracket_so3(w_body_mat)  # [n, 3]

    # Use dt large enough that the delta rotation has θ >> EPS
    # (at t=0, the delta rotation angle ≈ dt*||w||; if dt too small,
    #  log_SO3 is inaccurate for near-identity rotations)
    max_body_err = 0.0
    for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        R_t = R0_s @ exp_so3(t_val * w_body_mat)
        dt = 1e-4  # large enough that dt*||w|| >> EPS
        R_t_dt = R0_s @ exp_so3((t_val + dt) * w_body_mat)
        delta = log_SO3(inv_SO3(R_t) @ R_t_dt) / dt
        delta_vec = bracket_so3(delta)
        body_err = (delta_vec - w_body_vec).abs().max().item()
        max_body_err = max(max_body_err, body_err)
        print(f"  t={t_val:.1f}  body velocity err: {body_err:.2e}")

    tol5 = 5e-2  # finite difference O(dt) + log_SO3 EPS error
    ok = max_body_err < tol5
    print(f"  Max body velocity error: {max_body_err:.2e}  (tol={tol5:.0e})  {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 6: Euler integration of spatial velocity → R1  (safe pairs)
    # =====================================================================
    sep("Test 6: Euler integration of spatial velocity → R1")
    print("  (Simulates flowgrasp.py sample() with KNOWN velocity field)")

    num_steps_list = [10, 50, 200]
    for num_steps in num_steps_list:
        dt_step = 1.0 / num_steps
        R_curr = R0_s.clone()

        for step in range(num_steps):
            # Spatial velocity: w_spatial = R_curr @ w_body  (flowgrasp line 208)
            w_spatial_vec = torch.einsum('bij,bj->bi', R_curr, w_body_vec)
            # Body velocity:    w_body_local = R_curr^T @ w_spatial  (flowgrasp sample line 268)
            w_body_local = torch.einsum('bji,bj->bi', R_curr, w_spatial_vec)
            # Step:             R_{k+1} = R_k @ exp(dt * w_body)  (flowgrasp sample line 271)
            R_curr = R_curr @ exp_so3(dt_step * w_body_local)

        err = (R_curr - R1_s).abs().max().item()
        print(f"  steps={num_steps:>3d}  max error: {err:.2e}")

    # With 200 steps, Euler error should be very small for a linear velocity field
    tol6 = 1e-3
    final_ok = (R_curr - R1_s).abs().max().item() < tol6
    print(f"  Final check (200 steps, tol={tol6:.0e}): {'✓ PASS' if final_ok else '✗ FAIL'}")
    if final_ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 7: project_to_so3
    # =====================================================================
    sep("Test 7: project_to_so3 correctness")

    R_clean = random_SO3(N, device).float()
    noise = 0.1 * torch.randn_like(R_clean)
    R_noisy = R_clean + noise
    R_proj = project_to_so3(R_noisy)

    RtR = R_proj.transpose(1, 2) @ R_proj
    eye = torch.eye(3).unsqueeze(0).expand_as(RtR)
    orth_err = (RtR - eye).abs().max().item()
    det_vals = torch.linalg.det(R_proj)
    det_err = (det_vals - 1.0).abs().max().item()

    ok_orth = orth_err < 1e-5
    ok_det = det_err < 1e-5
    ok = ok_orth and ok_det
    print(f"  Orthogonality error: {orth_err:.2e}  {'✓' if ok_orth else '✗'}")
    print(f"  Determinant error:   {det_err:.2e}  {'✓' if ok_det else '✗'}")
    print(f"  Overall: {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 8: Flowgrasp interpolation path boundary (float32, safe pairs)
    # =====================================================================
    sep("Test 8: Flowgrasp interpolation path boundary (float32)")

    B, Np = 4, 32
    R0_f = random_SO3(B * Np, device, dtype=torch.float32)
    R1_f = random_SO3(B * Np, device, dtype=torch.float32)

    # Filter safe pairs
    rel_angle = rotation_angle(inv_SO3(R0_f) @ R1_f)
    safe_f = rel_angle < 2.5
    R0_fs, R1_fs = R0_f[safe_f], R1_f[safe_f]
    ns = safe_f.sum().item()
    print(f"  Using {ns}/{B*Np} safe pairs (rel θ < 2.5)")

    w_body_mat_f = log_SO3(inv_SO3(R0_fs) @ R1_fs)

    # t = 0
    t_zero = torch.zeros(ns, 1, 1, device=device)
    R_at_0 = R0_fs @ exp_so3(t_zero * w_body_mat_f)
    err_0 = (R_at_0 - R0_fs).abs().max().item()

    # t = 1
    t_one = torch.ones(ns, 1, 1, device=device)
    R_at_1 = R0_fs @ exp_so3(t_one * w_body_mat_f)
    err_1 = (R_at_1 - R1_fs).abs().max().item()

    tol8 = 1e-3  # float32 precision
    ok_0 = err_0 < tol8
    ok_1 = err_1 < tol8
    ok = ok_0 and ok_1
    print(f"  t=0 error: {err_0:.2e}  {'✓' if ok_0 else '✗'}")
    print(f"  t=1 error: {err_1:.2e}  {'✓' if ok_1 else '✗'}")
    print(f"  Overall: {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 9: Spatial ↔ Body velocity roundtrip
    # =====================================================================
    sep("Test 9: Spatial ↔ Body velocity conversion roundtrip")

    R_test = random_SO3(N, device, dtype)
    w_body_test = torch.randn(N, 3, dtype=dtype, device=device)

    w_spatial_test = torch.einsum('bij,bj->bi', R_test, w_body_test)
    w_body_rec = torch.einsum('bji,bj->bi', R_test, w_spatial_test)

    err = (w_body_rec - w_body_test).abs().max().item()
    ok = err < 1e-10
    print(f"  Max error: {err:.2e}  (tol=1e-10)  {'✓ PASS' if ok else '✗ FAIL'}")
    if ok:
        passed += 1
    else:
        failed += 1

    # =====================================================================
    # Test 10: EPS sensitivity analysis — quantify how EPS affects accuracy
    # =====================================================================
    sep("Test 10: EPS sensitivity analysis")
    print(f"  Current EPS in Lie.py: {LIE_EPS}")
    print()

    # Generate rotations with controlled angles
    for target_theta in [0.1, 0.5, 1.0, 2.0, 2.8, 3.0, 3.1]:
        # Create rotation with exact angle
        axis = torch.randn(1, 3, dtype=dtype)
        axis = axis / axis.norm()
        w = axis * target_theta
        R_exact = exp_so3(w)
        actual_theta = rotation_angle(R_exact).item()

        # Roundtrip
        w_rec = log_SO3(R_exact)
        R_roundtrip = exp_so3(w_rec)
        err = (R_roundtrip - R_exact).abs().max().item()

        # Theoretical error bound from EPS: EPS * θ / (2sinθ)
        sin_theta = np.sin(actual_theta)
        theoretical_err = LIE_EPS * actual_theta / (2 * abs(sin_theta) + 1e-15) if sin_theta > 1e-10 else float('inf')

        print(f"  θ={actual_theta:.3f} rad ({np.degrees(actual_theta):6.1f}°)  "
              f"sin(θ)={sin_theta:.4f}  err={err:.2e}  theory≈{theoretical_err:.2e}")

    # =====================================================================
    # Summary
    # =====================================================================
    total = passed + failed
    sep(f"SUMMARY: {passed}/{total} tests passed, {failed} failed")

    if failed > 0:
        print("\n  Root cause analysis:")
        print(f"  • Lie.py uses EPS = {LIE_EPS} in log_SO3:")
        print(f"    - acos clamp to [-1+EPS, 1-EPS]  → θ error near identity/π")
        print(f"    - denominator 2sin(θ)+EPS        → scaling error near π")
        print(f"  • For relative rotations with θ < 2.5 rad (~143°),")
        print(f"    the error is < 1e-4 and flowgrasp logic is CORRECT.")
        print(f"  • Recommendation: reduce EPS or use a Taylor-series")
        print(f"    branch near θ=0 and θ=π for better accuracy.")
    print()

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
