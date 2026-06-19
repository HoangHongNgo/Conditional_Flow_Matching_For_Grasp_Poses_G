"""Lie group utilities for batched SO(3) and SE(3) tensor operations."""

import torch
import numpy as np
from scipy.spatial.transform import Rotation


EPS = 1e-4


def is_SO3(R):
    """
    Check whether a batch of matrices is approximately in SO(3).

    Args:
        R (torch.Tensor): Rotation matrix candidates with shape [N, 3, 3].

    Returns:
        bool: True when every matrix is approximately orthonormal under the
        configured tolerance.
    """
    # R: [N, 3, 3], R @ R^T should be identity for every rotation matrix.
    test_0 = torch.allclose(R @ R.transpose(1, 2), torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)
    test_1 = torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).repeat(len(R), 1, 1).to(R), atol=EPS)

    test = test_0 and test_1

    return test


def is_SE3(T):
    """
    Check whether a batch of matrices is approximately in SE(3).

    Args:
        T (torch.Tensor): Homogeneous transform candidates with shape [N, 4, 4].

    Returns:
        bool: True when the rotation block is in SO(3) and the bottom row has
        homogeneous-transform form [0, 0, 0, 1].
    """
    # T: [N, 4, 4], with a valid SO(3) block and homogeneous bottom row [0, 0, 0, 1].
    test_0 = is_SO3(T[:, :3, :3])
    test_1 = torch.equal(T[:, 3, :3], torch.zeros_like(T[:, 3, :3]))
    test_2 = torch.equal(T[:, 3, 3], torch.ones_like(T[:, 3, 3]))

    test = test_0 and test_1 and test_2

    return test


def inv_SO3(R):
    """
    Invert a batch of SO(3) rotation matrices.

    Args:
        R (torch.Tensor): Valid rotation matrices with shape [N, 3, 3].

    Returns:
        torch.Tensor: Inverse rotations with shape [N, 3, 3].
    """
    assert R.shape[1:] == (3, 3), f"inv_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"
    assert is_SO3(R), "inv_SO3: input must be SO(3) matrices"

    # For rotation matrices, inverse equals transpose. Shape: [N, 3, 3].
    inv_R = R.transpose(1, 2)

    return inv_R


def inv_SE3(T):
    """
    Invert a batch of SE(3) homogeneous transform matrices.

    Args:
        T (torch.Tensor): Valid homogeneous transforms with shape [N, 4, 4].

    Returns:
        torch.Tensor: Inverse transforms with shape [N, 4, 4].
    """
    assert T.shape[1:] == (4, 4), f"inv_SE3: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "inv_SE3: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    inv_T = torch.eye(4).repeat(len(T), 1, 1).to(T)
    inv_T[:, :3, :3] = inv_SO3(R)
    # Translation inverse is -R^T p. Shape: [N, 3].
    inv_T[:, :3, 3] = - torch.einsum('nij,nj->ni', inv_SO3(R), p)

    return inv_T


def bracket_so3(w):
    """
    Convert between so(3) vectors and skew-symmetric matrices.

    Args:
        w (torch.Tensor): Either tangent vectors with shape [N, 3] or
            skew-symmetric matrices with shape [N, 3, 3].

    Returns:
        torch.Tensor: Skew matrices with shape [N, 3, 3] for vector input, or
        tangent vectors with shape [N, 3] for matrix input.
    """
    # Vector -> matrix: [N, 3] -> [N, 3, 3].
    if w.shape[1:] == (3,):
        zeros = w.new_zeros(len(w))

        out = torch.stack([
            torch.stack([zeros, -w[:, 2], w[:, 1]], dim=1),
            torch.stack([w[:, 2], zeros, -w[:, 0]], dim=1),
            torch.stack([-w[:, 1], w[:, 0], zeros], dim=1)
        ], dim=1)

    # Matrix -> vector: [N, 3, 3] -> [N, 3].
    elif w.shape[1:] == (3, 3):
        out = torch.stack([w[:, 2, 1], w[:, 0, 2], w[:, 1, 0]], dim=1)

    else:
        raise f"bracket_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w.shape)}"

    return out


def bracket_se3(S):
    """
    Convert between se(3) twist vectors and matrix representation.

    Args:
        S (torch.Tensor): Either twist vectors with shape [N, 6] ordered as
            [omega, v], or se(3) matrices with shape [N, 4, 4].

    Returns:
        torch.Tensor: Matrix representation with shape [N, 4, 4] for vector
        input, or twist vectors with shape [N, 6] for matrix input.
    """
    # Vector -> matrix: [N, 6] -> [N, 4, 4], where S = [omega, v].
    if S.shape[1:] == (6,):
        w_mat = bracket_so3(S[:, :3])

        out = torch.cat((
            torch.cat((w_mat, S[:, 3:].unsqueeze(2)), dim=2),
            S.new_zeros(len(S), 1, 4)
        ), dim=1)

    # Matrix -> vector: [N, 4, 4] -> [N, 6].
    elif S.shape[1:] == (4, 4):
        w_vec = bracket_so3(S[:, :3, :3])

        out = torch.cat((w_vec, S[:, :3, 3]), dim=1)

    else:
        raise f"bracket_se: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    return out


def log_SO3(R):
    """
    Map a batch of SO(3) rotations to so(3) skew matrices.

    Args:
        R (torch.Tensor): Valid rotation matrices with shape [N, 3, 3].

    Returns:
        torch.Tensor: Matrix logarithms with shape [N, 3, 3].
    """
    n = R.shape[0]
    assert R.shape == (n, 3, 3), f"log_SO3: input must be of shape (N, 3, 3). Current shape: {tuple(R.shape)}"
    assert is_SO3(R), "log_SO3: input must be SO(3) matrices"

    # theta: [N], computed from trace(R) = 1 + 2 cos(theta).
    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    w_mat = torch.zeros_like(R)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS))

    is_regular = (tr_R + 1 > EPS)
    is_singular = (tr_R + 1 <= EPS)

    theta = theta.unsqueeze(1).unsqueeze(2)

    # Regular branch uses the closed-form matrix logarithm for rotations away from pi.
    w_mat_regular = (1 / (2 * torch.sin(theta[is_regular]) + EPS)) * (R[is_regular] - R[is_regular].transpose(1, 2)) * theta[is_regular]

    # Singular branch handles rotations close to pi, where sin(theta) is unstable.
    w_mat_singular = (R[is_singular] - torch.eye(3).to(R)) / 2

    w_vec_singular = torch.sqrt(torch.diagonal(w_mat_singular, dim1=1, dim2=2) + 1)
    w_vec_singular[torch.isnan(w_vec_singular)] = 0

    w_1 = w_vec_singular[:, 0]
    w_2 = w_vec_singular[:, 1] * (torch.sign(w_mat_singular[:, 0, 1]) + (w_1 == 0))
    w_3 = w_vec_singular[:, 2] * torch.sign(4 * torch.sign(w_mat_singular[:, 0, 2]) + 2 * (w_1 == 0) * torch.sign(w_mat_singular[:, 1, 2]) + 1 * (w_1 == 0) * (w_2 == 0))

    w_vec_singular = torch.stack([w_1, w_2, w_3], dim=1)

    w_mat[is_regular] = w_mat_regular
    w_mat[is_singular] = bracket_so3(w_vec_singular) * torch.pi

    return w_mat


def rotation_matrix_to_lie_vector(R):
    """
    Convert batched SO(3) rotation matrices to Lie vectors.

    Args:
        R (torch.Tensor): Valid rotation matrices with shape [N, 3, 3].

    Returns:
        torch.Tensor: Lie vectors with shape [N, 3].
    """
    w_mat = log_SO3(R)  # [N, 3, 3]
    w_vec = bracket_so3(w_mat)  # [N, 3]

    return w_vec


def log_SE3(T):
    """
    Map a batch of SE(3) transforms to se(3) matrix representation.

    Args:
        T (torch.Tensor): Valid homogeneous transforms with shape [N, 4, 4].

    Returns:
        torch.Tensor: se(3) matrix logarithms with shape [N, 4, 4].
    """
    assert T.shape[1:] == (4, 4), f"log_SE3: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "log_SE3: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    tr_R = torch.diagonal(R, dim1=1, dim2=2).sum(1)
    theta = torch.acos(torch.clamp((tr_R - 1) / 2, -1 + EPS, 1 - EPS)).unsqueeze(1).unsqueeze(2)

    w_mat = log_SO3(R)
    w_mat_hat = w_mat / (theta + EPS)

    # inv_G maps translation p into the linear velocity component of the twist.
    inv_G = torch.eye(3).repeat(len(T), 1, 1).to(T) - (theta / 2) * w_mat_hat + (1 - (theta / (2 * torch.tan(theta / 2) + EPS))) * w_mat_hat @ w_mat_hat

    S = torch.zeros_like(T)
    S[:, :3, :3] = w_mat
    S[:, :3, 3] = torch.einsum('nij,nj->ni', inv_G, p)

    return S


def exp_so3(w_vec):
    """
    Map batched so(3) vectors or matrices to SO(3) rotations.

    Args:
        w_vec (torch.Tensor): Rotation tangent vectors with shape [N, 3] or
            skew-symmetric matrices with shape [N, 3, 3].

    Returns:
        torch.Tensor: Rotation matrices with shape [N, 3, 3].
    """
    if w_vec.shape[1:] == (3, 3):
        w_vec = bracket_so3(w_vec)
    elif w_vec.shape[1:] != (3,):
        raise f"exp_so3: input must be of shape (N, 3) or (N, 3, 3). Current shape: {tuple(w_vec.shape)}"

    R = torch.eye(3).repeat(len(w_vec), 1, 1).to(w_vec)

    # theta: [N], the rotation angle represented by each so(3) vector.
    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS

    w_vec_regular = w_vec[is_regular]
    theta_regular = theta[is_regular]

    theta_regular = theta_regular.unsqueeze(1)

    w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)

    theta_regular = theta_regular.unsqueeze(2)

    # Rodrigues formula. Shape: [N_regular, 3, 3].
    R[is_regular] = torch.eye(3).repeat(len(w_vec_regular), 1, 1).to(w_vec_regular) + torch.sin(theta_regular) * w_mat_hat_regular + (1 - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

    return R


def exp_se3(S):
    """
    Map batched se(3) vectors or matrices to SE(3) transforms.

    Args:
        S (torch.Tensor): Twist vectors with shape [N, 6] or se(3) matrices
            with shape [N, 4, 4].

    Returns:
        torch.Tensor: Homogeneous transforms with shape [N, 4, 4].
    """
    if S.shape[1:] == (4, 4):
        S = bracket_se3(S)
    elif S.shape[1:] != (6,):
        raise f"exp_se3: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    w_vec = S[:, :3]
    p = S[:, 3:]

    T = torch.eye(4).repeat(len(S), 1, 1).to(S)

    theta = w_vec.norm(dim=1)

    is_regular = theta > EPS
    is_singular = theta <= EPS

    w_vec_regular = w_vec[is_regular]
    theta_regular = theta[is_regular]

    theta_regular = theta_regular.unsqueeze(1)

    w_mat_hat_regular = bracket_so3(w_vec_regular / theta_regular)

    theta_regular = theta_regular.unsqueeze(2)

    # G maps the translational twist component into SE(3) translation.
    G = theta_regular * torch.eye(3).repeat(len(S), 1, 1).to(S) + (1 - torch.cos(theta_regular)) * w_mat_hat_regular + (theta_regular - torch.cos(theta_regular)) * w_mat_hat_regular @ w_mat_hat_regular

    T[is_regular, :3, :3] = exp_so3(w_vec_regular)
    T[is_regular, :3, 3] = torch.einsum('nij,nj->ni', G, p)

    T[is_singular, :3, :3] = torch.eye(3).repeat(is_singular.sum(), 1, 1)
    T[is_singular, :3, 3] = p

    return T


def large_adjoint(T):
    """
    Compute the SE(3) adjoint matrix for each transform.

    Args:
        T (torch.Tensor): Valid homogeneous transforms with shape [N, 4, 4].

    Returns:
        torch.Tensor: Large adjoint matrices with shape [N, 6, 6].
    """
    assert T.shape[1:] == (4, 4), f"large_adjoint: input must be of shape (N, 4, 4). Current shape: {tuple(T.shape)}"
    assert is_SE3(T), "large_adjoint: input must be SE(3) matrices"

    R = T[:, :3, :3]
    p = T[:, :3, 3]

    large_adj = T.new_zeros(len(T), 6, 6)
    large_adj[:, :3, :3] = R
    # Lower-left block is [p]x R. Shape: [N, 3, 3].
    large_adj[:, 3:, :3] = bracket_so3(p) @ R
    large_adj[:, 3:, 3:] = R

    return large_adj


def small_adjoint(S):
    """
    Compute the Lie algebra adjoint matrix for each se(3) element.

    Args:
        S (torch.Tensor): Twist vectors with shape [N, 6] or se(3) matrices
            with shape [N, 4, 4].

    Returns:
        torch.Tensor: Small adjoint matrices with shape [N, 6, 6].
    """
    if S.shape[1:] == (4, 4):
        w_mat = S[:, :3, :3]
        v_mat = bracket_so3(S[:, :3, 3])
    elif S.shape[1:] == (6,):
        w_mat = bracket_so3(S[:, :3])
        v_mat = bracket_so3(S[:, 3:])
    else:
        raise f"small_adj: input must be of shape (N, 6) or (N, 4, 4). Current shape: {tuple(S.shape)}"

    small_adj = S.new_zeros(len(S), 6, 6)
    small_adj[:, :3, :3] = w_mat
    # The lower-left block couples translational and angular components.
    small_adj[:, 3:, :3] = v_mat
    small_adj[:, 3:, 3:] = w_mat

    return small_adj


def Lie_bracket(u, v):
    """
    Compute the matrix Lie bracket [u, v] = uv - vu.

    Args:
        u (torch.Tensor): Batched so(3), se(3), or matrix elements with shape
            [N, 3], [N, 6], [N, 3, 3], or [N, 4, 4].
        v (torch.Tensor): Batched elements compatible with u after bracket
            conversion.

    Returns:
        torch.Tensor: Matrix Lie brackets with shape [N, 3, 3] or [N, 4, 4].
    """
    if u.shape[1:] == (3,):
        u = bracket_so3(u)
    elif u.shape[1:] == (6,):
        u = bracket_se3(u)

    if v.shape[1:] == (3,):
        v = bracket_so3(v)
    elif v.shape[1:] == (6,):
        v = bracket_se3(v)

    return u @ v - v @ u


def is_quat(quat):
    """
    Check whether a batch of quaternions has unit norm.

    Args:
        quat (torch.Tensor): Quaternion candidates with shape [N, 4].

    Returns:
        bool: True when every quaternion has norm one.
    """
    test = torch.allclose(quat.norm(dim=1), quat.new_ones(len(quat)))

    return test


def super_fibonacci_spiral(num_Rs):
    """
    Generate approximately uniform SO(3) rotations using a super-Fibonacci spiral.

    Args:
        num_Rs (int): Number of rotations to sample.

    Returns:
        numpy.ndarray: Rotation matrices with shape [num_Rs, 3, 3].
    """
    phi = 1.414213562304880242096980    # sqrt(2)
    psi = 1.533751168755204288118041

    s = np.arange(num_Rs) + 1 / 2

    t = s / num_Rs
    d = 2 * np.pi * s

    r = np.sqrt(t)
    R = np.sqrt(1 - t)

    alpha = d / phi
    beta = d / psi

    quats = np.stack([r * np.sin(alpha), r * np.cos(alpha), R * np.sin(beta), R * np.cos(beta)], axis=1)

    # scipy expects quaternion layout [x, y, z, w]. Shape: [num_Rs, 4].
    Rs = Rotation.from_quat(quats).as_matrix()

    return Rs


def SE3_geodesic_dist(T_1, T_2):
    """
    Compute a simple batched distance between two SE(3) transform batches.

    Args:
        T_1 (torch.Tensor): First transform batch with shape [N, 4, 4].
        T_2 (torch.Tensor): Second transform batch with shape [N, 4, 4].

    Returns:
        torch.Tensor: Per-pair distances with shape [N].
    """
    assert len(T_1) == len(T_2), f"SE3_geodesic_dist: inputs must have the same batch_size. Current shapes: T_1 - {tuple(T_1.shape)}, T_2 - {tuple(T_2.shape)}"
    assert is_SE3(T_1) and is_SE3(T_2), "SE3_geodesic_dist: inputs must be SE(3) matrices"

    R_1 = T_1[:, :3, :3]
    R_2 = T_2[:, :3, :3]
    p_1 = T_1[:, :3, 3]
    p_2 = T_2[:, :3, 3]

    delta_R = bracket_so3(log_SO3(torch.einsum('bij,bjk->bik', inv_SO3(R_1), R_2)))
    delta_p = p_1 - p_2

    # Combine rotation-vector and translation residuals. Shape: [N].
    dist = (delta_R ** 2 + delta_p ** 2).sum(1).sqrt()

    return dist


def get_fibonacci_sphere(num_points):
    """
    Generate approximately uniform points on the unit sphere.

    Args:
        num_points (int): Number of points to sample.

    Returns:
        numpy.ndarray: Unit-sphere points with shape [num_points, 3].
    """
    points = []

    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points += [np.array([x, y, z])]

    points = np.stack(points)

    return points
