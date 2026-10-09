"""Shared differentiable skinning API for the portability prototype.

The DQS math is adapted from NAVER Anny's Apache-2.0 implementation:
https://github.com/naver/anny/blob/main/src/anny/skinning/skinning.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import roma
import torch


Method = Literal["lbs", "dqs"]


@dataclass(frozen=True)
class DeformInput:
    vertices: torch.Tensor  # [B|1, V, 3]
    weights: torch.Tensor  # [B|1, V, K]
    joints: torch.Tensor  # [B|1, V, K]
    transforms: torch.Tensor  # [B|1, J, 4, 4]


def _broadcast_batch(value: torch.Tensor, batch: int) -> torch.Tensor:
    if value.shape[0] == batch:
        return value
    if value.shape[0] != 1:
        raise ValueError(f"cannot broadcast batch {value.shape[0]} to {batch}")
    return value.expand(batch, *value.shape[1:])


def _validate(inp: DeformInput) -> tuple[torch.Tensor, ...]:
    batch = max(inp.vertices.shape[0], inp.transforms.shape[0])
    vertices = _broadcast_batch(inp.vertices, batch)
    weights = _broadcast_batch(inp.weights, batch)
    joints = _broadcast_batch(inp.joints, batch)
    transforms = _broadcast_batch(inp.transforms, batch)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [B,V,3]")
    if weights.shape != joints.shape or weights.shape[:2] != vertices.shape[:2]:
        raise ValueError("weights/joints must have matching [B,V,K] shapes")
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape [B,J,4,4]")
    if torch.any(joints < 0) or torch.any(joints >= transforms.shape[1]):
        raise ValueError("joint index outside transform table")
    return vertices, weights, joints.long(), transforms


def _gather_transforms(transforms: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    batch = transforms.shape[0]
    batch_index = torch.arange(batch, device=transforms.device)[:, None, None]
    return transforms[batch_index, joints]


def deform_lbs(inp: DeformInput) -> torch.Tensor:
    vertices, weights, joints, transforms = _validate(inp)
    selected = _gather_transforms(transforms, joints)
    blended = torch.sum(weights[..., None, None] * selected, dim=2)
    homogeneous = torch.cat((vertices, torch.ones_like(vertices[..., :1])), dim=-1)
    return torch.einsum("bvij,bvj->bvi", blended, homogeneous)[..., :3]


def _to_dual_quaternion(transforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rigid = roma.Rigid.from_homogeneous(transforms)
    real = roma.rotmat_to_unitquat(rigid.linear)
    translation_quat = torch.cat(
        (0.5 * rigid.translation, torch.zeros_like(rigid.translation[..., :1])), dim=-1
    )
    dual = roma.quat_product(translation_quat, real)
    return real, dual


def deform_dqs(inp: DeformInput) -> torch.Tensor:
    vertices, weights, joints, transforms = _validate(inp)
    real, dual = _to_dual_quaternion(transforms)
    batch = transforms.shape[0]
    batch_index = torch.arange(batch, device=transforms.device)[:, None, None]
    selected_real = real[batch_index, joints]
    selected_dual = dual[batch_index, joints]

    # Quaternions q and -q encode the same rotation. Align every influence to
    # the first nonzero influence before averaging.
    reference_index = torch.argmax((weights > 0).to(torch.int64), dim=-1, keepdim=True)
    reference = torch.gather(
        selected_real, 2, reference_index[..., None].expand(-1, -1, -1, 4)
    )
    signs = torch.where(
        torch.sum(selected_real * reference, dim=-1, keepdim=True) < 0,
        -torch.ones((), dtype=weights.dtype, device=weights.device),
        torch.ones((), dtype=weights.dtype, device=weights.device),
    )
    weighted = weights[..., None] * signs
    mean_real = torch.sum(weighted * selected_real, dim=2)
    mean_dual = torch.sum(weighted * selected_dual, dim=2)
    norm = torch.linalg.vector_norm(mean_real, dim=-1, keepdim=True).clamp_min(1e-12)
    mean_real = mean_real / norm
    mean_dual = mean_dual / norm
    translation = 2.0 * roma.quat_product(mean_dual, roma.quat_conjugation(mean_real))[..., :3]
    return roma.quat_action(mean_real, vertices) + translation


def deform(inp: DeformInput, method: Method) -> torch.Tensor:
    return deform_lbs(inp) if method == "lbs" else deform_dqs(inp)


def recover_weights(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    transforms: torch.Tensor,
    targets: torch.Tensor,
    *,
    method: Method = "dqs",
    steps: int = 400,
    learning_rate: float = 0.08,
) -> torch.Tensor:
    """Small differentiable baseline, not a replacement for Dem Bones.

    It solves only weights for a fixed candidate-joint set and fixed transforms.
    Production work needs spatial smoothing, sparsity and collision/strain losses.
    """
    batch, vertex_count, influence_count = targets.shape[0], vertices.shape[1], joints.shape[2]
    logits = torch.zeros((1, vertex_count, influence_count), dtype=vertices.dtype,
                         device=vertices.device, requires_grad=True)
    optimizer = torch.optim.Adam((logits,), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=-1)
        prediction = deform(DeformInput(vertices, weights, joints, transforms), method)
        loss = torch.mean((prediction - targets) ** 2)
        loss.backward()
        optimizer.step()
    return torch.softmax(logits.detach(), dim=-1)

