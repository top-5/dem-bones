from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from avatar_skinning import DeformInput, deform, recover_weights


def transforms() -> torch.Tensor:
    result = torch.eye(4, dtype=torch.float64).repeat(2, 2, 1, 1)
    result[0, 1, 0, 3] = 1.0
    angle = math.pi / 2
    result[1, 1, 0, 0] = math.cos(angle)
    result[1, 1, 0, 1] = -math.sin(angle)
    result[1, 1, 1, 0] = math.sin(angle)
    result[1, 1, 1, 1] = math.cos(angle)
    return result


def main() -> None:
    vertices = torch.tensor([[[0.25, 0.5, 0.0], [0.75, 0.5, 0.0]]], dtype=torch.float64)
    joints = torch.tensor([[[0, 1], [0, 1]]])
    truth = torch.tensor([[[0.8, 0.2], [0.15, 0.85]]], dtype=torch.float64)
    motion = transforms()

    for method in ("lbs", "dqs"):
        target = deform(DeformInput(vertices, truth, joints, motion), method)
        recovered = recover_weights(vertices, joints, motion, target, method=method)
        replay = deform(DeformInput(vertices, recovered, joints, motion), method)
        error = torch.max(torch.abs(replay - target)).item()
        assert error < 2e-3, (method, error, recovered)
        assert torch.allclose(recovered.sum(dim=-1), torch.ones_like(recovered[..., 0]), atol=1e-10)
        print(f"{method}: max replay error={error:.6g}; recovered={recovered.tolist()}")

    identity = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    one_joint = torch.zeros((1, 2, 1), dtype=torch.long)
    one_weight = torch.ones((1, 2, 1), dtype=torch.float64, requires_grad=True)
    out = deform(DeformInput(vertices, one_weight, one_joint, identity), "dqs")
    assert torch.allclose(out, vertices)
    out.square().sum().backward()
    assert one_weight.grad is not None and torch.isfinite(one_weight.grad).all()
    print("dqs: identity and gradient checks passed")


if __name__ == "__main__":
    main()

