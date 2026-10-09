"""Run the shared LBS/DQS API against a real skinned GLB without storing it here."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
from pygltflib import GLTF2

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from avatar_skinning import DeformInput, deform


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_array(gltf: GLTF2, index: int) -> np.ndarray:
    accessor = gltf.accessors[index]
    view = gltf.bufferViews[accessor.bufferView]
    dtype = np.dtype(COMPONENT_DTYPES[accessor.componentType]).newbyteorder("<")
    width = TYPE_WIDTHS[accessor.type]
    stride = view.byteStride or dtype.itemsize * width
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    blob = gltf.binary_blob()
    if stride == dtype.itemsize * width:
        result = np.frombuffer(blob, dtype=dtype, count=accessor.count * width, offset=offset)
        return result.reshape(accessor.count, width).copy()
    result = np.empty((accessor.count, width), dtype=dtype)
    for row in range(accessor.count):
        result[row] = np.frombuffer(blob, dtype=dtype, count=width, offset=offset + row * stride)
    return result


def find_skinned_primitive(gltf: GLTF2):
    for node in gltf.nodes:
        if node.mesh is None or node.skin is None:
            continue
        for primitive in gltf.meshes[node.mesh].primitives:
            attrs = primitive.attributes
            if attrs.POSITION is not None and attrs.JOINTS_0 is not None and attrs.WEIGHTS_0 is not None:
                return node, primitive
    raise RuntimeError("no primitive with POSITION, JOINTS_0 and WEIGHTS_0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--provenance", default="local synthetic avatar")
    args = parser.parse_args()

    raw = args.glb.read_bytes()
    gltf = GLTF2().load_binary(str(args.glb))
    node, primitive = find_skinned_primitive(gltf)
    position = accessor_array(gltf, primitive.attributes.POSITION).astype(np.float32)
    joints = accessor_array(gltf, primitive.attributes.JOINTS_0).astype(np.int64)
    weights = accessor_array(gltf, primitive.attributes.WEIGHTS_0).astype(np.float32)
    joints_1 = getattr(primitive.attributes, "JOINTS_1", None)
    weights_1 = getattr(primitive.attributes, "WEIGHTS_1", None)
    if joints_1 is not None and weights_1 is not None:
        joints = np.concatenate((joints, accessor_array(gltf, joints_1)), axis=1)
        weights = np.concatenate((weights, accessor_array(gltf, weights_1)), axis=1)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, weight_sum, out=np.zeros_like(weights), where=weight_sum > 0)

    if len(position) > args.limit:
        chosen = np.linspace(0, len(position) - 1, args.limit, dtype=np.int64)
        position, joints, weights = position[chosen], joints[chosen], weights[chosen]

    skin = gltf.skins[node.skin]
    bone_count = len(skin.joints)
    # Synthetic stress transform for loader/backend equivalence. A future test will
    # ingest an authored clip and compute hierarchical joint-world * inverse-bind.
    transforms = torch.eye(4, dtype=torch.float32).repeat(1, bone_count, 1, 1)
    stressed_bone = int(joints[np.argmax(weights[:, 0]), 0])
    angle = torch.tensor(np.pi / 2, dtype=torch.float32)
    transforms[0, stressed_bone, 0, 0] = torch.cos(angle)
    transforms[0, stressed_bone, 0, 1] = -torch.sin(angle)
    transforms[0, stressed_bone, 1, 0] = torch.sin(angle)
    transforms[0, stressed_bone, 1, 1] = torch.cos(angle)

    inp = DeformInput(
        torch.from_numpy(position)[None],
        torch.from_numpy(weights)[None],
        torch.from_numpy(joints)[None],
        transforms,
    )
    timings = {}
    output = {}
    for method in ("lbs", "dqs"):
        started = time.perf_counter()
        output[method] = deform(inp, method)
        timings[method] = (time.perf_counter() - started) * 1000
        assert torch.isfinite(output[method]).all()

    delta = torch.linalg.vector_norm(output["dqs"] - output["lbs"], dim=-1)
    report = {
        "schema": "avatar-skinning-smoke/1",
        "source": {
            "name": args.glb.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "provenance": args.provenance,
        },
        "mesh": {"verticesTested": len(position), "bones": bone_count, "influences": weights.shape[1]},
        "stress": {"boneIndex": stressed_bone, "rotationDegrees": 90},
        "timingMs": timings,
        "dqsVsLbsMetres": {
            "mean": float(delta.mean()), "p95": float(torch.quantile(delta, 0.95)), "max": float(delta.max())
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
