"""Deterministic topology checks for avatar surface/template processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopologyReport:
    vertex_count: int
    face_count: int
    welded_vertex_count: int
    degenerate_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    connected_components: int
    watertight: bool


def weld_by_position(vertices: np.ndarray, faces: np.ndarray, decimals: int = 6):
    """Weld after primitive concatenation; UV seams are not surface components."""
    unique, inverse = np.unique(np.round(vertices, decimals), axis=0, return_inverse=True)
    welded_faces = inverse[np.asarray(faces, dtype=np.int64)]
    keep = np.logical_and.reduce(
        (
            welded_faces[:, 0] != welded_faces[:, 1],
            welded_faces[:, 1] != welded_faces[:, 2],
            welded_faces[:, 2] != welded_faces[:, 0],
        )
    )
    return unique, welded_faces[keep], int(np.count_nonzero(~keep))


def inspect_topology(vertices: np.ndarray, faces: np.ndarray) -> TopologyReport:
    welded_vertices, welded_faces, degenerates = weld_by_position(vertices, faces)
    edges = np.sort(
        np.concatenate(
            (welded_faces[:, [0, 1]], welded_faces[:, [1, 2]], welded_faces[:, [2, 0]]),
            axis=0,
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    parent = np.arange(len(welded_vertices))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in unique_edges:
        a, b = find(int(left)), find(int(right))
        if a != b:
            parent[b] = a
    used = np.unique(welded_faces)
    components = len({find(int(value)) for value in used}) if len(used) else 0
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    return TopologyReport(
        vertex_count=int(len(vertices)),
        face_count=int(len(faces)),
        welded_vertex_count=int(len(welded_vertices)),
        degenerate_faces=degenerates,
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        connected_components=components,
        watertight=boundary == 0 and nonmanifold == 0 and components == 1,
    )
