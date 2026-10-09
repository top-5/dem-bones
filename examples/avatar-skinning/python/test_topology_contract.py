from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from topology_contract import inspect_topology


def main() -> None:
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    closed = inspect_topology(vertices, faces)
    assert closed.watertight and closed.connected_components == 1

    seam_vertices = np.concatenate((vertices, vertices[[0]]), axis=0)
    seam_faces = faces.copy()
    seam_faces[0, 0] = 4
    seam = inspect_topology(seam_vertices, seam_faces)
    assert seam.watertight and seam.welded_vertex_count == 4

    opened = inspect_topology(vertices, faces[:-1])
    assert not opened.watertight and opened.boundary_edges == 3
    print("topology contract: closed, UV-welded, and open-surface checks passed")


if __name__ == "__main__":
    main()
