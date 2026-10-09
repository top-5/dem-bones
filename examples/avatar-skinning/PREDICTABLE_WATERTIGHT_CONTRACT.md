# Predictable watertight avatar contract

The deformation solver must not be asked to repair stochastic reconstruction
topology. Surface construction and skin-weight solving are separate stages.

## Deterministic surface stage

1. Start from a versioned shoulder-armpit-arm template in an open-arm pose
   (at least 20-25 degrees) or the corresponding fitted body-template region.
2. Transfer semantic triangle labels and source provenance before edits.
3. Wrap the template to the target body and clothing offset, close the fold with
   ordered simple boundary loops, and remesh the affected band to even 3-5 mm
   triangles.
4. Reject self-intersecting or figure-eight rims, diagonals that already exist
   in the surface, and a fold edge that coincides with an existing edge.
5. Derive patch orientation from neighbouring rim edges, not from a radial
   "faces away from bone" heuristic.

## Required gates

- Position-weld concatenated GLB primitives before component analysis; UV seams
  intentionally duplicate vertices.
- Require one connected surface after that weld unless the manifest declares an
  intentional opening such as a foot gap.
- Zero new boundary, non-manifold, or flipped edges relative to the declared
  source; a watertight template output requires all three counts to be zero.
- Stable vertex/triangle ordering or an explicit `from` provenance map.
- Deterministic output hash for identical template, inputs, options and solver
  version. Hardware/backend remain recorded in provenance.
- Raised-arm and dance pose gates include strain, hollow-web and body-collision
  metrics; a static rest-pose pass is insufficient.

## Weight solve order

Wrap/remesh first, then solve or transfer weights on the final topology. Preserve
trusted weights outside the edited band, initialise new vertices from semantic
bone-chain donors, run native Dem Bones for sparse LBS weights, then optionally
refine the affected band with differentiable DQS/collision/strain losses. Running
Dem Bones before remeshing creates weights for vertices that no longer exist and
loses the final adjacency needed by the smoothness term.

The TypeScript/WASM and Python backends share arrays and fixtures, but their roles
remain explicit: native/WASM solves sparse weights; PyTorch supplies differentiable
local refinement; runtime DQS deforms the validated result.
