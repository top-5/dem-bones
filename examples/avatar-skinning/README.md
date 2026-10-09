# Avatar skinning portability prototype

This prototype separates two operations which are easy to conflate:

1. `deform`: apply known weights and bone transforms with LBS or DQS;
2. `solve_weights`: infer sparse convex weights from corresponding posed meshes
   and known bone transforms.

Dem Bones implements the second operation (and can also solve transforms). The
NAVER Anny reference implements differentiable forward LBS/DQS, which makes it a
good foundation for a PyTorch weight optimizer but not a drop-in Dem Bones solver.

The public fork contains synthetic tests only. Do not add private avatar assets.

## Shared array contract

All adapters use row-major arrays:

- `vertices`: `[vertex, xyz]`
- `faces`: `[triangle, corner]`
- `weights`: `[vertex, influence]`
- `joints`: `[vertex, influence]`
- `transforms`: `[frame, bone, 4, 4]`
- `targets`: `[frame, vertex, xyz]`

Transforms map rest/object space to posed/object space. Weights are non-negative,
sum to one per vertex, and are pruned to a configured influence count only after
optimization.

## Python / NAVER-derived DQS smoke

The implementation in `python/avatar_skinning.py` is an independently packaged,
attributed adaptation of NAVER Anny's Apache-2.0 DQS reference. Run:

```powershell
uv run --with torch --with roma python examples/avatar-skinning/python/test_avatar_skinning.py
```

It validates identity, rigid translation, quaternion-sign invariance, normalized
weights, gradients, and a small weight-recovery problem through the same API used
by future native/WASM adapters.

Run the same deformation API on a local synthetic avatar without adding the GLB:

```powershell
uv run --with torch --with roma --with numpy --with pygltflib `
  python examples/avatar-skinning/python/glb_smoke.py C:\path\to\avatar.glb
```

## GLB ingestion

Keep file parsing outside the numerical solver:

- TypeScript/browser: `@gltf-transform/core` or Babylon's loader;
- Python/Seed: `trimesh` for GLB primitives, skin attributes and scenes;
- C++/WASM: `cgltf` is the preferred small C99 reader; `tinygltf` is a larger
  header-only C++ alternative;
- Open3D is useful for geometry processing but is not the right dependency merely
  to decode glTF skins/animations;
- OpenCV has no role in GLB parsing.

The importer converts GLB buffers into this array contract. The solver never owns
materials, images, accessors or scene graphs.

## Native Dem Bones smoke

The native test creates a two-bone, two-frame synthetic mesh sequence, locks the
known transforms, solves its weights, and checks them against ground truth:

```powershell
cmake -S examples/avatar-skinning/native -B .build/avatar-skinning-native
cmake --build .build/avatar-skinning-native --config Release
.\.build\avatar-skinning-native\Release\dem_bones_avatar_smoke.exe
```

## TypeScript versus WASM

A literal TypeScript port of Dem Bones would need sparse matrices, constrained
least squares, repeated SVD/polar decomposition and iterative clustering. It is
possible, but would duplicate mature Eigen code and be difficult to keep numerically
equivalent. TypeScript is appropriate for the adapter, orchestration and small DQS
reference; production solving should use native C++ offline and a focused WASM build
when browser-local solving is required.

The planned WASM surface is intentionally small:

```ts
solveWeights(input: WeightSolveInput): WeightSolveResult
deform(input: DeformInput): Float32Array
```

Compile without OpenMP first. Run it in a Web Worker. Add pthread/SIMD builds only
after deterministic scalar tests pass.

The TypeScript client, priority worker queue, Emscripten heap adapter, and C ABI
live in `ts/` and `wasm/`. Their interface tests do not require emsdk:

```powershell
cd examples/avatar-skinning/ts
npm test
```

The deterministic surface/solver boundary and its validation gates are specified
in [`PREDICTABLE_WATERTIGHT_CONTRACT.md`](PREDICTABLE_WATERTIGHT_CONTRACT.md).
