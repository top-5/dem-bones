# Dem Bones WASM adapter plan

The core solver is header-only C++/Eigen, so the preferred browser port is a
thin Emscripten binding rather than a line-by-line TypeScript rewrite.

Milestone order:

1. build Eigen + Dem Bones without OpenMP;
2. expose typed-array setters for rest vertices, topology, target frames, fixed
   transforms and optional initial weights;
3. expose `computeWeights()` and return sparse weights as joint/weight arrays;
4. validate against the native solver with the synthetic fixtures in this folder;
5. enable WASM SIMD; benchmark pthreads only in cross-origin-isolated workers;
6. connect `cgltf` only in a higher adapter layer, not inside the solver.

The native and WASM implementations must consume the same row-major array
contract documented in the parent README. A binding that accepts one giant JSON
string is intentionally avoided because it adds parsing and copy overhead.

