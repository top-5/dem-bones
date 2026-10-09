# Dem Bones WASM adapter

The core solver is header-only C++/Eigen, so the preferred browser port is a
thin Emscripten binding rather than a line-by-line TypeScript rewrite.

Implemented baseline:

1. build Eigen + Dem Bones without OpenMP;
2. copy typed arrays for rest vertices, target frames and fixed transforms;
3. expose `computeWeights()` and return dense vertex-major weights;
4. schedule solves in a priority worker queue (`hover`, `visible`, `background`).

```sh
emcmake cmake -S examples/avatar-skinning/wasm -B .build/avatar-skinning-wasm
cmake --build .build/avatar-skinning-wasm --config Release
node examples/avatar-skinning/wasm/synthetic-smoke.mjs "$PWD/.build/avatar-skinning-wasm/dem-bones.js"
```

The native and WASM implementations must consume the same row-major array
contract documented in the parent README. A binding that accepts one giant JSON
string is intentionally avoided because it adds parsing and copy overhead.

Cancellation removes queued work and suppresses the result of in-flight work.
The current Dem Bones API has no iteration callback, so an in-flight solve is not
preemptible. SIMD and pthread builds remain separate follow-ups; pthreads require
cross-origin isolation and must not be an accidental runtime dependency.

CI executes the compiled scalar module under Node against the same locked-transform
two-frame recovery problem as the native test. `node` is included only as a test
host; the emitted ES module also supports browsers and dedicated workers.
