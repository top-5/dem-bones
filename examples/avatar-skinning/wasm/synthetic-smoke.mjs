import { pathToFileURL } from "node:url";

const modulePath = process.argv[2];
if (!modulePath) throw new Error("usage: node synthetic-smoke.mjs <dem-bones.js>");
const createModule = (await import(pathToFileURL(modulePath).href)).default;
const module = await createModule();

const vertices = 4;
const frames = 2;
const bones = 2;
const rest = new Float64Array([
  0, 0, 0,
  1, 0, 0,
  2, 0, 0,
  3, 0, 0,
]);
const truth = new Float64Array([0, 0.25, 0.75, 1]);
const faces = new Uint32Array([0, 1, 2, 1, 2, 3]);
const targets = new Float64Array(frames * vertices * 3);
targets.set(rest, 0);
for (let vertex = 0; vertex < vertices; vertex++) {
  targets[vertices * 3 + vertex * 3] = rest[vertex * 3] + truth[vertex];
}
const transforms = new Float64Array(frames * bones * 16);
for (let frame = 0; frame < frames; frame++) {
  for (let bone = 0; bone < bones; bone++) {
    const offset = (frame * bones + bone) * 16;
    transforms[offset] = transforms[offset + 5] = transforms[offset + 10] = transforms[offset + 15] = 1;
  }
}
transforms[((1 * bones + 1) * 16) + 3] = 1;

function allocateF64(sourceOrLength) {
  const length = typeof sourceOrLength === "number" ? sourceOrLength : sourceOrLength.length;
  const pointer = module._malloc(length * 8);
  if (!pointer) throw new Error(`allocation failed: ${length} doubles`);
  if (sourceOrLength instanceof Float64Array) module.HEAPF64.set(sourceOrLength, pointer / 8);
  return pointer;
}
function allocateU32(source) {
  const pointer = module._malloc(source.byteLength);
  if (!pointer) throw new Error(`allocation failed: ${source.length} uint32 values`);
  module.HEAPU32.set(source, pointer / 4);
  return pointer;
}

const pointers = [allocateF64(rest), allocateU32(faces), allocateF64(targets), allocateF64(transforms), allocateF64(vertices * bones)];
const started = performance.now();
const status = module._dem_solve_weights(...pointers, vertices, faces.length / 3, frames, bones, 2, 20, 0);
const elapsed = performance.now() - started;
try {
  if (status !== 0) throw new Error(module.UTF8ToString(module._dem_last_error()));
  const weights = module.HEAPF64.slice(pointers[4] / 8, pointers[4] / 8 + vertices * bones);
  let maxError = 0;
  for (let vertex = 0; vertex < vertices; vertex++) {
    maxError = Math.max(maxError, Math.abs(weights[vertex * bones + 1] - truth[vertex]));
  }
  if (!Number.isFinite(maxError) || maxError >= 1e-4) {
    throw new Error(`WASM recovered incorrect weights: max error ${maxError}`);
  }
  console.log(JSON.stringify({ backend: "wasm-scalar", solveMilliseconds: elapsed, maxWeightError: maxError, weights: [...weights] }));
} finally {
  for (const pointer of pointers.reverse()) module._free(pointer);
}
