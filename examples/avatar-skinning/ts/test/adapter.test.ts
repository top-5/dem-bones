import assert from "node:assert/strict";
import test from "node:test";
import { EmscriptenWeightSolver, type DemBonesEmscriptenModule } from "../src/emscripten-backend.ts";
import { WeightSolverWorkerRuntime } from "../src/worker-runtime.ts";
import type { WeightSolveInput, WeightSolverBackend } from "../src/types.ts";
import { inputTransferables } from "../src/worker-protocol.ts";

function input(): WeightSolveInput {
  return { vertexCount: 3, frameCount: 1, boneCount: 2, restPositions: new Float64Array(9),
    triangleIndices: new Uint32Array([0, 1, 2]),
    targetPositions: new Float64Array(9), transforms: new Float64Array(32) };
}

test("Emscripten adapter copies output and frees every allocation", () => {
  const heap = new Float64Array(1024);
  let cursor = 8;
  const freed: number[] = [];
  const module: DemBonesEmscriptenModule = {
    HEAPF64: heap,
    HEAPU32: new Uint32Array(heap.buffer),
    _malloc(bytes) { const pointer = cursor; cursor += bytes; return pointer; },
    _free(pointer) { freed.push(pointer); },
    _dem_solve_weights(_r, _f, _t, _m, _initial, _locks, output) { heap.set([1, 0, .25, .75, .5, .5], output / 8); return 0; },
  };
  const result = new EmscriptenWeightSolver(module).solveWeights(input());
  assert.deepEqual([...result.weights], [1, 0, .25, .75, .5, .5]);
  assert.equal(freed.length, 5);
});

test("Emscripten adapter transfers initial weights and topology-band locks", () => {
  const heap = new Float64Array(2048); let cursor=8; const freed:number[]=[];
  let observed:number[]=[];
  const module: DemBonesEmscriptenModule={HEAPF64:heap,HEAPU32:new Uint32Array(heap.buffer),
    _malloc(bytes){const pointer=cursor;cursor+=bytes;return pointer;},_free(pointer){freed.push(pointer);},
    _dem_solve_weights(_r,_f,_t,_m,initial,locks,output,_vc,_fc,_frames,_bones){
      observed=[...heap.slice(initial/8,initial/8+6),...heap.slice(locks/8,locks/8+3)];
      heap.set([1,0,.5,.5,0,1],output/8);return 0;
    }};
  const value=input(); value.initialWeights=new Float64Array([1,0,.75,.25,0,1]); value.lockWeights=new Float64Array([1,0,1]);
  new EmscriptenWeightSolver(module).solveWeights(value);
  assert.deepEqual(observed,[1,0,.75,.25,0,1,1,0,1]);
  assert.equal(freed.length,7);
});

test("worker queue promotes hover over queued background work", async () => {
  const release: Array<() => void> = [];
  const backend: WeightSolverBackend = { async solveWeights(value) {
    await new Promise<void>(resolve => release.push(resolve));
    return { weights: new Float64Array([value.restPositions[0]]), vertexCount: 1, boneCount: 1, solveMilliseconds: 0 };
  }};
  const outputs: number[] = [];
  const runtime = new WeightSolverWorkerRuntime(backend, message => {
    if (message.kind === "solved") outputs.push(message.result.weights[0]);
  });
  const make = (value: number): WeightSolveInput => ({ vertexCount: 1, frameCount: 1, boneCount: 1,
    restPositions: new Float64Array([value, 0, 0, value, 1, 0, value, 0, 1]),
    triangleIndices: new Uint32Array([0, 1, 2]), targetPositions: new Float64Array(9), transforms: new Float64Array(16) });
  runtime.accept({ kind: "solve", id: 1, priority: "background", input: make(1) });
  runtime.accept({ kind: "solve", id: 2, priority: "background", input: make(2) });
  runtime.accept({ kind: "solve", id: 3, priority: "hover", input: make(3) });
  for (let i = 0; i < 3; ++i) {
    release.shift()!();
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  assert.deepEqual(outputs, [1, 3, 2]);
});

test("worker transfer includes topology and optional constraint arrays",()=>{
  const value=input(); value.initialWeights=new Float64Array(6); value.lockWeights=new Float64Array(3);
  const transferred=inputTransferables(value);
  assert.equal(transferred.length,6);
  assert.ok(transferred.includes(value.triangleIndices.buffer));
  assert.ok(transferred.includes(value.initialWeights.buffer));
  assert.ok(transferred.includes(value.lockWeights.buffer));
});
