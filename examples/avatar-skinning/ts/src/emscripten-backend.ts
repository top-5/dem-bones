import { validateWeightSolveInput, type WeightSolveInput, type WeightSolveResult, type WeightSolverBackend } from "./types.ts";

export interface DemBonesEmscriptenModule {
  HEAPF64: Float64Array;
  HEAPU32: Uint32Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _dem_solve_weights(rest: number, faces: number, targets: number, transforms: number, initialWeights: number,
    lockWeights: number, output: number,
    vertexCount: number, faceCount: number, frameCount: number, boneCount: number, maxInfluences: number,
    iterations: number, smoothness: number): number;
  UTF8ToString?(pointer: number): string;
  _dem_last_error?(): number;
}

/** Owns all temporary Emscripten heap allocations around the C ABI. */
export class EmscriptenWeightSolver implements WeightSolverBackend {
  private readonly module: DemBonesEmscriptenModule;
  constructor(module: DemBonesEmscriptenModule) { this.module = module; }

  solveWeights(input: WeightSolveInput): WeightSolveResult {
    validateWeightSolveInput(input);
    const module = this.module;
    const allocations: number[] = [];
    const allocF64 = (sourceOrLength: Float64Array | number): number => {
      const length = typeof sourceOrLength === "number" ? sourceOrLength : sourceOrLength.length;
      const pointer = module._malloc(length * Float64Array.BYTES_PER_ELEMENT);
      if (!pointer) throw new Error(`WASM allocation failed for ${length} doubles`);
      allocations.push(pointer);
      if (sourceOrLength instanceof Float64Array) module.HEAPF64.set(sourceOrLength, pointer / 8);
      return pointer;
    };
    const allocU32 = (source: Uint32Array): number => {
      const pointer = module._malloc(source.byteLength);
      if (!pointer) throw new Error(`WASM allocation failed for ${source.length} uint32 values`);
      allocations.push(pointer); module.HEAPU32.set(source, pointer / 4); return pointer;
    };
    const started = performance.now();
    try {
      const rest = allocF64(input.restPositions);
      const faces = allocU32(input.triangleIndices);
      const targets = allocF64(input.targetPositions);
      const transforms = allocF64(input.transforms);
      const initialWeights = input.initialWeights ? allocF64(input.initialWeights) : 0;
      const lockWeights = input.lockWeights ? allocF64(input.lockWeights) : 0;
      const outputLength = input.vertexCount * input.boneCount;
      const output = allocF64(outputLength);
      const status = module._dem_solve_weights(rest, faces, targets, transforms, initialWeights, lockWeights, output,
        input.vertexCount, input.triangleIndices.length / 3, input.frameCount, input.boneCount,
        input.maxInfluences ?? Math.min(4, input.boneCount), input.iterations ?? 20, input.smoothness ?? 0);
      if (status !== 0) {
        const errorPointer = module._dem_last_error?.() ?? 0;
        const detail = errorPointer && module.UTF8ToString ? module.UTF8ToString(errorPointer) : `status ${status}`;
        throw new Error(`Dem Bones solve failed: ${detail}`);
      }
      const weights = module.HEAPF64.slice(output / 8, output / 8 + outputLength);
      return { weights, vertexCount: input.vertexCount, boneCount: input.boneCount, solveMilliseconds: performance.now() - started };
    } finally {
      for (let i = allocations.length - 1; i >= 0; --i) module._free(allocations[i]);
    }
  }
}
