import { validateWeightSolveInput, type WeightSolveInput, type WeightSolveResult, type WeightSolverBackend } from "./types.ts";

export interface DemBonesEmscriptenModule {
  HEAPF64: Float64Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _dem_solve_weights(rest: number, targets: number, transforms: number, output: number,
    vertexCount: number, frameCount: number, boneCount: number, maxInfluences: number,
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
    const alloc = (sourceOrLength: Float64Array | number): number => {
      const length = typeof sourceOrLength === "number" ? sourceOrLength : sourceOrLength.length;
      const pointer = module._malloc(length * Float64Array.BYTES_PER_ELEMENT);
      if (!pointer) throw new Error(`WASM allocation failed for ${length} doubles`);
      allocations.push(pointer);
      if (sourceOrLength instanceof Float64Array) module.HEAPF64.set(sourceOrLength, pointer / 8);
      return pointer;
    };
    const started = performance.now();
    try {
      const rest = alloc(input.restPositions);
      const targets = alloc(input.targetPositions);
      const transforms = alloc(input.transforms);
      const outputLength = input.vertexCount * input.boneCount;
      const output = alloc(outputLength);
      const status = module._dem_solve_weights(rest, targets, transforms, output,
        input.vertexCount, input.frameCount, input.boneCount,
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
