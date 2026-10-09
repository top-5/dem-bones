export interface WeightSolveInput {
  restPositions: Float64Array;
  targetPositions: Float64Array;
  transforms: Float64Array;
  vertexCount: number;
  frameCount: number;
  boneCount: number;
  maxInfluences?: number;
  iterations?: number;
  smoothness?: number;
}

export interface WeightSolveResult {
  /** Dense vertex-major weights: weights[vertex * boneCount + bone]. */
  weights: Float64Array;
  vertexCount: number;
  boneCount: number;
  solveMilliseconds: number;
}

export interface WeightSolverBackend {
  solveWeights(input: WeightSolveInput): WeightSolveResult | Promise<WeightSolveResult>;
}

export function validateWeightSolveInput(input: WeightSolveInput): void {
  for (const [name, value] of [["vertexCount", input.vertexCount], ["frameCount", input.frameCount], ["boneCount", input.boneCount]] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive integer`);
  }
  const expect = (name: string, actual: number, wanted: number) => {
    if (actual !== wanted) throw new RangeError(`${name} has ${actual} values; expected ${wanted}`);
  };
  expect("restPositions", input.restPositions.length, input.vertexCount * 3);
  expect("targetPositions", input.targetPositions.length, input.frameCount * input.vertexCount * 3);
  expect("transforms", input.transforms.length, input.frameCount * input.boneCount * 16);
  if (input.maxInfluences !== undefined && (!Number.isSafeInteger(input.maxInfluences) || input.maxInfluences < 1 || input.maxInfluences > input.boneCount))
    throw new RangeError("maxInfluences must be an integer in [1, boneCount]");
  if (input.iterations !== undefined && (!Number.isSafeInteger(input.iterations) || input.iterations < 1))
    throw new RangeError("iterations must be a positive integer");
  if (input.smoothness !== undefined && (!Number.isFinite(input.smoothness) || input.smoothness < 0))
    throw new RangeError("smoothness must be finite and non-negative");
}
