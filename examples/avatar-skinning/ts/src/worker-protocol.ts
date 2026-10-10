import type { WeightSolveInput, WeightSolveResult } from "./types.ts";
export type SolvePriority = "hover" | "visible" | "background";
export const priorityValue: Record<SolvePriority, number> = { hover: 0, visible: 1, background: 2 };
export interface SolveRequest { kind: "solve"; id: number; priority: SolvePriority; input: WeightSolveInput; }
export interface CancelRequest { kind: "cancel"; id: number; }
export type WorkerRequest = SolveRequest | CancelRequest;
export type WorkerResponse = { kind: "solved"; id: number; result: WeightSolveResult } | { kind: "failed"; id: number; error: string };
export function inputTransferables(input: WeightSolveInput): Transferable[] {
  return [input.restPositions.buffer, input.triangleIndices.buffer, input.targetPositions.buffer, input.transforms.buffer,
    input.initialWeights?.buffer, input.lockWeights?.buffer]
    .filter((value): value is ArrayBuffer => value instanceof ArrayBuffer);
}
