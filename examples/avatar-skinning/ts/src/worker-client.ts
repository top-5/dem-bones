import { validateWeightSolveInput, type WeightSolveInput, type WeightSolveResult } from "./types.ts";
import { inputTransferables, type SolvePriority, type WorkerRequest, type WorkerResponse } from "./worker-protocol.ts";
export interface WorkerLike {
  postMessage(message: WorkerRequest, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<WorkerResponse>) => void): void;
  removeEventListener(type: "message", listener: (event: MessageEvent<WorkerResponse>) => void): void;
}
export interface SolveHandle { id: number; result: Promise<WeightSolveResult>; cancel(): void; }
export class WeightSolverWorkerClient {
  private nextId = 1;
  private readonly pending = new Map<number, { resolve(value: WeightSolveResult): void; reject(error: Error): void }>();
  private readonly worker: WorkerLike;
  private readonly onMessage = ({ data }: MessageEvent<WorkerResponse>) => {
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    data.kind === "solved" ? pending.resolve(data.result) : pending.reject(new Error(data.error));
  };
  constructor(worker: WorkerLike) { this.worker = worker; worker.addEventListener("message", this.onMessage); }
  solveWeights(input: WeightSolveInput, priority: SolvePriority = "background", transferOwnership = false): SolveHandle {
    validateWeightSolveInput(input);
    const id = this.nextId++;
    const result = new Promise<WeightSolveResult>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.worker.postMessage({ kind: "solve", id, priority, input }, transferOwnership ? inputTransferables(input) : []);
    return { id, result, cancel: () => {
      const pending = this.pending.get(id);
      if (!pending) return;
      this.pending.delete(id);
      this.worker.postMessage({ kind: "cancel", id });
      pending.reject(new DOMException("Weight solve cancelled", "AbortError"));
    }};
  }
  dispose(): void {
    this.worker.removeEventListener("message", this.onMessage);
    for (const pending of this.pending.values()) pending.reject(new Error("Weight solver client disposed"));
    this.pending.clear();
  }
}
