import type { WeightSolverBackend } from "./types.ts";
import { priorityValue, type SolveRequest, type WorkerRequest, type WorkerResponse } from "./worker-protocol.ts";
interface Pending extends SolveRequest { sequence: number; }

/** Serial priority queue. Running native solves are intentionally not advertised as preemptible. */
export class WeightSolverWorkerRuntime {
  private readonly queue: Pending[] = [];
  private readonly cancelled = new Set<number>();
  private running = false;
  private sequence = 0;
  private readonly backend: WeightSolverBackend;
  private readonly send: (message: WorkerResponse, transfer?: Transferable[]) => void;
  constructor(backend: WeightSolverBackend, send: (message: WorkerResponse, transfer?: Transferable[]) => void) {
    this.backend = backend;
    this.send = send;
  }
  accept(message: WorkerRequest): void {
    if (message.kind === "cancel") { this.cancelled.add(message.id); return; }
    this.queue.push({ ...message, sequence: this.sequence++ });
    this.queue.sort((a, b) => priorityValue[a.priority] - priorityValue[b.priority] || a.sequence - b.sequence);
    void this.drain();
  }
  private async drain(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.queue.length) {
        const request = this.queue.shift()!;
        if (this.cancelled.delete(request.id)) continue;
        try {
          const result = await this.backend.solveWeights(request.input);
          if (!this.cancelled.delete(request.id)) this.send({ kind: "solved", id: request.id, result }, [result.weights.buffer]);
        } catch (error) {
          this.send({ kind: "failed", id: request.id, error: error instanceof Error ? error.message : String(error) });
        }
      }
    } finally { this.running = false; }
  }
}
