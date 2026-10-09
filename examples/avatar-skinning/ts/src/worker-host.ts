import type { WeightSolverBackend } from "./types.ts";
import type { WorkerRequest, WorkerResponse } from "./worker-protocol.ts";
import { WeightSolverWorkerRuntime } from "./worker-runtime.ts";

export interface WorkerScopeLike {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<WorkerRequest>) => void): void;
}

/** Installs the protocol on a DedicatedWorkerGlobalScope after its WASM backend is loaded. */
export function installWeightSolverWorker(scope: WorkerScopeLike, backend: WeightSolverBackend): WeightSolverWorkerRuntime {
  const runtime = new WeightSolverWorkerRuntime(backend, (message, transfer) => scope.postMessage(message, transfer));
  scope.addEventListener("message", event => runtime.accept(event.data));
  return runtime;
}
