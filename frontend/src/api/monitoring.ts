import { apiRequest } from "./client";
import type {
  DatasetsOutput,
  EvaluationResponse,
  MessageOutput,
  MonitorDatasetStats,
  MonitorRun,
  RunsOutput,
  RunUpdate,
  TrainingMetric,
} from "types";

// Note: auth is injected automatically by `apiRequest` from localStorage.

/* ---------------- DATASETS ---------------- */

export function getDatasets(page = 1, limit = 50) {
  return apiRequest<DatasetsOutput>(`/datasets/?page=${page}&limit=${limit}`);
}

export function getDatasetStats(datasetId: number) {
  return apiRequest<MonitorDatasetStats>(`/bioner/datasets/${datasetId}/full-stats`);
}

/* ---------------- RUNS ---------------- */

/** Newest page of runs as a flat array (used by the provider for overlays/pickers). */
export async function getDatasetRuns(datasetId: number, page = 1, limit = 20) {
  const res = await apiRequest<RunsOutput>(`/bioner/datasets/${datasetId}/runs?page=${page}&limit=${limit}`);
  return res.runs;
}

/** Paginated runs (with pagination metadata) for the comparison run table. */
export function getDatasetRunsPaged(datasetId: number, page = 1, limit = 20) {
  return apiRequest<RunsOutput>(`/bioner/datasets/${datasetId}/runs?page=${page}&limit=${limit}`);
}

/** Rename a run and/or mark it as the dataset's preferred run. */
export function updateRun(runId: number, payload: RunUpdate) {
  return apiRequest<MonitorRun>(`/bioner/runs/${runId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

/** Delete a run and its dependent metrics/model/evaluation rows. */
export function deleteRun(runId: number) {
  return apiRequest<MessageOutput>(`/bioner/runs/${runId}`, {
    method: "DELETE",
  });
}

export function getRunEvaluation(runId: number) {
  return apiRequest<EvaluationResponse>(`/bioner/runs/${runId}/evaluation`);
}

export function getAllRunEvaluations(datasetId: number) {
  return apiRequest<EvaluationResponse[]>(`/bioner/datasets/${datasetId}/runs/evaluations`);
}

/** Alias kept for the heatmap view; uses the per-dataset evaluations endpoint. */
export function getAllEvaluations(datasetId: number) {
  return getAllRunEvaluations(datasetId);
}

/** Per-epoch loss curve for a single run, ordered by epoch. */
export function getRunMetrics(runId: number) {
  return apiRequest<TrainingMetric[]>(`/bioner/runs/${runId}/metrics`);
}

/* ---------------- TRAINING ---------------- */

export function startTraining(payload: {
  dataset_id: number | null;
  labels: string[];
  base_model: string;
  val_ratio: number;
}) {
  return apiRequest<{ run_id: number }>("/bioner/training/start", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: payload.dataset_id,
      labels: payload.labels,
      base_model: payload.base_model,
      val_ratio: payload.val_ratio,
    }),
  });
}

export function stopTraining(runId: number) {
  return apiRequest<MessageOutput>(`/bioner/training/stop/${runId}`, {
    method: "POST",
  });
}

/* ---------------- WS ---------------- */

export function getTrainingWSUrl(token: string) {
  const backendHost = import.meta.env.VITE_BACKEND_HOST;
  let base: string;
  if (backendHost) {
    base = backendHost.replace(/^http/, "ws");
  } else {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    base = `${proto}://${window.location.host}`;
  }
  return `${base}/api/v1/bioner/ws/training?token=${token}`;
}
