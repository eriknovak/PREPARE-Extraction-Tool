import { apiRequest } from "./client";
import type {
  ActiveModelResponse,
  DatasetsOutput,
  EvaluationResponse,
  MessageOutput,
  ModelsOutput,
  ModelSummary,
  MonitorDatasetStats,
  MonitorRun,
  RunErrorAnalysis,
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

/** Aggregated record/term counts and label distribution across multiple datasets. */
export function getMultiDatasetStats(datasetIds: number[]) {
  return apiRequest<MonitorDatasetStats>("/bioner/datasets/full-stats", {
    method: "POST",
    body: JSON.stringify({ dataset_ids: datasetIds }),
  });
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

/** Per-label error analysis (confusion counts + example errors) for a run. */
export function getRunErrorAnalysis(runId: number) {
  return apiRequest<RunErrorAnalysis>(`/bioner/runs/${runId}/error-analysis`);
}

/** Alias kept for the heatmap view; uses the per-dataset evaluations endpoint. */
export function getAllEvaluations(datasetId: number) {
  return getAllRunEvaluations(datasetId);
}

/** Per-epoch loss curve for a single run, ordered by epoch. */
export function getRunMetrics(runId: number) {
  return apiRequest<TrainingMetric[]>(`/bioner/runs/${runId}/metrics`);
}

/* ---------------- MODELS ---------------- */

/** List trained models the user can select for NER extraction. */
export async function getModels(): Promise<ModelSummary[]> {
  const res = await apiRequest<ModelsOutput>("/bioner/models");
  return res.models;
}

/** The model a dataset currently uses for extraction (null active_model = default). */
export function getDatasetActiveModel(datasetId: number) {
  return apiRequest<ActiveModelResponse>(`/bioner/datasets/${datasetId}/active-model`);
}

/** Set (modelId) or clear (null = default) the dataset's active extraction model. */
export function setDatasetActiveModel(datasetId: number, modelId: number | null) {
  return apiRequest<ActiveModelResponse>(`/bioner/datasets/${datasetId}/active-model`, {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

/* ---------------- TRAINING ---------------- */

export function startTraining(payload: {
  dataset_ids: number[];
  eval_dataset_ids: number[];
  labels: string[];
  base_model: string;
  val_ratio: number;
  num_epochs: number;
  learning_rate: number;
  train_batch_size: number;
}) {
  return apiRequest<{ run_id: number }>("/bioner/training/start", {
    method: "POST",
    body: JSON.stringify({
      dataset_ids: payload.dataset_ids,
      eval_dataset_ids: payload.eval_dataset_ids,
      labels: payload.labels,
      base_model: payload.base_model,
      val_ratio: payload.val_ratio,
      num_epochs: payload.num_epochs,
      learning_rate: payload.learning_rate,
      train_batch_size: payload.train_batch_size,
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
  // NOTE: The access token is passed via the URL query string because the
  // browser WebSocket API cannot set custom request headers. The backend reads
  // `token` from the query. This is a known limitation: tokens in URLs can leak
  // into server/proxy access logs and browser history. encodeURIComponent keeps
  // the value safely encoded; a more robust approach would be a short-lived
  // one-time ticket or a subprotocol-based handshake instead of a long-lived
  // token in the URL.
  return `${base}/api/v1/bioner/ws/training?token=${encodeURIComponent(token)}`;
}
