import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { useToast } from "@hooks/useToast";
import {
  getAllEvaluations,
  getDatasetRuns,
  getDatasets,
  getDatasetStats,
  getRunEvaluation,
  getTrainingWSUrl,
  startTraining as apiStartTraining,
  stopTraining as apiStopTraining,
} from "@api/monitoring";
import type {
  EvaluationResponse,
  MonitorDataset,
  MonitorDatasetStats,
  MonitorRun,
  TrainingMetric,
} from "types";

import { DEFAULT_MODEL, MonitorContext } from "../hooks/useMonitor";
import type { MonitorContextValue, MonitorView } from "../hooks/useMonitor";

/** Provides all Monitor state + actions to the page and its views. */
const MonitorProvider = ({ children }: { children: ReactNode }) => {
  const toast = useToast();

  const showAlert = (
    payload: { message?: string; detail?: string; suggestion?: string },
    type: "error" | "success" | "info" = "error"
  ) => {
    const base = payload?.message || payload?.detail || "Unknown error";
    const message = payload?.suggestion ? `${base} — ${payload.suggestion}` : base;
    toast.showToast(message, type);
  };

  const [token] = useState<string | null>(() => localStorage.getItem("access_token"));

  const [activeView, setActiveView] = useState<MonitorView>("comparison");

  const [progress, setProgress] = useState(0);
  const [, setTotalEpochs] = useState(4);

  const [datasets, setDatasets] = useState<MonitorDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationResponse[]>([]);
  const [evaluationsLoading, setEvaluationsLoading] = useState(false);
  const [datasetStats, setDatasetStats] = useState<MonitorDatasetStats | null>(null);

  const [runs, setRuns] = useState<MonitorRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [valSplitRatio, setValSplitRatio] = useState<number>(0.1);

  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);

  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string>("");

  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  // Model selection
  const [baseModel] = useState<string>(DEFAULT_MODEL);
  const [customModel, setCustomModel] = useState<string>("");
  const [useCustomModel, setUseCustomModel] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const totalEpochsRef = useRef(4);
  const [activeRunId, setActiveRunId] = useState<number | null>(null);

  // ------------------ RESET ------------------

  const resetAll = () => {
    setRuns([]);
    setSelectedRun(null);
    setEvaluation(null);
    setTrainingMetrics([]);
    setIsTraining(false);
    setTrainingStatus("");
    setSelectedLabels([]);
    setEvaluations([]);
  };

  // ------------------ DATASETS ------------------

  useEffect(() => {
    if (!token) return;

    const fetchDatasets = async () => {
      try {
        const data = await getDatasets(1, 50);
        setDatasets(data.datasets ?? []);
      } catch (err) {
        console.error(err);
        setDatasets([]);
      }
    };

    fetchDatasets();
  }, [token]);

  const selectDataset = async (id: number) => {
    setSelectedDatasetId(id);
    resetAll();
    setDatasetStats(null);

    try {
      const data = await getDatasetStats(id);
      // guard against stale responses when switching datasets rapidly
      setSelectedDatasetId((current) => {
        if (current === id) setDatasetStats(data);
        return current;
      });
    } catch (err) {
      console.error(err);
      setSelectedDatasetId((current) => {
        if (current === id) setDatasetStats(null);
        return current;
      });
    }
  };

  // ------------------ ALL RUN EVAL (comparison) ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;

    let cancelled = false;
    setEvaluationsLoading(true);
    getAllEvaluations(selectedDatasetId)
      .then((data) => {
        if (cancelled) return;
        setEvaluations(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        if (cancelled) return;
        setEvaluations([]);
      })
      .finally(() => {
        if (cancelled) return;
        setEvaluationsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDatasetId, token]);

  // ------------------ RUNS ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;

    let cancelled = false;

    const fetchRuns = async () => {
      try {
        const data = await getDatasetRuns(selectedDatasetId);
        if (cancelled) return;
        const runsArray = Array.isArray(data) ? data : [];

        setRuns(runsArray);
        setSelectedRun(runsArray?.[0]?.run_id ?? null);
      } catch (e) {
        if (cancelled) return;
        console.error(e);
        setRuns([]);
        setSelectedRun(null);
      }
    };

    fetchRuns();

    return () => {
      cancelled = true;
    };
  }, [selectedDatasetId, token]);

  // ------------------ SINGLE EVAL ------------------

  useEffect(() => {
    if (!selectedRun || !token) {
      setEvaluation(null);
      return;
    }

    let cancelled = false;
    setEvaluationLoading(true);
    getRunEvaluation(selectedRun)
      .then((data) => {
        if (cancelled) return;
        setEvaluation(data);
      })
      .catch(() => {
        if (cancelled) return;
        setEvaluation(null);
      })
      .finally(() => {
        if (cancelled) return;
        setEvaluationLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRun, token]);

  // ------------------ WEBSOCKET ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;
    let closed = false;
    const ws = new WebSocket(getTrainingWSUrl(token));
    wsRef.current = ws;

    ws.onmessage = (event) => {
      if (closed) return;

      let data;
      try {
        data = JSON.parse(event.data);
      } catch (e) {
        console.error("Failed to parse training WS message:", e);
        return;
      }

      switch (data.type) {
        case "training_start":
          setIsTraining(true);
          setTrainingMetrics([]);
          setProgress(0);
          setTrainingStatus("Training started…");

          totalEpochsRef.current = data.num_epochs ?? 4;
          setTotalEpochs(data.num_epochs ?? 4);
          break;

        case "training_info":
          setIsTraining(true);
          setTrainingStatus(`Training started (${data.train_size} samples)`);
          break;

        case "epoch_update": {
          const epoch = Number(data.epoch ?? 0);

          setProgress(() => {
            const safeTotal = totalEpochsRef.current;
            if (safeTotal <= 0) return 0;
            return Math.min(100, (epoch / safeTotal) * 100);
          });

          break;
        }

        case "train_log": {
          const epoch = Number(data.epoch ?? 0);
          const loss = Number(data.loss ?? 0);

          setTrainingMetrics((prev) => [...prev, { epoch, loss }]);
          break;
        }

        case "completed":
          setIsTraining(false);
          setTrainingStatus(`Completed — saved to ${data.output_path ?? "unknown"}`);
          break;

        case "stopped":
          setIsTraining(false);
          setTrainingStatus("Training stopped.");
          setProgress(0);
          break;

        case "error":
          setIsTraining(false);
          setTrainingStatus(`Error: ${data.message}`);
          showAlert({ message: data.message, suggestion: data.suggestion }, "error");
          break;
      }
    };

    return () => {
      closed = true;
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId, token]);

  // ------------------ TRAINING ------------------

  const resolvedModel = useCustomModel ? customModel.trim() : baseModel;

  const startTraining = async () => {
    if (!selectedDatasetId) return;

    setTrainingMetrics([]);
    setIsTraining(true);
    setTrainingStatus("Submitting…");

    try {
      const data = await apiStartTraining({
        dataset_id: selectedDatasetId,
        labels: selectedLabels,
        base_model: resolvedModel,
        val_ratio: valSplitRatio,
      });

      setActiveRunId(data.run_id);
      setTrainingStatus("Training started successfully");
    } catch (err) {
      setIsTraining(false);
      setTrainingStatus("Training failed to start");
      showAlert({ message: err instanceof Error ? err.message : String(err) }, "error");
    }
  };

  const stopTraining = async () => {
    if (!activeRunId || !token) return;

    await apiStopTraining(activeRunId);

    setIsTraining(false);
    setTrainingStatus("Stop requested.");
  };

  const value: MonitorContextValue = {
    activeView,
    setActiveView,

    datasets,
    selectedDatasetId,
    selectDataset,
    datasetStats,

    runs,
    selectedRun,
    setSelectedRun,

    evaluation,
    evaluationLoading,
    evaluations,
    evaluationsLoading,

    isTraining,
    progress,
    trainingMetrics,
    trainingStatus,

    selectedLabels,
    setSelectedLabels,
    valSplitRatio,
    setValSplitRatio,
    baseModel,
    customModel,
    setCustomModel,
    useCustomModel,
    setUseCustomModel,

    startTraining,
    stopTraining,

    toast,
  };

  return <MonitorContext.Provider value={value}>{children}</MonitorContext.Provider>;
};

export default MonitorProvider;
