import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { useToast } from "@hooks/useToast";
import {
  getDatasetRuns,
  getDatasets,
  getMultiDatasetStats,
  getRunEvaluation,
  getTrainingWSUrl,
  startTraining as apiStartTraining,
  stopTraining as apiStopTraining,
} from "@api/monitoring";
import type { EvaluationResponse, MonitorDataset, MonitorDatasetStats, MonitorRun, TrainingMetric } from "types";

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

  const [activeView, setActiveView] = useState<MonitorView>("models");

  const [progress, setProgress] = useState(0);
  const [, setTotalEpochs] = useState(4);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const [datasets, setDatasets] = useState<MonitorDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationResponse[]>([]);
  const [evaluationsLoading] = useState(false);

  // Training datasets (multi-select) + optional eval datasets, with their
  // aggregated stats. Default to the single top-level selected dataset.
  const [trainingDatasetIds, setTrainingDatasetIds] = useState<number[]>([]);
  const [evalDatasetIds, setEvalDatasetIds] = useState<number[]>([]);
  const [trainingStats, setTrainingStats] = useState<MonitorDatasetStats | null>(null);

  const [runs, setRuns] = useState<MonitorRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [valSplitRatio, setValSplitRatio] = useState<number>(0.1);

  // Hyperparameters (defaults match the bioner trainer's current values).
  const [numEpochs, setNumEpochs] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<number>(5e-6);
  const [trainBatchSize, setTrainBatchSize] = useState<number>(8);

  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);

  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string>("");

  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  // Model selection
  const [baseModel, setBaseModel] = useState<string>(DEFAULT_MODEL);
  const [customModel, setCustomModel] = useState<string>("");
  const [useCustomModel, setUseCustomModel] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isTrainingRef = useRef(false);
  const totalEpochsRef = useRef(4);
  const totalStepsRef = useRef(0);
  const [activeRunId, setActiveRunId] = useState<number | null>(null);

  // keep a ref of isTraining so the WS reconnect logic can read it without re-subscribing
  useEffect(() => {
    isTrainingRef.current = isTraining;
  }, [isTraining]);

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

  const selectDataset = (id: number) => {
    setSelectedDatasetId(id);
    resetAll();
    // Default training to the picked dataset; clear any eval datasets.
    setTrainingDatasetIds([id]);
    setEvalDatasetIds([]);
  };

  // ------------------ TRAINING DATASET STATS (aggregated) ------------------

  useEffect(() => {
    if (!token || trainingDatasetIds.length === 0) {
      setTrainingStats(null);
      return;
    }

    let cancelled = false;
    getMultiDatasetStats(trainingDatasetIds)
      .then((data) => {
        if (!cancelled) setTrainingStats(data);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error(err);
        setTrainingStats(null);
      });

    return () => {
      cancelled = true;
    };
  }, [trainingDatasetIds, token]);

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

    const handleMessage = (event: MessageEvent) => {
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
          setCurrentStep(0);
          setTrainingStatus("Training started…");

          totalEpochsRef.current = data.num_epochs ?? 4;
          setTotalEpochs(data.num_epochs ?? 4);
          totalStepsRef.current = data.total_steps ?? 0;
          setTotalSteps(data.total_steps ?? 0);
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
          // Only plot points that actually carry a loss — train_log events also
          // fire for eval / summary logs (no loss), which would otherwise inject
          // a spurious 0 and drop the curve to the axis.
          if (data.loss == null) break;
          const loss = Number(data.loss);
          if (!Number.isFinite(loss)) break;

          const epoch = Number(data.epoch ?? 0);
          const step = data.step != null ? Number(data.step) : null;
          const evalLoss = data.eval_loss != null ? Number(data.eval_loss) : null;

          if (step != null) {
            setCurrentStep(step);
            const total = totalStepsRef.current;
            if (total > 0) {
              setProgress(Math.min(100, Math.round((step / total) * 100)));
            }
          }

          setTrainingMetrics((prev) => [...prev, { epoch, loss, step, eval_loss: evalLoss }]);
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

    const scheduleReconnect = () => {
      if (closed) return;
      // exponential backoff: 1s, 2s, 4s … capped at 30s
      const attempt = reconnectAttemptsRef.current;
      const delay = Math.min(30000, 1000 * 2 ** attempt);
      reconnectAttemptsRef.current = attempt + 1;
      // only surface the transient state if a training run is actually in flight
      if (isTrainingRef.current) {
        setTrainingStatus("Connection lost — reconnecting…");
      }
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(getTrainingWSUrl(token));
      wsRef.current = ws;

      ws.onopen = () => {
        // successful connection — reset the backoff
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = handleMessage;

      ws.onclose = () => {
        // ignore the close triggered by intentional teardown (cleanup sets closed)
        if (closed) return;
        scheduleReconnect();
      };

      // onerror is followed by onclose in browsers; let onclose drive the reconnect
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      reconnectAttemptsRef.current = 0;
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId, token]);

  // ------------------ TRAINING ------------------

  const resolvedModel = useCustomModel ? customModel.trim() : baseModel;

  const startTraining = async () => {
    if (trainingDatasetIds.length === 0) return;

    setTrainingMetrics([]);
    setIsTraining(true);
    setTrainingStatus("Submitting…");

    try {
      const data = await apiStartTraining({
        dataset_ids: trainingDatasetIds,
        eval_dataset_ids: evalDatasetIds,
        labels: selectedLabels,
        base_model: resolvedModel,
        val_ratio: valSplitRatio,
        num_epochs: numEpochs,
        learning_rate: learningRate,
        train_batch_size: trainBatchSize,
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
    trainingStats,

    runs,
    selectedRun,
    setSelectedRun,

    evaluation,
    evaluationLoading,
    evaluations,
    evaluationsLoading,

    isTraining,
    progress,
    currentStep,
    totalSteps,
    trainingMetrics,
    trainingStatus,

    trainingDatasetIds,
    setTrainingDatasetIds,
    evalDatasetIds,
    setEvalDatasetIds,
    selectedLabels,
    setSelectedLabels,
    valSplitRatio,
    setValSplitRatio,
    baseModel,
    setBaseModel,
    customModel,
    setCustomModel,
    useCustomModel,
    setUseCustomModel,

    numEpochs,
    setNumEpochs,
    learningRate,
    setLearningRate,
    trainBatchSize,
    setTrainBatchSize,

    startTraining,
    stopTraining,

    toast,
  };

  return <MonitorContext.Provider value={value}>{children}</MonitorContext.Provider>;
};

export default MonitorProvider;
