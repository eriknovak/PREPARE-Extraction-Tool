import { useEffect, useRef, useState } from "react";
import classNames from "classnames";

import Button from "@components/Button";
import Card from "@components/Card";
import Layout from "@components/Layout";
import ProgressBar from "@components/ProgressBar";
import Select from "@components/Select";
import StatCard from "@components/StatCard";
import { ToastContainer } from "@components/Toast/ToastContainer";
import { usePageTitle } from "@hooks/usePageTitle";
import { useToast } from "@hooks/useToast";
import {
  getAllEvaluations,
  getDatasetRuns,
  getDatasets,
  getDatasetStats,
  getRunEvaluation,
  getTrainingWSUrl,
  startTraining,
  stopTraining,
} from "@api/monitoring";
import type {
  EvaluationResponse,
  MonitorDataset,
  MonitorDatasetStats,
  MonitorRun,
  TrainingMetric,
} from "types";

import LabelSelector from "./LabelSelector";
import TrainingLossChart from "./charts/TrainingLossChart";
import PerformanceChart from "./charts/PerformanceChart";
import ModelComparisonHeatmap from "./charts/ModelComparisonHeatmap";
import styles from "./styles.module.css";

const DEFAULT_MODEL = "urchade/gliner_small-v2.1";

const Monitor = () => {
  usePageTitle("Monitoring");

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

  const trainingCardRef = useRef<HTMLDivElement | null>(null);

  const scrollToTraining = () =>
    trainingCardRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });

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

  const startTrainingHandler = async () => {
    if (!selectedDatasetId) return;

    setTrainingMetrics([]);
    setIsTraining(true);
    setTrainingStatus("Submitting…");

    try {
      const data = await startTraining({
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

  const stopTrainingHandler = async () => {
    if (!activeRunId || !token) return;

    await stopTraining(activeRunId);

    setIsTraining(false);
    setTrainingStatus("Stop requested.");
  };

  // ------------------ RENDER ------------------

  return (
    <Layout>
      <div className={styles.page}>
        {/* HEADER */}
        <header className={styles.header}>
          <h1 className={styles.header__title}>Monitoring Dashboard</h1>
          <p className={styles.header__subtitle}>
            Train NER models, follow live training metrics, and compare run performance.
          </p>
        </header>

        {/* ─────────────── SETUP ─────────────── */}

        {/* DATASET */}
        <Card title="1 · Dataset">
          {datasets.length > 0 ? (
            <div className={styles["dataset-list"]}>
              {datasets.map((d) => (
                <Button
                  key={d.id}
                  onClick={() => selectDataset(d.id)}
                  variant={selectedDatasetId === d.id ? "primary" : "outline"}
                >
                  {d.name}
                </Button>
              ))}
            </div>
          ) : (
            <p className={styles.muted}>No datasets available.</p>
          )}
        </Card>

        {/* STATS + LABELS */}
        {datasetStats && (
          <Card title="2 · Labels & statistics">
            <div className={styles.stats}>
              <StatCard label="Records" value={datasetStats.totalRecords} />
              <StatCard label="Terms" value={datasetStats.totalTerms} />
            </div>

            <LabelSelector datasetStats={datasetStats} onChange={setSelectedLabels} />
          </Card>
        )}

        {/* TRAINING */}
        <div ref={trainingCardRef}>
          <Card title="3 · Training">
            {/* Model selector */}
            <div className={styles.field}>
              <p className={styles.field__label}>Base model</p>

              <div className={styles["radio-group"]}>
                <label className={styles.radio}>
                  <input
                    type="radio"
                    checked={!useCustomModel}
                    onChange={() => setUseCustomModel(false)}
                  />
                  <span>
                    Default: <code className={styles.code}>{DEFAULT_MODEL}</code>
                  </span>
                </label>

                <label className={styles.radio}>
                  <input
                    type="radio"
                    checked={useCustomModel}
                    onChange={() => setUseCustomModel(true)}
                  />
                  Custom model path or HuggingFace ID
                </label>

                {useCustomModel && (
                  <input
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="e.g. urchade/gliner_medium-v2.1 or /model/gliner/my-model"
                    className={styles.input}
                  />
                )}
              </div>
            </div>

            <div className={styles.field}>
              <label className={styles.field__label}>Train / Eval split</label>

              <Select
                value={String(valSplitRatio)}
                onValueChange={(v) => setValSplitRatio(Number(v))}
                fullWidth={false}
                options={[
                  { value: "0", label: "No split (100% train)" },
                  { value: "0.1", label: "90 / 10" },
                  { value: "0.2", label: "80 / 20" },
                  { value: "0.3", label: "70 / 30" },
                ]}
              />
            </div>

            <div className={styles.actions}>
              <Button
                variant="primary"
                onClick={startTrainingHandler}
                disabled={isTraining || !selectedDatasetId}
              >
                Start
              </Button>
              <Button variant="danger" onClick={stopTrainingHandler} disabled={!isTraining}>
                Stop
              </Button>

              {trainingStatus && (
                <p
                  className={classNames(styles.status, {
                    [styles["status--active"]]: isTraining,
                  })}
                >
                  {trainingStatus}
                </p>
              )}
            </div>
          </Card>
        </div>

        {/* ─────────────── LIVE TRAINING ‖ EVALUATION ─────────────── */}

        <div className={styles["charts-row"]}>
          <Card title="Training progress">
            <div className={styles.progress}>
              <ProgressBar value={progress} />
            </div>

            <TrainingLossChart
              metrics={trainingMetrics}
              isTraining={isTraining}
              hasRuns={runs.length > 0}
              onConfigure={scrollToTraining}
            />
          </Card>

          <Card
            title="Evaluation"
            actions={
              runs.length > 0 ? (
                <Select
                  value={selectedRun !== null ? String(selectedRun) : ""}
                  onValueChange={(v) => setSelectedRun(Number(v))}
                  placeholder="Select a run"
                  fullWidth={false}
                  options={runs.map((r) => ({ value: String(r.run_id), label: `Run #${r.run_id}` }))}
                />
              ) : undefined
            }
          >
            <PerformanceChart
              evaluation={evaluation}
              loading={evaluationLoading}
              hasSelectedRun={selectedRun !== null}
            />
          </Card>
        </div>

        {/* ─────────────── COMPARISON ─────────────── */}

        <Card title="Run comparison">
          <ModelComparisonHeatmap evaluations={evaluations} loading={evaluationsLoading} />
        </Card>

        <ToastContainer toasts={toast.toasts} onDismiss={toast.dismissToast} duration={5000} />
      </div>
    </Layout>
  );
};

export default Monitor;
