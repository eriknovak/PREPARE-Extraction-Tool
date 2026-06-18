import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
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
  getAllRunEvaluations,
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
  PerLabelMetrics,
  TrainingMetric,
} from "types";

import LabelSelector from "./LabelSelector";
import { CHART } from "./chartColors";
import styles from "./styles.module.css";

const DEFAULT_MODEL = "urchade/gliner_small-v2.1";

/** Prefer relaxed F1, then exact F1, then legacy f1 field. */
const readF1 = (m: PerLabelMetrics): number => m.relaxed_f1 ?? m.exact_f1 ?? m.f1 ?? 0;

const readMetric = (m: PerLabelMetrics, mode: "f1" | "precision" | "recall"): number => {
  if (mode === "precision") return m.precision ?? 0;
  if (mode === "recall") return m.recall ?? 0;
  return readF1(m);
};

interface HeatmapRow {
  run: number;
  labels: Record<string, number>;
}

interface HoveredCell {
  run: number;
  label: string;
  value: number;
}

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
  const [, setAllRunEvaluations] = useState<EvaluationResponse[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationResponse[]>([]);
  const [hovered, setHovered] = useState<HoveredCell | null>(null);
  const [datasetStats, setDatasetStats] = useState<MonitorDatasetStats | null>(null);

  const [runs, setRuns] = useState<MonitorRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [valSplitRatio, setValSplitRatio] = useState<number>(0.1);

  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);

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

  const [metricMode, setMetricMode] = useState<"f1" | "precision" | "recall">("f1");

  // ------------------ RESET ------------------

  const resetAll = () => {
    setRuns([]);
    setSelectedRun(null);
    setEvaluation(null);
    setTrainingMetrics([]);
    setIsTraining(false);
    setTrainingStatus("");
    setSelectedLabels([]);
    setAllRunEvaluations([]);
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

    const data = await getDatasetStats(id);
    setDatasetStats(data);
  };

  useEffect(() => {
    if (!selectedDatasetId || !token) return;

    getAllEvaluations(selectedDatasetId).then((data) => {
      setEvaluations(Array.isArray(data) ? data : []);
    });
  }, [selectedDatasetId, token]);

  // ------------------ RUNS ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;

    const fetchRuns = async () => {
      try {
        const data = await getDatasetRuns(selectedDatasetId);
        const runsArray = Array.isArray(data) ? data : [];

        setRuns(runsArray);
        setSelectedRun(runsArray?.[0]?.run_id ?? null);
      } catch (e) {
        console.error(e);
        setRuns([]);
        setSelectedRun(null);
      }
    };

    fetchRuns();
  }, [selectedDatasetId, token]);

  // ------------------ ALL RUN EVAL ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;

    getAllRunEvaluations(selectedDatasetId).then((data) => {
      setAllRunEvaluations(data ?? []);
    });
  }, [selectedDatasetId, token]);

  // ------------------ SINGLE EVAL ------------------

  useEffect(() => {
    if (!selectedRun || !token) return;

    getRunEvaluation(selectedRun).then((data) => {
      setEvaluation(data);
    });
  }, [selectedRun, token]);

  // ------------------ LABEL NORMALIZATION ------------------

  const normalizeLabel = (label: string) => label.normalize("NFD").replace(/[̀-ͯ]/g, "");

  // ------------------ SAFE API DATA ------------------

  const safeRuns = evaluations;

  const labelKeys = useMemo(
    () =>
      Array.from(
        new Set(safeRuns.flatMap((run) => Object.keys(run?.per_label ?? {}).map(normalizeLabel)))
      ).sort(),
    [safeRuns]
  );

  const heatmapData = useMemo<HeatmapRow[]>(() => {
    return safeRuns.map((run) => {
      const row: HeatmapRow = {
        run: run.run_id,
        labels: {},
      };

      const perLabel: { [label: string]: PerLabelMetrics } = run.per_label ?? {};

      Object.entries(perLabel).forEach(([label, metrics]) => {
        const norm = normalizeLabel(label);
        row.labels[norm] = readMetric(metrics, metricMode);
      });

      // ensure all labels exist
      labelKeys.forEach((label) => {
        if (row.labels[label] === undefined) {
          row.labels[label] = 0;
        }
      });

      return row;
    });
  }, [safeRuns, metricMode, labelKeys]);

  const getColor = (value: number) => {
    // clamp 0-1, interpolate from the "loss" (red) token to the "exactF1" (green) token
    const v = Math.max(0, Math.min(1, value));

    const lerp = (from: number, to: number) => Math.round(from + (to - from) * v);
    const hexToRgb = (hex: string): [number, number, number] => [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16),
    ];

    const [r0, g0, b0] = hexToRgb(CHART.loss);
    const [r1, g1, b1] = hexToRgb(CHART.exactF1);

    return `rgb(${lerp(r0, r1)},${lerp(g0, g1)},${lerp(b0, b1)})`;
  };

  // ------------------ WEBSOCKET ------------------

  useEffect(() => {
    if (!selectedDatasetId || !token) return;
    const ws = new WebSocket(getTrainingWSUrl(token));
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

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

    return () => ws.close();
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

  // ------------------ CHART DATA ------------------

  const chartData = evaluation?.per_label
    ? (Object.entries(evaluation.per_label) as [string, PerLabelMetrics][])
        .filter(([k]) => !["micro avg", "macro avg", "weighted avg"].includes(k))
        .map(([label, m]) => ({
          label,
          exact_f1: m.exact_f1 ?? 0,
          relaxed_f1: m.relaxed_f1 ?? 0,
          precision: m.precision ?? 0,
          recall: m.recall ?? 0,
        }))
    : [];

  // ------------------ RENDER ------------------

  return (
    <Layout>
      <h1 className={styles.monitor__title}>Monitoring Dashboard</h1>

      {/* DATASET */}
      <Card title="Dataset">
        <div className={styles["monitor__dataset-list"]}>
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
      </Card>

      {/* STATS */}
      {datasetStats && (
        <Card title={`Dataset ${selectedDatasetId}`}>
          <div className={styles.monitor__stats}>
            <StatCard label="Records" value={datasetStats.totalRecords} />
            <StatCard label="Terms" value={datasetStats.totalTerms} />
          </div>

          <LabelSelector
            datasetId={selectedDatasetId}
            datasetStats={datasetStats}
            onChange={setSelectedLabels}
          />
        </Card>
      )}

      {/* TRAINING */}
      <Card title="Training">
        {/* Model selector */}
        <div className={styles.monitor__field}>
          <p className={styles.monitor__label}>Base model</p>

          <div className={styles["monitor__radio-group"]}>
            <label className={styles.monitor__radio}>
              <input type="radio" checked={!useCustomModel} onChange={() => setUseCustomModel(false)} />
              <span>
                Default: <code className={styles.monitor__code}>{DEFAULT_MODEL}</code>
              </span>
            </label>

            <label className={styles.monitor__radio}>
              <input type="radio" checked={useCustomModel} onChange={() => setUseCustomModel(true)} />
              Custom model path or HuggingFace ID
            </label>

            {useCustomModel && (
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="e.g. urchade/gliner_medium-v2.1 or /model/gliner/my-model"
                className={styles.monitor__input}
              />
            )}
          </div>
        </div>

        <div className={styles.monitor__field}>
          <label className={styles.monitor__label}>Train / Eval Split</label>

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

        <div className={styles.monitor__actions}>
          <Button variant="primary" onClick={startTrainingHandler} disabled={isTraining}>
            Start
          </Button>
          <Button variant="danger" onClick={stopTrainingHandler} disabled={!isTraining}>
            Stop
          </Button>
        </div>

        {trainingStatus && (
          <p
            className={classNames(styles.monitor__status, {
              [styles["monitor__status--active"]]: isTraining,
            })}
          >
            {trainingStatus}
          </p>
        )}
      </Card>

      {/* GRID CHARTS */}
      <div className={styles.monitor__grid}>
        {/* TRAINING PROGRESS */}
        <Card title="Training Progress">
          <div className={styles.monitor__progress}>
            <ProgressBar progress={progress} />
          </div>

          {trainingMetrics.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingMetrics}>
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line dataKey="loss" stroke={CHART.loss} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className={styles.monitor__empty}>No training data</p>
          )}
        </Card>

        {/* SELECT RUN */}
        <Card title="Select Run">
          <Select
            value={selectedRun !== null ? String(selectedRun) : ""}
            onValueChange={(v) => setSelectedRun(Number(v))}
            placeholder="Select a run"
            fullWidth={false}
            options={runs.map((r) => ({ value: String(r.run_id), label: `Run #${r.run_id}` }))}
          />

          <div className={styles["monitor__run-info"]}>Selected Run: {selectedRun ?? "None"}</div>
        </Card>

        {/* PER LABEL */}
        <Card title="Per-label Evaluation">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData}>
                <XAxis dataKey="label" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="exact_f1" fill={CHART.exactF1} />
                <Bar dataKey="relaxed_f1" fill={CHART.relaxedF1} />
                <Bar dataKey="precision" fill={CHART.precision} />
                <Bar dataKey="recall" fill={CHART.recall} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className={styles.monitor__empty}>No evaluation data</p>
          )}
        </Card>
      </div>

      {/* HEATMAP */}
      <Card title="Run Comparison Heatmap (All Runs)">
        <div className={styles["monitor__heatmap-controls"]}>
          <label className={styles.monitor__label}>Metric:</label>
          <Select
            value={metricMode}
            onValueChange={(v) => setMetricMode(v as "f1" | "precision" | "recall")}
            fullWidth={false}
            options={[
              { value: "f1", label: "F1" },
              { value: "precision", label: "Precision" },
              { value: "recall", label: "Recall" },
            ]}
          />
        </div>

        {heatmapData.length > 0 ? (
          <div className={styles.monitor__heatmap}>
            <svg width={900} height={400}>
              {/* LABEL HEADERS */}
              {labelKeys.map((label, i) => (
                <text
                  key={label}
                  x={120 + i * 80}
                  y={20}
                  fontSize={12}
                  textAnchor="middle"
                  className={styles["monitor__heatmap-text"]}
                >
                  {label}
                </text>
              ))}

              {/* ROWS */}
              {heatmapData.map((row, rowIndex) => (
                <g key={row.run}>
                  {/* RUN LABEL */}
                  <text
                    x={10}
                    y={60 + rowIndex * 40}
                    fontSize={12}
                    className={styles["monitor__heatmap-text"]}
                  >
                    Run {row.run}
                  </text>

                  {/* CELLS */}
                  {labelKeys.map((label, colIndex) => {
                    const value = row.labels[label];

                    return (
                      <rect
                        key={label}
                        x={120 + colIndex * 80}
                        y={40 + rowIndex * 40}
                        width={70}
                        height={30}
                        fill={getColor(value)}
                        className={styles["monitor__heatmap-cell"]}
                        onMouseEnter={() => setHovered({ run: row.run, label, value })}
                        onMouseLeave={() => setHovered(null)}
                      />
                    );
                  })}
                </g>
              ))}
            </svg>

            {/* TOOLTIP */}
            {hovered && (
              <div className={styles.monitor__tooltip}>
                <div>
                  <b>Run:</b> {hovered.run}
                </div>
                <div>
                  <b>Label:</b> {hovered.label}
                </div>
                <div>
                  <b>Value:</b> {hovered.value.toFixed(3)}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className={styles.monitor__empty}>No data</p>
        )}
      </Card>

      <ToastContainer toasts={toast.toasts} onDismiss={toast.dismissToast} duration={5000} />
    </Layout>
  );
};

export default Monitor;
