import { useEffect, useMemo, useState } from "react";

import { getRunMetrics } from "@api/monitoring";
import Card from "@components/Card";
import Select from "@components/Select";
import { BarChart, ChartState, LineChart } from "@components/charts";
import type { EvaluationResponse, TrainingMetric } from "types";

import ModelComparisonHeatmap from "../charts/ModelComparisonHeatmap";
import PerformanceChart from "../charts/PerformanceChart";
import {
  METRIC_OPTIONS,
  formatEpoch,
  formatLoss,
  normalizeLabel,
  readMetric,
  type MetricMode,
} from "../chartData";
import { useMonitor } from "../hooks/useMonitor";
import styles from "./ComparisonView.module.css";

const CHART_HEIGHT = 280;

/** Aggregate rows returned by the backend that should not appear as labels. */
const AGGREGATE_LABELS = new Set(["micro avg", "macro avg", "weighted avg"]);

interface RunLoss {
  runId: number;
  metrics: TrainingMetric[];
}

/** Build a multi-series loss overlay: one line per run, aligned on the epoch axis. */
const buildLossOverlay = (runLosses: RunLoss[]) => {
  const withData = runLosses.filter((r) => r.metrics.length > 0);
  const epochs = Array.from(new Set(withData.flatMap((r) => r.metrics.map((m) => m.epoch)))).sort(
    (a, b) => a - b
  );
  const series = withData.map((r) => {
    const byEpoch = new Map(r.metrics.map((m) => [m.epoch, m.loss]));
    return {
      name: `Run #${r.runId}`,
      // null renders a gap where a run has no point for that epoch
      data: epochs.map((e) => byEpoch.get(e) ?? null) as unknown as number[],
    };
  });
  return { epochs, series };
};

/** Build a grouped bar comparison: one series per run, one category per label. */
const buildEvalComparison = (evaluations: EvaluationResponse[], metric: MetricMode) => {
  const labels = Array.from(
    new Set(
      evaluations.flatMap((run) =>
        Object.keys(run?.per_label ?? {})
          .filter((label) => !AGGREGATE_LABELS.has(label.toLowerCase()))
          .map(normalizeLabel)
      )
    )
  ).sort();

  const series = evaluations.map((run) => {
    const byLabel = new Map<string, number>();
    Object.entries(run?.per_label ?? {}).forEach(([label, m]) => {
      if (AGGREGATE_LABELS.has(label.toLowerCase())) return;
      const norm = normalizeLabel(label);
      if (!byLabel.has(norm)) byLabel.set(norm, readMetric(m, metric));
    });
    return {
      name: `Run #${run.run_id}`,
      data: labels.map((l) => byLabel.get(l) ?? 0),
    };
  });

  return { labels, series };
};

/**
 * Comparison view — compares model/run performance for the selected dataset:
 * loss across runs, evaluation across runs, a runs × labels heatmap, and a
 * single-model detail section with its own run picker.
 */
const ComparisonView = () => {
  const {
    selectedDatasetId,
    runs,
    selectedRun,
    setSelectedRun,
    evaluation,
    evaluationLoading,
    evaluations,
    evaluationsLoading,
    toast,
  } = useMonitor();

  const [evalMetric, setEvalMetric] = useState<MetricMode>("exact_f1");

  // Historical loss for every run on the dataset (for the overlay).
  const [allLosses, setAllLosses] = useState<RunLoss[]>([]);
  const [allLossesLoading, setAllLossesLoading] = useState(false);

  // Historical loss for the single run picked in the detail section.
  const [runLoss, setRunLoss] = useState<TrainingMetric[]>([]);
  const [runLossLoading, setRunLossLoading] = useState(false);

  const runIdsKey = useMemo(() => runs.map((r) => r.run_id).join(","), [runs]);

  // Fetch loss curves for all runs whenever the run set changes.
  useEffect(() => {
    if (runs.length === 0) {
      setAllLosses([]);
      return;
    }
    let cancelled = false;
    setAllLossesLoading(true);
    Promise.all(
      runs.map((r) =>
        getRunMetrics(r.run_id)
          .then((metrics) => ({ runId: r.run_id, metrics }))
          .catch(() => ({ runId: r.run_id, metrics: [] as TrainingMetric[] }))
      )
    )
      .then((results) => {
        if (!cancelled) setAllLosses(results);
      })
      .catch(() => {
        if (!cancelled) toast.showToast("Failed to load run loss curves", "error");
      })
      .finally(() => {
        if (!cancelled) setAllLossesLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIdsKey]);

  // Fetch the picked run's loss curve.
  useEffect(() => {
    if (selectedRun === null) {
      setRunLoss([]);
      return;
    }
    let cancelled = false;
    setRunLossLoading(true);
    getRunMetrics(selectedRun)
      .then((metrics) => {
        if (!cancelled) setRunLoss(metrics);
      })
      .catch(() => {
        if (!cancelled) {
          setRunLoss([]);
          toast.showToast("Failed to load run loss curve", "error");
        }
      })
      .finally(() => {
        if (!cancelled) setRunLossLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun]);

  const lossOverlay = useMemo(() => buildLossOverlay(allLosses), [allLosses]);
  const evalComparison = useMemo(
    () => buildEvalComparison(evaluations, evalMetric),
    [evaluations, evalMetric]
  );

  const runOptions = useMemo(
    () =>
      runs.map((r) => ({
        value: String(r.run_id),
        label: r.status ? `Run #${r.run_id} · ${r.status}` : `Run #${r.run_id}`,
      })),
    [runs]
  );

  if (!selectedDatasetId) {
    return <Card title="Comparison">Select a dataset to compare runs.</Card>;
  }

  // ── Loss comparison ──
  let lossBody: React.ReactNode;
  if (allLossesLoading) {
    lossBody = <ChartState variant="loading" message="Loading loss curves…" height={CHART_HEIGHT} />;
  } else if (lossOverlay.series.length === 0) {
    lossBody = (
      <ChartState
        variant="empty"
        title="No loss data yet"
        message="Train runs on this dataset to overlay their loss curves here."
        height={CHART_HEIGHT}
      />
    );
  } else {
    lossBody = (
      <LineChart
        xData={lossOverlay.epochs}
        xName="Epoch"
        yName="Loss"
        height={CHART_HEIGHT}
        xAxisFormatter={formatEpoch}
        valueFormatter={formatLoss}
        series={lossOverlay.series}
      />
    );
  }

  // ── Evaluation comparison ──
  let evalBody: React.ReactNode;
  if (evaluationsLoading) {
    evalBody = <ChartState variant="loading" message="Loading run evaluations…" height={CHART_HEIGHT} />;
  } else if (evalComparison.labels.length === 0 || evalComparison.series.length === 0) {
    evalBody = (
      <ChartState
        variant="empty"
        title="Nothing to compare yet"
        message="Evaluate runs on this dataset to compare their per-label scores side by side."
        height={CHART_HEIGHT}
      />
    );
  } else {
    evalBody = (
      <BarChart
        categories={evalComparison.labels}
        height={CHART_HEIGHT}
        yMin={0}
        yMax={1}
        xLabelRotate={evalComparison.labels.length > 5 ? 30 : 0}
        series={evalComparison.series}
      />
    );
  }

  // ── Single-model loss ──
  let detailLossBody: React.ReactNode;
  if (runLossLoading) {
    detailLossBody = <ChartState variant="loading" message="Loading loss curve…" height={CHART_HEIGHT} />;
  } else if (selectedRun === null) {
    detailLossBody = (
      <ChartState
        variant="empty"
        title="No run selected"
        message="Pick a run above to view its loss curve."
        height={CHART_HEIGHT}
      />
    );
  } else if (runLoss.length === 0) {
    detailLossBody = (
      <ChartState
        variant="empty"
        title="No loss data"
        message="This run has no recorded loss metrics."
        height={CHART_HEIGHT}
      />
    );
  } else {
    const { xData, loss } = {
      xData: runLoss.map((m) => m.epoch),
      loss: runLoss.map((m) => m.loss),
    };
    detailLossBody = (
      <LineChart
        xData={xData}
        xName="Epoch"
        yName="Loss"
        showLegend={false}
        height={CHART_HEIGHT}
        xAxisFormatter={formatEpoch}
        valueFormatter={formatLoss}
        series={[{ name: "Loss", data: loss, area: true }]}
      />
    );
  }

  return (
    <div className={styles.view}>
      <Card title="Loss comparison">{lossBody}</Card>

      <Card
        title="Evaluation comparison"
        actions={
          <div className={styles.controls}>
            <label className={styles.controls__label}>Metric</label>
            <Select
              value={evalMetric}
              onValueChange={(v) => setEvalMetric(v as MetricMode)}
              fullWidth={false}
              size="small"
              options={METRIC_OPTIONS}
            />
          </div>
        }
      >
        {evalBody}
      </Card>

      <Card title="Per-label heatmap">
        <ModelComparisonHeatmap evaluations={evaluations} loading={evaluationsLoading} />
      </Card>

      <div className={styles.divider}>Single model</div>

      <Card
        title="Model detail"
        actions={
          <div className={styles.controls}>
            <label className={styles.controls__label}>Run</label>
            <Select
              value={selectedRun !== null ? String(selectedRun) : undefined}
              onValueChange={(v) => setSelectedRun(Number(v))}
              placeholder="Select a run"
              fullWidth={false}
              size="small"
              options={runOptions}
            />
          </div>
        }
      >
        <div className={styles.detail}>
          <div className={styles.detail__block}>
            <h3 className={styles.detail__title}>Loss curve</h3>
            {detailLossBody}
          </div>
          <div className={styles.detail__block}>
            <h3 className={styles.detail__title}>Per-label evaluation</h3>
            <PerformanceChart
              evaluation={evaluation}
              loading={evaluationLoading}
              hasSelectedRun={selectedRun !== null}
            />
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ComparisonView;
