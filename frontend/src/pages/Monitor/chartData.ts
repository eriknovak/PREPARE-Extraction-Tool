import type { HeatmapCell } from "@components/charts";
import type { EvaluationResponse, PerLabelMetrics, TrainingMetric } from "types";

/** Selectable metric for the per-label and comparison views. */
export type MetricMode = "exact_f1" | "relaxed_f1" | "precision" | "recall";

export const METRIC_OPTIONS: { value: MetricMode; label: string }[] = [
  { value: "exact_f1", label: "Exact F1" },
  { value: "relaxed_f1", label: "Relaxed F1" },
  { value: "precision", label: "Precision" },
  { value: "recall", label: "Recall" },
];

/** Aggregate rows returned by the backend that should not appear as labels. */
const AGGREGATE_LABELS = new Set(["micro avg", "macro avg", "weighted avg"]);

/** Strip combining diacritics so label keys compare/sort consistently. */
export const normalizeLabel = (label: string): string =>
  label.normalize("NFD").replace(/[̀-ͯ]/g, "");

/** Read a single metric from a per-label entry, with sensible fallbacks. */
export const readMetric = (m: PerLabelMetrics, mode: MetricMode): number => {
  switch (mode) {
    case "exact_f1":
      return m.exact_f1 ?? m.f1 ?? 0;
    case "relaxed_f1":
      return m.relaxed_f1 ?? 0;
    case "precision":
      return m.precision ?? 0;
    case "recall":
      return m.recall ?? 0;
  }
};

const isLabelRow = (label: string): boolean => !AGGREGATE_LABELS.has(label.toLowerCase());

/** Loss curve data for the training line chart. */
export interface LossSeries {
  xData: number[];
  loss: number[];
}

export const buildLossSeries = (metrics: TrainingMetric[]): LossSeries => ({
  xData: metrics.map((m) => m.epoch),
  loss: metrics.map((m) => m.loss),
});

/** Per-label evaluation data for the grouped bar chart. */
export interface PerformanceData {
  categories: string[];
  exactF1: number[];
  relaxedF1: number[];
  precision: number[];
  recall: number[];
}

export const buildPerformanceData = (
  evaluation: EvaluationResponse | null
): PerformanceData => {
  const entries = Object.entries(evaluation?.per_label ?? {}).filter(([label]) =>
    isLabelRow(label)
  );

  return {
    categories: entries.map(([label]) => label),
    exactF1: entries.map(([, m]) => m.exact_f1 ?? m.f1 ?? 0),
    relaxedF1: entries.map(([, m]) => m.relaxed_f1 ?? 0),
    precision: entries.map(([, m]) => m.precision ?? 0),
    recall: entries.map(([, m]) => m.recall ?? 0),
  };
};

/** Run-vs-label comparison matrix for the heatmap. */
export interface ComparisonMatrix {
  /** Column labels (entity labels). */
  xLabels: string[];
  /** Row labels (one per run). */
  yLabels: string[];
  /** Cells keyed by [labelIndex, runIndex]. */
  cells: HeatmapCell[];
}

export const buildComparisonMatrix = (
  evaluations: EvaluationResponse[],
  mode: MetricMode
): ComparisonMatrix => {
  const xLabels = Array.from(
    new Set(
      evaluations.flatMap((run) =>
        Object.keys(run?.per_label ?? {})
          .filter(isLabelRow)
          .map(normalizeLabel)
      )
    )
  ).sort();

  const yLabels = evaluations.map((run) => `Run #${run.run_id}`);

  const cells: HeatmapCell[] = [];
  evaluations.forEach((run, y) => {
    // collapse normalized-label collisions, keeping the first value seen
    const byLabel = new Map<string, number>();
    Object.entries(run?.per_label ?? {}).forEach(([label, metrics]) => {
      if (!isLabelRow(label)) return;
      const norm = normalizeLabel(label);
      if (!byLabel.has(norm)) byLabel.set(norm, readMetric(metrics, mode));
    });

    xLabels.forEach((label, x) => {
      cells.push({ x, y, value: byLabel.get(label) ?? 0 });
    });
  });

  return { xLabels, yLabels, cells };
};
