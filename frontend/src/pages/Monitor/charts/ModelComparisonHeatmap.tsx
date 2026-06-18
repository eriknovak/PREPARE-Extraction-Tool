import { useMemo, useState } from "react";

import { ChartState, Heatmap } from "@components/charts";
import Select from "@components/Select";
import type { EvaluationResponse } from "types";

import { METRIC_OPTIONS, buildComparisonMatrix, type MetricMode } from "../chartData";
import styles from "./styles.module.css";

const ROW_HEIGHT = 40;
const CHROME_HEIGHT = 96; // axis labels + visualMap legend

interface Props {
  evaluations: EvaluationResponse[];
  loading?: boolean;
}

/**
 * Compares every run's per-label scores on the dataset as a heatmap. The metric
 * is switchable; rows scroll vertically when there are many runs.
 */
const ModelComparisonHeatmap = ({ evaluations, loading }: Props) => {
  const [metric, setMetric] = useState<MetricMode>("exact_f1");

  const matrix = useMemo(
    () => buildComparisonMatrix(evaluations, metric),
    [evaluations, metric]
  );

  const metricLabel =
    METRIC_OPTIONS.find((o) => o.value === metric)?.label ?? "Score";

  let body: React.ReactNode;
  if (loading) {
    body = <ChartState variant="loading" message="Loading run evaluations…" height={260} />;
  } else if (matrix.yLabels.length === 0 || matrix.xLabels.length === 0) {
    body = (
      <ChartState
        variant="empty"
        title="Nothing to compare yet"
        message="Train and evaluate runs on this dataset to compare their per-label performance side by side."
        height={260}
      />
    );
  } else {
    const chartHeight = matrix.yLabels.length * ROW_HEIGHT + CHROME_HEIGHT;
    body = (
      <div className={styles.scroll}>
        <Heatmap
          xLabels={matrix.xLabels}
          yLabels={matrix.yLabels}
          data={matrix.cells}
          height={chartHeight}
          min={0}
          max={1}
          tooltipFormatter={({ xLabel, yLabel, value }) =>
            `${yLabel} · ${xLabel}<br/>${metricLabel}: <b>${value.toFixed(3)}</b>`
          }
        />
      </div>
    );
  }

  return (
    <div className={styles.comparison}>
      <div className={styles.controls}>
        <label className={styles.controls__label}>Metric</label>
        <Select
          value={metric}
          onValueChange={(v) => setMetric(v as MetricMode)}
          fullWidth={false}
          options={METRIC_OPTIONS}
        />
      </div>
      {body}
    </div>
  );
};

export default ModelComparisonHeatmap;
