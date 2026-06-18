import { BarChart, ChartState } from "@components/charts";
import type { EvaluationResponse } from "types";

import { CHART } from "../chartColors";
import { buildPerformanceData } from "../chartData";

const CHART_HEIGHT = 280;

interface Props {
  evaluation: EvaluationResponse | null;
  loading?: boolean;
  /** Whether a run is currently selected. */
  hasSelectedRun: boolean;
}

/** Per-label evaluation bars (exact/relaxed F1, precision, recall). */
const PerformanceChart = ({ evaluation, loading, hasSelectedRun }: Props) => {
  if (loading) {
    return <ChartState variant="loading" message="Loading evaluation…" height={CHART_HEIGHT} />;
  }

  const data = buildPerformanceData(evaluation);

  if (data.categories.length > 0) {
    return (
      <BarChart
        categories={data.categories}
        height={CHART_HEIGHT}
        yMin={0}
        yMax={1}
        xLabelRotate={data.categories.length > 5 ? 30 : 0}
        series={[
          { name: "Exact F1", data: data.exactF1, color: CHART.exactF1 },
          { name: "Relaxed F1", data: data.relaxedF1, color: CHART.relaxedF1 },
          { name: "Precision", data: data.precision, color: CHART.precision },
          { name: "Recall", data: data.recall, color: CHART.recall },
        ]}
      />
    );
  }

  if (!hasSelectedRun) {
    return (
      <ChartState
        variant="empty"
        title="No run selected"
        message="Select a run to view its per-label evaluation scores."
        height={CHART_HEIGHT}
      />
    );
  }

  return (
    <ChartState
      variant="empty"
      title="No evaluation data"
      message="This run has no evaluation results yet."
      height={CHART_HEIGHT}
    />
  );
};

export default PerformanceChart;
