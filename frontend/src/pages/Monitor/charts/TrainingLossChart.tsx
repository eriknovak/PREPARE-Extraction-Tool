import Button from "@components/Button";
import { ChartState, LineChart } from "@components/charts";
import type { TrainingMetric } from "types";

import { CHART } from "../chartColors";
import { buildLossSeries, formatEpoch, formatLoss } from "../chartData";

const CHART_HEIGHT = 280;

interface Props {
  metrics: TrainingMetric[];
  isTraining: boolean;
  /** Whether the selected dataset has any past training runs. */
  hasRuns: boolean;
  /** Jump to the training controls (used by the first-run empty state). */
  onConfigure?: () => void;
}

/** Live training-loss curve with loading / empty / guided-first-run states. */
const TrainingLossChart = ({ metrics, isTraining, hasRuns, onConfigure }: Props) => {
  if (metrics.length > 0) {
    const { xData, loss } = buildLossSeries(metrics);
    return (
      <LineChart
        xData={xData}
        xName="Epoch"
        yName="Loss"
        showLegend={false}
        height={CHART_HEIGHT}
        xAxisFormatter={formatEpoch}
        valueFormatter={formatLoss}
        series={[{ name: "Loss", data: loss, color: CHART.loss, area: true }]}
      />
    );
  }

  if (isTraining) {
    return <ChartState variant="loading" message="Waiting for training metrics…" height={CHART_HEIGHT} />;
  }

  if (!hasRuns) {
    return (
      <ChartState
        variant="empty"
        title="No training runs yet"
        message="Pick a dataset and labels above, choose a base model, then start your first run to watch the loss curve here in real time."
        height={CHART_HEIGHT}
        action={
          onConfigure && (
            <Button variant="primary" size="small" onClick={onConfigure}>
              Configure a run
            </Button>
          )
        }
      />
    );
  }

  return (
    <ChartState
      variant="empty"
      title="No active training"
      message="Start a new run to stream its loss curve live."
      height={CHART_HEIGHT}
    />
  );
};

export default TrainingLossChart;
