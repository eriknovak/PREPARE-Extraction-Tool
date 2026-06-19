import Card from "@components/Card";

import { useMonitor } from "../hooks/useMonitor";

/**
 * Training view — trains a new model on the selected dataset: train/eval split,
 * primary model selection, start/stop, and live training progression.
 *
 * Stub: implemented in a follow-up PR against the `useMonitor()` contract.
 */
const TrainingView = () => {
  const { selectedDatasetId } = useMonitor();

  if (!selectedDatasetId) {
    return <Card title="Training">Select a dataset to train a model.</Card>;
  }

  return <Card title="Training">Training view coming soon.</Card>;
};

export default TrainingView;
