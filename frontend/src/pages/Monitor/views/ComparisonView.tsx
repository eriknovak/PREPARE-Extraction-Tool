import Card from "@components/Card";

import { useMonitor } from "../hooks/useMonitor";

/**
 * Comparison view — compares model/run performance for the selected dataset:
 * loss across runs, evaluation across runs, a runs × labels heatmap, and a
 * single-model detail section with its own run picker.
 *
 * Stub: implemented in a follow-up PR against the `useMonitor()` contract.
 */
const ComparisonView = () => {
  const { selectedDatasetId } = useMonitor();

  if (!selectedDatasetId) {
    return <Card title="Comparison">Select a dataset to compare runs.</Card>;
  }

  return <Card title="Comparison">Comparison view coming soon.</Card>;
};

export default ComparisonView;
