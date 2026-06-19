import { createContext, useContext } from "react";

import type { useToast } from "@hooks/useToast";
import type {
  EvaluationResponse,
  MonitorDataset,
  MonitorDatasetStats,
  MonitorRun,
  TrainingMetric,
} from "types";

export const DEFAULT_MODEL = "urchade/gliner_small-v2.1";

export type MonitorView = "comparison" | "training";

/**
 * The shared Monitor contract. Both `index.tsx` and the view components
 * (ComparisonView / TrainingView) consume this via `useMonitor()`. The shape is
 * frozen so the views can be developed independently against it.
 */
export interface MonitorContextValue {
  // ── view toggle ──
  activeView: MonitorView;
  setActiveView: (view: MonitorView) => void;

  // ── datasets ──
  datasets: MonitorDataset[];
  selectedDatasetId: number | null;
  selectDataset: (id: number) => void;
  datasetStats: MonitorDatasetStats | null;

  // ── runs ──
  runs: MonitorRun[];
  selectedRun: number | null;
  setSelectedRun: (id: number | null) => void;

  // ── evaluations ──
  evaluation: EvaluationResponse | null;
  evaluationLoading: boolean;
  evaluations: EvaluationResponse[];
  evaluationsLoading: boolean;

  // ── training state ──
  isTraining: boolean;
  progress: number;
  trainingMetrics: TrainingMetric[];
  trainingStatus: string;

  // ── training config ──
  selectedLabels: string[];
  setSelectedLabels: (labels: string[]) => void;
  valSplitRatio: number;
  setValSplitRatio: (ratio: number) => void;
  baseModel: string;
  customModel: string;
  setCustomModel: (model: string) => void;
  useCustomModel: boolean;
  setUseCustomModel: (use: boolean) => void;

  // ── actions ──
  startTraining: () => Promise<void>;
  stopTraining: () => Promise<void>;

  // ── feedback ──
  toast: ReturnType<typeof useToast>;
}

export const MonitorContext = createContext<MonitorContextValue | null>(null);

/** Access the shared Monitor state. Must be used within a `MonitorProvider`. */
export const useMonitor = (): MonitorContextValue => {
  const ctx = useContext(MonitorContext);
  if (!ctx) {
    throw new Error("useMonitor must be used within a MonitorProvider");
  }
  return ctx;
};
