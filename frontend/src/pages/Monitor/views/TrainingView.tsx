import classNames from "classnames";

import Button from "@components/Button";
import Card from "@components/Card";
import ProgressBar from "@components/ProgressBar";
import Select from "@components/Select";
import StatCard from "@components/StatCard";

import LabelSelector from "../LabelSelector";
import TrainingLossChart from "../charts/TrainingLossChart";
import { DEFAULT_MODEL, useMonitor } from "../hooks/useMonitor";
import styles from "./TrainingView.module.css";

const SPLIT_OPTIONS = [
  { value: "0", label: "No split (100% train)" },
  { value: "0.1", label: "90 / 10" },
  { value: "0.2", label: "80 / 20" },
  { value: "0.3", label: "70 / 30" },
];

/**
 * Training view — trains a new model on the selected dataset: dataset stats and
 * label selection, train/eval split, primary model selection, start/stop, and
 * live training progression (progress bar + loss curve).
 */
const TrainingView = () => {
  const {
    selectedDatasetId,
    datasetStats,
    setSelectedLabels,
    valSplitRatio,
    setValSplitRatio,
    customModel,
    setCustomModel,
    useCustomModel,
    setUseCustomModel,
    isTraining,
    progress,
    trainingMetrics,
    trainingStatus,
    startTraining,
    stopTraining,
    runs,
  } = useMonitor();

  if (!selectedDatasetId) {
    return <Card title="Training">Select a dataset to train a model.</Card>;
  }

  return (
    <div className={styles.view}>
      {/* STATS + LABELS */}
      {datasetStats && (
        <Card title="Labels & statistics">
          <div className={styles.stats}>
            <StatCard label="Records" value={datasetStats.totalRecords} />
            <StatCard label="Terms" value={datasetStats.totalTerms} />
          </div>

          <LabelSelector datasetStats={datasetStats} onChange={setSelectedLabels} />
        </Card>
      )}

      {/* TRAINING CONFIG */}
      <Card title="Training configuration">
        {/* Model selector */}
        <div className={styles.field}>
          <p className={styles.field__label}>Base model</p>

          <div className={styles["radio-group"]}>
            <label className={styles.radio}>
              <input type="radio" checked={!useCustomModel} onChange={() => setUseCustomModel(false)} />
              <span>
                Default: <code className={styles.code}>{DEFAULT_MODEL}</code>
              </span>
            </label>

            <label className={styles.radio}>
              <input type="radio" checked={useCustomModel} onChange={() => setUseCustomModel(true)} />
              <span>Custom model path or HuggingFace ID</span>
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

        {/* Train / Eval split */}
        <div className={styles.field}>
          <p className={styles.field__label}>Train / Eval split</p>

          <Select
            value={String(valSplitRatio)}
            onValueChange={(v) => setValSplitRatio(Number(v))}
            fullWidth={false}
            options={SPLIT_OPTIONS}
          />
        </div>

        {/* Actions */}
        <div className={styles.actions}>
          <Button variant="primary" onClick={startTraining} disabled={isTraining || !selectedDatasetId}>
            Start
          </Button>
          <Button variant="danger" onClick={stopTraining} disabled={!isTraining}>
            Stop
          </Button>

          {trainingStatus && (
            <p className={classNames(styles.status, { [styles["status--active"]]: isTraining })}>{trainingStatus}</p>
          )}
        </div>
      </Card>

      {/* LIVE TRAINING PROGRESSION */}
      <Card title="Training progress">
        <div className={styles.progress}>
          <ProgressBar value={progress} />
        </div>

        <TrainingLossChart metrics={trainingMetrics} isTraining={isTraining} hasRuns={runs.length > 0} />
      </Card>
    </div>
  );
};

export default TrainingView;
