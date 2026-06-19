import { useState } from "react";
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
 * Training view — trains a new model on one or more selected datasets: dataset
 * pickers (training + optional evaluation), aggregated stats and label
 * selection, train/eval split, base model selection, advanced hyperparameters,
 * start/stop, and live training progression (progress bar + loss curve).
 */
const TrainingView = () => {
  const {
    datasets,
    selectedDatasetId,
    trainingStats,
    trainingDatasetIds,
    setTrainingDatasetIds,
    evalDatasetIds,
    setEvalDatasetIds,
    setSelectedLabels,
    valSplitRatio,
    setValSplitRatio,
    customModel,
    setCustomModel,
    useCustomModel,
    setUseCustomModel,
    numEpochs,
    setNumEpochs,
    learningRate,
    setLearningRate,
    trainBatchSize,
    setTrainBatchSize,
    isTraining,
    progress,
    trainingMetrics,
    trainingStatus,
    startTraining,
    stopTraining,
    runs,
  } = useMonitor();

  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!selectedDatasetId) {
    return <Card title="Training">Select a dataset to train a model.</Card>;
  }

  const datasetOptions = datasets.map((d) => ({ value: String(d.id), label: d.name }));
  // Eval datasets are distinct from training datasets (overlaps are dropped server-side).
  const evalOptions = datasetOptions.filter((o) => !trainingDatasetIds.includes(Number(o.value)));
  const hasEvalDatasets = evalDatasetIds.length > 0;

  return (
    <div className={styles.view}>
      {/* DATASETS */}
      <Card title="Training data">
        <div className={styles.field}>
          <p className={styles.field__label}>Training datasets</p>
          <Select
            multiSelect
            values={trainingDatasetIds.map(String)}
            onValuesChange={(vals) => setTrainingDatasetIds(vals.map(Number))}
            options={datasetOptions}
            placeholder="Select datasets…"
            fullWidth={false}
          />
        </div>

        <div className={styles.field}>
          <p className={styles.field__label}>Evaluation datasets (optional)</p>
          <Select
            multiSelect
            values={evalDatasetIds.map(String)}
            onValuesChange={(vals) => setEvalDatasetIds(vals.map(Number))}
            options={evalOptions}
            placeholder="Held-out split of training data"
            fullWidth={false}
          />
          <p className={styles.hint}>
            {hasEvalDatasets
              ? "Evaluating on the selected datasets."
              : "No eval datasets — a held-out split of the training data is used."}
          </p>
        </div>
      </Card>

      {/* STATS + LABELS */}
      {trainingStats && (
        <Card title="Labels & statistics">
          <div className={styles.stats}>
            <StatCard label="Records" value={trainingStats.totalRecords} />
            <StatCard label="Terms" value={trainingStats.totalTerms} />
          </div>

          <LabelSelector datasetStats={trainingStats} onChange={setSelectedLabels} />
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
            disabled={hasEvalDatasets}
          />
          {hasEvalDatasets && <p className={styles.hint}>Ignored while evaluation datasets are selected.</p>}
        </div>

        {/* Advanced hyperparameters */}
        <div className={styles.field}>
          <button
            type="button"
            className={styles["advanced-toggle"]}
            onClick={() => setShowAdvanced((s) => !s)}
            aria-expanded={showAdvanced}
          >
            {showAdvanced ? "▾" : "▸"} Advanced (hyperparameters)
          </button>

          {showAdvanced && (
            <div className={styles.hyperparams}>
              <label className={styles.hyperparam}>
                <span className={styles.field__label}>Epochs</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={numEpochs}
                  onChange={(e) => setNumEpochs(Math.max(1, Math.round(Number(e.target.value) || 1)))}
                  className={styles.input}
                />
              </label>

              <label className={styles.hyperparam}>
                <span className={styles.field__label}>Learning rate</span>
                <input
                  type="number"
                  min={0}
                  step="0.000001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className={styles.input}
                />
              </label>

              <label className={styles.hyperparam}>
                <span className={styles.field__label}>Batch size</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={trainBatchSize}
                  onChange={(e) => setTrainBatchSize(Math.max(1, Math.round(Number(e.target.value) || 1)))}
                  className={styles.input}
                />
              </label>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className={styles.actions}>
          <Button variant="primary" onClick={startTraining} disabled={isTraining || trainingDatasetIds.length === 0}>
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
