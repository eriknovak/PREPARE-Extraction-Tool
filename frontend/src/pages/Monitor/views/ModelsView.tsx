import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCheck, faPen, faTrash, faXmark } from "@fortawesome/free-solid-svg-icons";

import { deleteRun, getModelDetail, getModels, getRunMetrics, setActiveModel, updateRun } from "@api/monitoring";
import Button from "@components/Button";
import Card from "@components/Card";
import { ConfirmDialog } from "@components/ConfirmDialog";
import { BarChart, ChartState, LineChart } from "@components/charts";
import type { ModelDetailResponse, ModelSummary, TrainingMetric } from "types";

import { buildLossSeries, formatEpoch, formatLoss } from "../chartData";
import { useMonitor } from "../hooks/useMonitor";
import styles from "./ModelsView.module.css";

// ─────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────

const CHART_HEIGHT = 280;

const DEFAULT_ROW: ModelSummary = {
  id: -1,
  name: "Default (bioner)",
  version: "default",
  is_active: false,
};

// ─────────────────────────────────────────────────────────────────
// Presentational component — pure props, no API calls
// ─────────────────────────────────────────────────────────────────

export interface ModelDetailProps {
  detail: ModelDetailResponse | null;
  metrics: TrainingMetric[];
}

/** Renders training loss curve + per-label eval for a trained model.
 *  Pass `detail=null` to show the Default-model empty state. */
export const ModelDetail = ({ detail, metrics }: ModelDetailProps) => {
  if (!detail) {
    return (
      <Card title="Model detail">
        <p className={styles.emptyDetail}>Untrained default model — no training history.</p>
      </Card>
    );
  }

  // ── Loss curve ──
  const hasLoss = metrics.length > 0;
  const { xData, loss: lossValues, evalLoss, hasStep } = buildLossSeries(metrics);
  const xName = hasStep ? "Step" : "Epoch";
  const hasEvalLoss = evalLoss.some((v) => v !== null);

  const lossSeries = [
    { name: "Train loss", data: lossValues, area: true },
    ...(hasEvalLoss ? [{ name: "Eval loss", data: evalLoss }] : []),
  ];

  // ── Per-label eval ──
  const labels = detail.labels ?? [];
  const hasEval =
    labels.length > 0 &&
    (Object.keys(detail.per_label_trained).length > 0 || Object.keys(detail.per_label_baseline).length > 0);

  const baseF1Values = labels.map((l) => detail.per_label_baseline[l]?.exact_f1 ?? 0);
  const trainedF1Values = labels.map((l) => detail.per_label_trained[l]?.exact_f1 ?? 0);

  // ── Stats ──
  const stats = detail.train_stats;
  const labelDist = stats?.label_distribution ?? {};
  const distLabels = Object.keys(labelDist);

  return (
    <div className={styles.detail}>
      {/* ── Training loss chart ── */}
      <Card title="Training loss">
        {hasLoss ? (
          <LineChart
            xData={xData as (string | number)[]}
            xName={xName}
            yName="Loss"
            height={CHART_HEIGHT}
            xAxisFormatter={formatEpoch}
            valueFormatter={formatLoss}
            series={lossSeries}
          />
        ) : (
          <ChartState
            variant="empty"
            title="No loss data"
            message="This model has no recorded training metrics."
            height={CHART_HEIGHT}
          />
        )}
      </Card>

      {/* ── Base vs trained per-label ── */}
      <Card title="Per-label evaluation (base vs trained)">
        {hasEval ? (
          <>
            <BarChart
              categories={labels}
              height={CHART_HEIGHT}
              yMin={0}
              yMax={1}
              xLabelRotate={labels.length > 5 ? 30 : 0}
              series={[
                { name: "Base (exact F1)", data: baseF1Values },
                { name: "Trained (exact F1)", data: trainedF1Values },
              ]}
            />
            <table className={styles.perLabelTable}>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Base F1</th>
                  <th>Trained F1</th>
                  <th>Δ</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((label) => {
                  const base = detail.per_label_baseline[label]?.exact_f1 ?? null;
                  const trained = detail.per_label_trained[label]?.exact_f1 ?? null;
                  const delta = base != null && trained != null ? trained - base : null;
                  return (
                    <tr key={label}>
                      <td>{label}</td>
                      <td>{base != null ? `${(base * 100).toFixed(1)}%` : "—"}</td>
                      <td>{trained != null ? `${(trained * 100).toFixed(1)}%` : "—"}</td>
                      <td className={delta == null ? undefined : delta >= 0 ? styles.deltaPos : styles.deltaNeg}>
                        {delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        ) : (
          <ChartState
            variant="empty"
            title="No evaluation data"
            message="Run an evaluation on this model to see per-label scores."
            height={CHART_HEIGHT}
          />
        )}
      </Card>

      {/* ── Training stats ── */}
      <Card title="Training statistics">
        <div className={styles.statsGrid}>
          {stats?.record_count != null && (
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Records</span>
              <span className={styles.statValue}>{stats.record_count.toLocaleString()}</span>
            </div>
          )}
          {stats?.term_count != null && (
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Terms</span>
              <span className={styles.statValue}>{stats.term_count.toLocaleString()}</span>
            </div>
          )}
          {detail.train_dataset_ids.length > 0 && (
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Train datasets</span>
              <span className={styles.statValue}>{detail.train_dataset_ids.join(", ")}</span>
            </div>
          )}
          {detail.eval_dataset_ids.length > 0 && (
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Eval datasets</span>
              <span className={styles.statValue}>{detail.eval_dataset_ids.join(", ")}</span>
            </div>
          )}
        </div>
      </Card>

      {/* ── Label coverage ── */}
      {(labels.length > 0 || distLabels.length > 0) && (
        <Card title="Label coverage">
          <div className={styles.tagList}>
            {labels.map((label) => (
              <span key={label} className={styles.tag}>
                {label}
                {labelDist[label] != null && <span className={styles.tagCount}>{labelDist[label]}</span>}
              </span>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────
// Container — data-fetching wrapper
// ─────────────────────────────────────────────────────────────────

const ModelsView = () => {
  const { toast } = useMonitor();

  const [models, setModels] = useState<ModelSummary[]>([]);
  const [activeModelId, setActiveModelId] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<ModelDetailResponse | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);

  // Rename state
  const [renameId, setRenameId] = useState<number | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  // Delete confirm
  const [deleteTarget, setDeleteTarget] = useState<ModelSummary | null>(null);

  // ── Load model list ──
  const reloadModels = useCallback(async () => {
    const list = await getModels();
    setModels(list);
    const active = list.find((m) => m.is_active);
    setActiveModelId(active ? active.id : null); // null => Default is active
  }, []);

  useEffect(() => {
    reloadModels().catch(() => toast.showToast("Failed to load models", "error"));
  }, [reloadModels, toast]);

  // Default row shows active when no trained model is active.
  const rows = useMemo<ModelSummary[]>(
    () => [{ ...DEFAULT_ROW, is_active: activeModelId === null }, ...models],
    [models, activeModelId]
  );

  // ── Load detail for selected row ──
  useEffect(() => {
    if (selectedId == null || selectedId < 0) {
      setDetail(null);
      setMetrics([]);
      return;
    }

    let cancelled = false;
    setLoadingDetail(true);

    function modelRunMetrics(id: number): Promise<TrainingMetric[]> {
      const m = models.find((x) => x.id === id);
      return m?.run_id ? getRunMetrics(m.run_id) : Promise.resolve([]);
    }

    Promise.all([getModelDetail(selectedId), modelRunMetrics(selectedId)])
      .then(([d, m]) => {
        if (cancelled) return;
        setDetail(d);
        setMetrics(m);
      })
      .catch(() => {
        if (cancelled) return;
        toast.showToast("Failed to load model detail", "error");
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId, models, toast]);

  // ── Handlers ──
  const handleUse = useCallback(
    async (model: ModelSummary) => {
      try {
        const next = model.id < 0 ? null : model.id;
        await setActiveModel(next);
        await reloadModels();
        toast.showToast("Active model updated", "success");
      } catch (err) {
        const msg = err instanceof Error ? err.message : "";
        const isConflict =
          msg === "Cannot change the model while an extraction job is running" || msg.startsWith("HTTP 409");
        toast.showToast(
          isConflict ? "Cannot change the model while an extraction job is running" : "Failed to set the active model",
          "error"
        );
      }
    },
    [reloadModels, toast]
  );

  const handleRenameStart = useCallback((model: ModelSummary) => {
    setRenameId(model.id);
    setRenameDraft(model.name);
    // focus the input on next tick
    setTimeout(() => renameInputRef.current?.focus(), 0);
  }, []);

  const handleRenameConfirm = useCallback(
    async (model: ModelSummary) => {
      const trimmed = renameDraft.trim();
      if (!trimmed) {
        setRenameId(null);
        return;
      }
      if (!model.run_id) {
        setRenameId(null);
        return;
      }
      try {
        await updateRun(model.run_id, { name: trimmed });
        setRenameId(null);
        await reloadModels();
        toast.showToast("Model renamed", "success");
      } catch {
        toast.showToast("Failed to rename model", "error");
      }
    },
    [renameDraft, reloadModels, toast]
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTarget || !deleteTarget.run_id) return;
    try {
      await deleteRun(deleteTarget.run_id);
      if (selectedId === deleteTarget.id) {
        setSelectedId(null);
      }
      setDeleteTarget(null);
      await reloadModels();
      toast.showToast("Model deleted", "success");
    } catch {
      toast.showToast("Failed to delete model", "error");
    }
  }, [deleteTarget, selectedId, reloadModels, toast]);

  return (
    <div className={styles.layout}>
      {/* ── Model list ── */}
      <ul className={styles.list} role="listbox" aria-label="Trained models">
        {rows.map((m) => {
          const isRenaming = renameId === m.id;
          const isReal = m.id >= 0;

          return (
            <li key={m.id} className={styles.listItem}>
              {isRenaming ? (
                <>
                  <input
                    ref={renameInputRef}
                    className={styles.renameInput}
                    value={renameDraft}
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRenameConfirm(m);
                      if (e.key === "Escape") setRenameId(null);
                    }}
                    aria-label="Rename model"
                  />
                  <div className={styles.actions}>
                    <Button
                      size="icon"
                      variant="ghost"
                      title="Confirm rename"
                      aria-label="Confirm rename"
                      onClick={() => handleRenameConfirm(m)}
                    >
                      <FontAwesomeIcon icon={faCheck} />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      title="Cancel rename"
                      aria-label="Cancel rename"
                      onClick={() => setRenameId(null)}
                    >
                      <FontAwesomeIcon icon={faXmark} />
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    className={styles.row}
                    role="option"
                    aria-selected={selectedId === m.id}
                    onClick={() => setSelectedId(m.id)}
                  >
                    <span className={styles.name}>{m.name}</span>
                    {m.score != null && <span className={styles.score}>{(m.score * 100).toFixed(1)}%</span>}
                    {m.is_active && <span className={styles.active}>● active</span>}
                  </button>

                  <div className={styles.actions}>
                    <Button
                      size="small"
                      variant={m.is_active ? "primary" : "outline"}
                      title="Use this model for extraction"
                      onClick={() => handleUse(m)}
                    >
                      Use
                    </Button>
                    {isReal && (
                      <>
                        <Button
                          size="icon"
                          variant="ghost"
                          title="Rename model"
                          aria-label="Rename model"
                          onClick={() => handleRenameStart(m)}
                        >
                          <FontAwesomeIcon icon={faPen} />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          colorScheme="danger"
                          title="Delete model"
                          aria-label="Delete model"
                          onClick={() => setDeleteTarget(m)}
                        >
                          <FontAwesomeIcon icon={faTrash} />
                        </Button>
                      </>
                    )}
                  </div>
                </>
              )}
            </li>
          );
        })}
      </ul>

      {/* ── Detail pane ── */}
      <section className={styles.detail}>
        {loadingDetail ? (
          <Card title="Model detail">
            <ChartState variant="loading" message="Loading model detail…" height={CHART_HEIGHT} />
          </Card>
        ) : selectedId == null ? (
          <p className={styles.placeholder}>Select a model to see its training and evaluation detail.</p>
        ) : selectedId < 0 ? (
          <ModelDetail detail={null} metrics={[]} />
        ) : (
          <ModelDetail detail={detail} metrics={metrics} />
        )}
      </section>

      {/* ── Delete confirm dialog ── */}
      <ConfirmDialog
        isOpen={deleteTarget != null}
        title="Delete model"
        message={deleteTarget ? `Are you sure you want to delete "${deleteTarget.name}"? This cannot be undone.` : ""}
        confirmText="Delete"
        variant="danger"
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
};

export default ModelsView;
