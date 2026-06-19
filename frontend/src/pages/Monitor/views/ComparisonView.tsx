import { useCallback, useEffect, useMemo, useState } from "react";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleCheck, faPen, faStar, faTrash, faTrophy } from "@fortawesome/free-solid-svg-icons";

import {
  deleteRun,
  getDatasetActiveModel,
  getDatasetRunsPaged,
  getRunMetrics,
  setDatasetActiveModel,
  updateRun,
} from "@api/monitoring";
import Button from "@components/Button";
import Card from "@components/Card";
import ConfirmDialog from "@components/ConfirmDialog";
import Select from "@components/Select";
import Table, { type Column, type SortState } from "@components/Table";
import { BarChart, ChartState, LineChart } from "@components/charts";
import type { EvaluationResponse, MonitorRun, PaginationMetadata, TrainingMetric } from "types";

import ModelComparisonHeatmap from "../charts/ModelComparisonHeatmap";
import PerformanceChart from "../charts/PerformanceChart";
import {
  METRIC_OPTIONS,
  formatEpoch,
  formatLoss,
  normalizeLabel,
  readMetric,
  type MetricMode,
} from "../chartData";
import { useMonitor } from "../hooks/useMonitor";
import styles from "./ComparisonView.module.css";

const CHART_HEIGHT = 280;
const RUNS_PAGE_SIZE = 20;

/** Aggregate rows returned by the backend that should not appear as labels. */
const AGGREGATE_LABELS = new Set(["micro avg", "macro avg", "weighted avg"]);

interface RunLoss {
  runId: number;
  metrics: TrainingMetric[];
}

/** Display label for a run: its name, falling back to `Run #id`. */
const runDisplayName = (run: MonitorRun): string => run.name?.trim() || `Run #${run.run_id}`;

/** Format a macro-F1 score (0–1) as a percentage, or a dash when absent. */
const formatScore = (score: number | null | undefined): string =>
  typeof score === "number" ? `${(score * 100).toFixed(1)}%` : "—";

/** Format an ISO timestamp compactly, or a dash when absent. */
const formatDate = (value: string | null | undefined): string => {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
};

/** Build a multi-series loss overlay: one line per run, aligned on the epoch axis. */
const buildLossOverlay = (runLosses: RunLoss[], nameOf: Map<number, string>) => {
  const withData = runLosses.filter((r) => r.metrics.length > 0);
  const epochs = Array.from(new Set(withData.flatMap((r) => r.metrics.map((m) => m.epoch)))).sort(
    (a, b) => a - b
  );
  const series = withData.map((r) => {
    const byEpoch = new Map(r.metrics.map((m) => [m.epoch, m.loss]));
    return {
      name: nameOf.get(r.runId) ?? `Run #${r.runId}`,
      // null renders a gap where a run has no point for that epoch
      data: epochs.map((e) => byEpoch.get(e) ?? null) as unknown as number[],
    };
  });
  return { epochs, series };
};

/** Build a grouped bar comparison: one series per run, one category per label. */
const buildEvalComparison = (evaluations: EvaluationResponse[], metric: MetricMode) => {
  const labels = Array.from(
    new Set(
      evaluations.flatMap((run) =>
        Object.keys(run?.per_label ?? {})
          .filter((label) => !AGGREGATE_LABELS.has(label.toLowerCase()))
          .map(normalizeLabel)
      )
    )
  ).sort();

  const series = evaluations.map((run) => {
    const byLabel = new Map<string, number>();
    Object.entries(run?.per_label ?? {}).forEach(([label, m]) => {
      if (AGGREGATE_LABELS.has(label.toLowerCase())) return;
      const norm = normalizeLabel(label);
      if (!byLabel.has(norm)) byLabel.set(norm, readMetric(m, metric));
    });
    return {
      name: `Run #${run.run_id}`,
      data: labels.map((l) => byLabel.get(l) ?? 0),
    };
  });

  return { labels, series };
};

/**
 * Comparison view — manages and compares model/run performance for the selected
 * dataset: a leaderboard, a sortable/filterable run table with rename/delete and
 * "preferred" designation, loss/evaluation overlays, a runs × labels heatmap,
 * and a single-model detail section.
 */
const ComparisonView = () => {
  const {
    selectedDatasetId,
    selectedRun,
    setSelectedRun,
    evaluation,
    evaluationLoading,
    evaluations,
    evaluationsLoading,
    toast,
  } = useMonitor();

  const [evalMetric, setEvalMetric] = useState<MetricMode>("exact_f1");

  // ── Run list (paginated, owned by this view) ──
  const [runList, setRunList] = useState<MonitorRun[]>([]);
  const [pagination, setPagination] = useState<PaginationMetadata | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [sort, setSort] = useState<SortState>({ key: "created_at", direction: "desc" });
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [labelFilter, setLabelFilter] = useState<string>("");

  // Pending rename / delete confirmation targets.
  const [renameTarget, setRenameTarget] = useState<MonitorRun | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<MonitorRun | null>(null);

  // Historical loss for the runs currently loaded (bounded by the visible pages).
  const [allLosses, setAllLosses] = useState<RunLoss[]>([]);
  const [allLossesLoading, setAllLossesLoading] = useState(false);

  // The model id currently selected for extraction on this dataset (null = default).
  const [activeModelId, setActiveModelId] = useState<number | null>(null);

  // Historical loss for the single run picked in the detail section.
  const [runLoss, setRunLoss] = useState<TrainingMetric[]>([]);
  const [runLossLoading, setRunLossLoading] = useState(false);

  // Fetch the first page of runs whenever the dataset changes.
  useEffect(() => {
    if (!selectedDatasetId) {
      setRunList([]);
      setPagination(null);
      return;
    }
    let cancelled = false;
    setRunsLoading(true);
    getDatasetRunsPaged(selectedDatasetId, 1, RUNS_PAGE_SIZE)
      .then((res) => {
        if (cancelled) return;
        setRunList(res.runs);
        setPagination(res.pagination);
      })
      .catch(() => {
        if (cancelled) return;
        setRunList([]);
        setPagination(null);
        toast.showToast("Failed to load runs", "error");
      })
      .finally(() => {
        if (!cancelled) setRunsLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId]);

  // Fetch the dataset's active extraction model whenever the dataset changes.
  useEffect(() => {
    if (!selectedDatasetId) {
      setActiveModelId(null);
      return;
    }
    let cancelled = false;
    getDatasetActiveModel(selectedDatasetId)
      .then((res) => {
        if (!cancelled) setActiveModelId(res.active_model?.id ?? null);
      })
      .catch(() => {
        if (!cancelled) setActiveModelId(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedDatasetId]);

  const loadMore = useCallback(() => {
    if (!selectedDatasetId || !pagination) return;
    const nextPage = pagination.page + 1;
    setLoadingMore(true);
    getDatasetRunsPaged(selectedDatasetId, nextPage, RUNS_PAGE_SIZE)
      .then((res) => {
        setRunList((prev) => {
          const seen = new Set(prev.map((r) => r.run_id));
          return [...prev, ...res.runs.filter((r) => !seen.has(r.run_id))];
        });
        setPagination(res.pagination);
      })
      .catch(() => toast.showToast("Failed to load more runs", "error"))
      .finally(() => setLoadingMore(false));
  }, [selectedDatasetId, pagination, toast]);

  const runIdsKey = useMemo(() => runList.map((r) => r.run_id).join(","), [runList]);
  const nameByRunId = useMemo(
    () => new Map(runList.map((r) => [r.run_id, runDisplayName(r)])),
    [runList]
  );

  // Fetch loss curves for the loaded runs whenever the set changes (bounded).
  useEffect(() => {
    if (runList.length === 0) {
      setAllLosses([]);
      return;
    }
    let cancelled = false;
    setAllLossesLoading(true);
    Promise.all(
      runList.map((r) =>
        getRunMetrics(r.run_id)
          .then((metrics) => ({ runId: r.run_id, metrics }))
          .catch(() => ({ runId: r.run_id, metrics: [] as TrainingMetric[] }))
      )
    )
      .then((results) => {
        if (!cancelled) setAllLosses(results);
      })
      .catch(() => {
        if (!cancelled) toast.showToast("Failed to load run loss curves", "error");
      })
      .finally(() => {
        if (!cancelled) setAllLossesLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIdsKey]);

  // Fetch the picked run's loss curve.
  useEffect(() => {
    if (selectedRun === null) {
      setRunLoss([]);
      return;
    }
    let cancelled = false;
    setRunLossLoading(true);
    getRunMetrics(selectedRun)
      .then((metrics) => {
        if (!cancelled) setRunLoss(metrics);
      })
      .catch(() => {
        if (!cancelled) {
          setRunLoss([]);
          toast.showToast("Failed to load run loss curve", "error");
        }
      })
      .finally(() => {
        if (!cancelled) setRunLossLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRun]);

  const lossOverlay = useMemo(() => buildLossOverlay(allLosses, nameByRunId), [allLosses, nameByRunId]);
  const evalComparison = useMemo(
    () => buildEvalComparison(evaluations, evalMetric),
    [evaluations, evalMetric]
  );

  // ── Best run: the preferred one, else the highest-scoring loaded run. ──
  const bestRunId = useMemo(() => {
    const preferred = runList.find((r) => r.preferred);
    if (preferred) return preferred.run_id;
    let best: MonitorRun | null = null;
    for (const r of runList) {
      if (typeof r.score !== "number") continue;
      if (best === null || (r.score ?? 0) > (best.score ?? 0)) best = r;
    }
    return best?.run_id ?? null;
  }, [runList]);

  const statusOptions = useMemo(() => {
    const set = new Set(runList.map((r) => r.status).filter((s): s is string => !!s));
    return [{ value: "", label: "All statuses" }, ...Array.from(set).sort().map((s) => ({ value: s, label: s }))];
  }, [runList]);

  const labelOptions = useMemo(() => {
    const set = new Set(runList.flatMap((r) => r.labels ?? []));
    return [{ value: "", label: "All labels" }, ...Array.from(set).sort().map((l) => ({ value: l, label: l }))];
  }, [runList]);

  const visibleRuns = useMemo(() => {
    const filtered = runList.filter((r) => {
      if (statusFilter && r.status !== statusFilter) return false;
      if (labelFilter && !(r.labels ?? []).includes(labelFilter)) return false;
      return true;
    });
    const dir = sort.direction === "asc" ? 1 : -1;
    const sorted = [...filtered].sort((a, b) => {
      let av: number | string;
      let bv: number | string;
      switch (sort.key) {
        case "score":
          av = a.score ?? -1;
          bv = b.score ?? -1;
          break;
        case "status":
          av = a.status ?? "";
          bv = b.status ?? "";
          break;
        case "created_at":
        default:
          av = a.created_at ? new Date(a.created_at).getTime() : 0;
          bv = b.created_at ? new Date(b.created_at).getTime() : 0;
          break;
      }
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
    return sorted;
  }, [runList, statusFilter, labelFilter, sort]);

  const leaderboard = useMemo(
    () =>
      [...runList]
        .filter((r) => typeof r.score === "number")
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        .slice(0, 5),
    [runList]
  );

  const handleSort = (key: string) => {
    setSort((prev) =>
      prev.key === key
        ? { key, direction: prev.direction === "asc" ? "desc" : "asc" }
        : { key, direction: "desc" }
    );
  };

  const handleTogglePreferred = useCallback(
    async (run: MonitorRun) => {
      const next = !run.preferred;
      try {
        await updateRun(run.run_id, { preferred: next });
        // NOTE: a later PR wires the preferred run into the extraction flow.
        setRunList((prev) =>
          prev.map((r) => ({
            ...r,
            preferred: r.run_id === run.run_id ? next : next ? false : r.preferred,
          }))
        );
        toast.showToast(next ? "Marked as preferred run" : "Cleared preferred run", "success");
      } catch {
        toast.showToast("Failed to update preferred run", "error");
      }
    },
    [toast]
  );

  const handleUseForExtraction = useCallback(
    async (run: MonitorRun) => {
      if (!selectedDatasetId || run.model_id == null) return;
      const isActive = activeModelId === run.model_id;
      const nextId = isActive ? null : run.model_id;
      try {
        const res = await setDatasetActiveModel(selectedDatasetId, nextId);
        setActiveModelId(res.active_model?.id ?? null);
        toast.showToast(
          nextId === null ? "Reverted to default extraction model" : "Model set for extraction",
          "success"
        );
      } catch {
        toast.showToast("Failed to update extraction model", "error");
      }
    },
    [selectedDatasetId, activeModelId, toast]
  );

  const handleRename = useCallback(
    async (name?: string) => {
      const target = renameTarget;
      setRenameTarget(null);
      if (!target) return;
      const trimmed = (name ?? "").trim();
      try {
        const updated = await updateRun(target.run_id, { name: trimmed || null });
        setRunList((prev) => prev.map((r) => (r.run_id === target.run_id ? { ...r, name: updated.name } : r)));
        toast.showToast("Run renamed", "success");
      } catch {
        toast.showToast("Failed to rename run", "error");
      }
    },
    [renameTarget, toast]
  );

  const handleDelete = useCallback(async () => {
    const target = deleteTarget;
    setDeleteTarget(null);
    if (!target) return;
    try {
      await deleteRun(target.run_id);
      setRunList((prev) => prev.filter((r) => r.run_id !== target.run_id));
      setPagination((prev) => (prev ? { ...prev, total: Math.max(0, prev.total - 1) } : prev));
      if (selectedRun === target.run_id) setSelectedRun(null);
      toast.showToast("Run deleted", "success");
    } catch {
      toast.showToast("Failed to delete run", "error");
    }
  }, [deleteTarget, selectedRun, setSelectedRun, toast]);

  const runOptions = useMemo(
    () =>
      runList.map((r) => ({
        value: String(r.run_id),
        label: r.status ? `${runDisplayName(r)} · ${r.status}` : runDisplayName(r),
      })),
    [runList]
  );

  const columns: Column<MonitorRun>[] = useMemo(
    () => [
      {
        key: "name",
        header: "Run",
        render: (run) => (
          <span className={styles.runName}>
            {run.run_id === bestRunId && (
              <FontAwesomeIcon icon={faTrophy} className={styles.bestIcon} title="Best model" />
            )}
            {run.preferred && (
              <FontAwesomeIcon icon={faStar} className={styles.preferredIcon} title="Preferred run" />
            )}
            {run.model_id != null && activeModelId === run.model_id && (
              <FontAwesomeIcon
                icon={faCircleCheck}
                className={styles.activeIcon}
                title="Active for extraction"
              />
            )}
            {runDisplayName(run)}
          </span>
        ),
      },
      { key: "status", header: "Status", sortable: true },
      {
        key: "score",
        header: "Macro-F1",
        sortable: true,
        align: "right",
        render: (run) => formatScore(run.score),
      },
      {
        key: "base_model",
        header: "Base model",
        render: (run) => run.base_model ?? "—",
      },
      {
        key: "created_at",
        header: "Created",
        sortable: true,
        render: (run) => formatDate(run.created_at),
      },
      {
        key: "actions",
        header: "",
        align: "right",
        render: (run) => (
          <div className={styles.rowActions}>
            <Button
              size="icon"
              variant="ghost"
              colorScheme={run.model_id != null && activeModelId === run.model_id ? "primary" : "default"}
              title={
                run.model_id == null
                  ? "No trained model for this run"
                  : activeModelId === run.model_id
                    ? "Active for extraction (click to use default)"
                    : "Use this model for extraction"
              }
              aria-label="Use model for extraction"
              disabled={run.model_id == null}
              onClick={() => handleUseForExtraction(run)}
            >
              <FontAwesomeIcon icon={faCircleCheck} />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              colorScheme={run.preferred ? "primary" : "default"}
              title={run.preferred ? "Unset preferred" : "Set as preferred"}
              aria-label="Toggle preferred run"
              onClick={() => handleTogglePreferred(run)}
            >
              <FontAwesomeIcon icon={faStar} />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              title="Rename run"
              aria-label="Rename run"
              onClick={() => setRenameTarget(run)}
            >
              <FontAwesomeIcon icon={faPen} />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              colorScheme="danger"
              title="Delete run"
              aria-label="Delete run"
              onClick={() => setDeleteTarget(run)}
            >
              <FontAwesomeIcon icon={faTrash} />
            </Button>
          </div>
        ),
      },
    ],
    [bestRunId, handleTogglePreferred, handleUseForExtraction, activeModelId]
  );

  if (!selectedDatasetId) {
    return <Card title="Comparison">Select a dataset to compare runs.</Card>;
  }

  const hasMore = pagination ? runList.length < pagination.total : false;

  // ── Loss comparison ──
  let lossBody: React.ReactNode;
  if (allLossesLoading) {
    lossBody = <ChartState variant="loading" message="Loading loss curves…" height={CHART_HEIGHT} />;
  } else if (lossOverlay.series.length === 0) {
    lossBody = (
      <ChartState
        variant="empty"
        title="No loss data yet"
        message="Train runs on this dataset to overlay their loss curves here."
        height={CHART_HEIGHT}
      />
    );
  } else {
    lossBody = (
      <LineChart
        xData={lossOverlay.epochs}
        xName="Epoch"
        yName="Loss"
        height={CHART_HEIGHT}
        xAxisFormatter={formatEpoch}
        valueFormatter={formatLoss}
        series={lossOverlay.series}
      />
    );
  }

  // ── Evaluation comparison ──
  let evalBody: React.ReactNode;
  if (evaluationsLoading) {
    evalBody = <ChartState variant="loading" message="Loading run evaluations…" height={CHART_HEIGHT} />;
  } else if (evalComparison.labels.length === 0 || evalComparison.series.length === 0) {
    evalBody = (
      <ChartState
        variant="empty"
        title="Nothing to compare yet"
        message="Evaluate runs on this dataset to compare their per-label scores side by side."
        height={CHART_HEIGHT}
      />
    );
  } else {
    evalBody = (
      <BarChart
        categories={evalComparison.labels}
        height={CHART_HEIGHT}
        yMin={0}
        yMax={1}
        xLabelRotate={evalComparison.labels.length > 5 ? 30 : 0}
        series={evalComparison.series}
      />
    );
  }

  // ── Single-model loss ──
  let detailLossBody: React.ReactNode;
  if (runLossLoading) {
    detailLossBody = <ChartState variant="loading" message="Loading loss curve…" height={CHART_HEIGHT} />;
  } else if (selectedRun === null) {
    detailLossBody = (
      <ChartState
        variant="empty"
        title="No run selected"
        message="Pick a run above to view its loss curve."
        height={CHART_HEIGHT}
      />
    );
  } else if (runLoss.length === 0) {
    detailLossBody = (
      <ChartState
        variant="empty"
        title="No loss data"
        message="This run has no recorded loss metrics."
        height={CHART_HEIGHT}
      />
    );
  } else {
    const { xData, loss } = {
      xData: runLoss.map((m) => m.epoch),
      loss: runLoss.map((m) => m.loss),
    };
    detailLossBody = (
      <LineChart
        xData={xData}
        xName="Epoch"
        yName="Loss"
        showLegend={false}
        height={CHART_HEIGHT}
        xAxisFormatter={formatEpoch}
        valueFormatter={formatLoss}
        series={[{ name: "Loss", data: loss, area: true }]}
      />
    );
  }

  return (
    <div className={styles.view}>
      <Card title="Leaderboard">
        {leaderboard.length === 0 ? (
          <ChartState
            variant="empty"
            title="No scored runs yet"
            message="Evaluate runs on this dataset to rank them by macro-F1."
            height={140}
          />
        ) : (
          <ol className={styles.leaderboard}>
            {leaderboard.map((run, i) => (
              <li
                key={run.run_id}
                className={run.run_id === bestRunId ? styles.leaderboard__best : undefined}
              >
                <span className={styles.leaderboard__rank}>#{i + 1}</span>
                <span className={styles.leaderboard__name}>
                  {run.run_id === bestRunId && (
                    <FontAwesomeIcon icon={faTrophy} className={styles.bestIcon} />
                  )}
                  {runDisplayName(run)}
                </span>
                <span className={styles.leaderboard__score}>{formatScore(run.score)}</span>
              </li>
            ))}
          </ol>
        )}
      </Card>

      <Card
        title="Runs"
        actions={
          <div className={styles.controls}>
            <Select
              value={statusFilter}
              onValueChange={setStatusFilter}
              fullWidth={false}
              size="small"
              options={statusOptions}
            />
            <Select
              value={labelFilter}
              onValueChange={setLabelFilter}
              fullWidth={false}
              size="small"
              options={labelOptions}
            />
          </div>
        }
      >
        <Table
          columns={columns}
          data={visibleRuns}
          keyExtractor={(run) => run.run_id}
          isLoading={runsLoading}
          emptyMessage="No runs for this dataset yet."
          sort={sort}
          onSortChange={handleSort}
          ariaLabel="Training runs"
        />
        {hasMore && (
          <div className={styles.loadMore}>
            <Button variant="outline" size="small" onClick={loadMore} disabled={loadingMore}>
              {loadingMore ? "Loading…" : `Load more (${runList.length} of ${pagination?.total ?? 0})`}
            </Button>
          </div>
        )}
      </Card>

      <Card title="Loss comparison">{lossBody}</Card>

      <Card
        title="Evaluation comparison"
        actions={
          <div className={styles.controls}>
            <label className={styles.controls__label}>Metric</label>
            <Select
              value={evalMetric}
              onValueChange={(v) => setEvalMetric(v as MetricMode)}
              fullWidth={false}
              size="small"
              options={METRIC_OPTIONS}
            />
          </div>
        }
      >
        {evalBody}
      </Card>

      <Card title="Per-label heatmap">
        <ModelComparisonHeatmap evaluations={evaluations} loading={evaluationsLoading} />
      </Card>

      <div className={styles.divider}>Single model</div>

      <Card
        title="Model detail"
        actions={
          <div className={styles.controls}>
            <label className={styles.controls__label}>Run</label>
            <Select
              value={selectedRun !== null ? String(selectedRun) : undefined}
              onValueChange={(v) => setSelectedRun(Number(v))}
              placeholder="Select a run"
              fullWidth={false}
              size="small"
              options={runOptions}
            />
          </div>
        }
      >
        <div className={styles.detail}>
          <div className={styles.detail__block}>
            <h3 className={styles.detail__title}>Loss curve</h3>
            {detailLossBody}
          </div>
          <div className={styles.detail__block}>
            <h3 className={styles.detail__title}>Per-label evaluation</h3>
            <PerformanceChart
              evaluation={evaluation}
              loading={evaluationLoading}
              hasSelectedRun={selectedRun !== null}
            />
          </div>
        </div>
      </Card>

      <ConfirmDialog
        isOpen={renameTarget !== null}
        title="Rename run"
        message="Enter a new name for this run (leave blank to clear)."
        variant="info"
        confirmText="Save"
        showInput
        inputPlaceholder={renameTarget ? runDisplayName(renameTarget) : "Run name"}
        onConfirm={handleRename}
        onCancel={() => setRenameTarget(null)}
      />

      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Delete run"
        message={
          deleteTarget
            ? `Delete "${runDisplayName(deleteTarget)}"? This removes its metrics, evaluation, and trained model artifact. This cannot be undone.`
            : ""
        }
        variant="danger"
        confirmText="Delete"
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
};

export default ComparisonView;
