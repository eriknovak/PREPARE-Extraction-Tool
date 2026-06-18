import { useEffect, useState } from "react";
import classNames from "classnames";

import type { MonitorDatasetStats } from "types";

import styles from "./styles.module.css";

interface Props {
  datasetId: number | null;
  datasetStats: MonitorDatasetStats | null;
  onChange: (selectedLabels: string[]) => void;
}

const LabelSelector = ({ datasetStats, onChange }: Props) => {
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  // default select all labels when dataset changes
  useEffect(() => {
    if (!datasetStats?.labelDistribution) return;

    const allLabels = Object.keys(datasetStats.labelDistribution);
    setSelectedLabels(allLabels);
    onChange(allLabels); // immediately notify parent
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetStats]);

  const toggleLabel = (label: string) => {
    setSelectedLabels((prev) => {
      const updated = prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label];

      onChange(updated); // instant update to parent

      return updated;
    });
  };

  if (!datasetStats?.labelDistribution) return null;

  return (
    <div className={styles["label-selector"]}>
      <h3 className={styles["label-selector__title"]}>Select labels for training</h3>

      <div className={styles["label-selector__list"]}>
        {Object.keys(datasetStats.labelDistribution).map((label) => {
          const active = selectedLabels.includes(label);

          return (
            <button
              key={label}
              type="button"
              onClick={() => toggleLabel(label)}
              className={classNames(styles["label-toggle"], {
                [styles["label-toggle--active"]]: active,
              })}
            >
              {label} ({datasetStats.labelDistribution[label]})
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default LabelSelector;
