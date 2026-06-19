import Button from "@components/Button";
import Card from "@components/Card";

import { useMonitor } from "../hooks/useMonitor";
import styles from "./DatasetSelector.module.css";

/**
 * Dataset picker. Selecting a dataset scopes everything shown below it
 * (both the Comparison and Training views) to that dataset.
 */
const DatasetSelector = () => {
  const { datasets, selectedDatasetId, selectDataset } = useMonitor();

  return (
    <Card title="Dataset">
      {datasets.length > 0 ? (
        <div className={styles["dataset-list"]}>
          {datasets.map((d) => (
            <Button
              key={d.id}
              onClick={() => selectDataset(d.id)}
              variant={selectedDatasetId === d.id ? "primary" : "outline"}
            >
              {d.name}
            </Button>
          ))}
        </div>
      ) : (
        <p className={styles.muted}>No datasets available.</p>
      )}
    </Card>
  );
};

export default DatasetSelector;
