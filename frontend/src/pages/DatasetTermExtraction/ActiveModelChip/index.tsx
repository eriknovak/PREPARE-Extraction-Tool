import React from "react";
import { Link } from "react-router-dom";
import styles from "./styles.module.css";

interface ActiveModelChipProps {
  modelName: string;
}

const ActiveModelChip: React.FC<ActiveModelChipProps> = ({ modelName }) => {
  return (
    <span className={styles.chip} title="Extraction model (set in Monitor)">
      <span className={styles.chipLabel}>Model:</span>
      <span className={styles.chipName}>{modelName}</span>
      <Link to="/monitor" className={styles.chipLink}>
        Monitor
      </Link>
    </span>
  );
};

export default ActiveModelChip;
