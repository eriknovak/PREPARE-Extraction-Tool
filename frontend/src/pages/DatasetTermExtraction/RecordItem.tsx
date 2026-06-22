import React from "react";
import classNames from "classnames";

import type { Record as RecordType } from "@/types";

import styles from "./styles.module.css";

interface RecordItemProps {
  record: RecordType;
  isSelected: boolean;
  onClick: () => void;
}

const RecordItem: React.FC<RecordItemProps> = ({ record, isSelected, onClick }) => {
  return (
    <div
      className={classNames(styles["record-item"], { [styles["record-item--selected"]]: isSelected })}
      onClick={onClick}
    >
      <div className={styles["record-item__header"]}>
        <span className={styles["record-item__id"]}>Patient ID: {record.patient_id}</span>
        <span className={styles["record-item__id"]}>{record.seq_number && `#${record.seq_number}`}</span>
      </div>
      <div className={styles["record-item__preview"]}>
        {record.text.slice(0, 150)}
        {record.text.length > 150 ? "..." : ""}
      </div>
      <div className={styles["record-item__status"]}>
        {record.reviewed && (
          <span className={classNames(styles["status-badge"], styles["status-badge--reviewed"])}>Reviewed</span>
        )}
        <span className={styles["record-item__term-count"]}>
          {record.source_term_count > 0
            ? `${record.source_term_count} term${record.source_term_count !== 1 ? "s" : ""}`
            : "No terms"}
        </span>
      </div>
    </div>
  );
};

export default RecordItem;
