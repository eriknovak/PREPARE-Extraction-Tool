import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleQuestion } from "@fortawesome/free-solid-svg-icons";

import styles from "./MonitorHeader.module.css";

/**
 * Standalone page header for the Monitor page. Matches the style of the
 * workflow page headers (title + info tooltip) but omits the back/forward
 * workflow navigation, since Monitor is not part of the linear workflow.
 */
const MonitorHeader = () => {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>
        Monitoring Dashboard
        <span className={styles["info-tooltip"]}>
          <FontAwesomeIcon icon={faCircleQuestion} className={styles["info-tooltip__icon"]} />
          <span className={styles["info-tooltip__content"]}>
            <strong>Monitoring Dashboard</strong>
            <p>
              Train custom NER models and compare their performance. Pick a dataset, then switch
              between:
            </p>
            <ul>
              <li>
                <strong>Comparison</strong> — compare runs by loss, evaluation metrics, and a
                runs × labels heatmap, and inspect a single model in detail.
              </li>
              <li>
                <strong>Training</strong> — configure the train/eval split and base model, then
                launch a run and follow its live progress.
              </li>
            </ul>
          </span>
        </span>
      </h1>
      <p className={styles.subtitle}>
        Train NER models, follow live training metrics, and compare run performance.
      </p>
    </header>
  );
};

export default MonitorHeader;
