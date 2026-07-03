import Layout from "@components/Layout";
import { ToastContainer } from "@components/Toast/ToastContainer";
import { usePageTitle } from "@hooks/usePageTitle";

import { useMonitor } from "./hooks/useMonitor";
import MonitorHeader from "./components/MonitorHeader";
import ViewTabs from "./components/ViewTabs";
import ModelsView from "./views/ModelsView";
import TrainingView from "./views/TrainingView";
import styles from "./styles.module.css";

/** Page body — consumes the shared Monitor state and renders the active view. */
const MonitorContent = () => {
  const { activeView, toast } = useMonitor();

  return (
    <div className={styles.page}>
      <MonitorHeader />

      <ViewTabs />

      {activeView === "models" ? <ModelsView /> : <TrainingView />}

      <ToastContainer toasts={toast.toasts} onDismiss={toast.dismissToast} duration={5000} />
    </div>
  );
};

const Monitor = () => {
  usePageTitle("Monitoring");

  return (
    <Layout>
      <MonitorContent />
    </Layout>
  );
};

export default Monitor;
