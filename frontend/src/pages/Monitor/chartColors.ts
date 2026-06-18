/** Chart color palette for the monitoring dashboard.
 *
 * These hex values mirror the design tokens in `src/index.css` and are used
 * directly by recharts and SVG elements that cannot consume CSS variables.
 */
export const CHART = {
  loss: "#dc2626",
  precision: "#144d70",
  recall: "#3b82f6",
  exactF1: "#269b6f",
  relaxedF1: "#f59e0b",
} as const;
