import type { EChartsOption } from "echarts";

import EChart from "../EChart";

export interface LineSeries {
  /** Series label shown in the legend and tooltip. */
  name: string;
  /** Y values, aligned by index with `xData`. */
  data: number[];
  /** Optional explicit line color (defaults to the theme palette). */
  color?: string;
  /** Smooth the line. Defaults to true. */
  smooth?: boolean;
  /** Fill the area under the line. */
  area?: boolean;
}

export interface LineChartProps {
  /** Category values along the x-axis. */
  xData: Array<string | number>;
  /** One or more line series. */
  series: LineSeries[];
  height?: number;
  xName?: string;
  yName?: string;
  yMin?: number;
  yMax?: number;
  showLegend?: boolean;
  /** See {@link EChart}. Defaults to false so live data animates in. */
  notMerge?: boolean;
}

/** Generic multi-series line chart. */
const LineChart = ({
  xData,
  series,
  height = 300,
  xName,
  yName,
  yMin,
  yMax,
  showLegend = true,
  notMerge,
}: LineChartProps) => {
  const option: EChartsOption = {
    grid: { left: 8, right: 16, top: showLegend ? 36 : 16, bottom: 8, containLabel: true },
    legend: showLegend ? { top: 0 } : undefined,
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: xData.map(String),
      name: xName,
      boundaryGap: false,
    },
    yAxis: { type: "value", name: yName, min: yMin, max: yMax },
    series: series.map((s) => ({
      name: s.name,
      type: "line",
      data: s.data,
      smooth: s.smooth ?? true,
      showSymbol: false,
      lineStyle: s.color ? { color: s.color, width: 2 } : { width: 2 },
      itemStyle: s.color ? { color: s.color } : undefined,
      areaStyle: s.area ? { opacity: 0.08 } : undefined,
    })),
  };

  return <EChart option={option} height={height} notMerge={notMerge} />;
};

export default LineChart;
