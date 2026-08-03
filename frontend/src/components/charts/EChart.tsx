import ReactEChartsCore from 'echarts-for-react/lib/core';
import { BarChart, LineChart, SankeyChart } from 'echarts/charts';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  BarChart,
  LineChart,
  SankeyChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  CanvasRenderer,
]);

interface EChartProps {
  option: Record<string, unknown>;
  style?: React.CSSProperties;
}

export function EChart({ option, style }: EChartProps) {
  return <ReactEChartsCore echarts={echarts} option={option} style={style} />;
}
