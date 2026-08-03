import { useMemo, useState } from 'react';
import {
  ArrowTrendingRegular,
  CloudRegular,
  CurrencyDollarEuroRegular,
  DropRegular,
  ArrowDownloadRegular,
  FlashRegular,
  ShieldCheckmarkRegular,
} from '@fluentui/react-icons';
import Papa from 'papaparse';
import { exportSimulation } from '../../services/api';
import { useProjectStore } from '../../store/useProjectStore';
import { useResultStore } from '../../store/useResultStore';
import type { ResultFrequency, ResultRow, ResultTableName } from '../../types/results';
import { EChart } from '../charts/EChart';
import { Button } from '../ui/Button';

function number(row: ResultRow, key: string): number {
  const value = row[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function aggregateRows(rows: ResultRow[], frequency: ResultFrequency): ResultRow[] {
  if (frequency === 'daily') return rows;
  const groups = new Map<string, ResultRow[]>();
  for (const row of rows) {
    const date = String(row.date ?? '');
    const key = frequency === 'monthly' ? date.slice(0, 7) : date.slice(0, 4);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.entries()].map(([date, values]) => {
    const numericKeys = Object.keys(values[0] ?? {}).filter((key) =>
      values.some((row) => typeof row[key] === 'number'),
    );
    const aggregated: ResultRow = { date };
    numericKeys.forEach((key) => {
      aggregated[key] = values.reduce((sum, row) => sum + number(row, key), 0);
    });
    return aggregated;
  });
}

function formatMetric(value: number, unit: string): string {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} M ${unit}`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)} k ${unit}`;
  return `${value.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} ${unit}`;
}

export function ResultsDashboard() {
  const [selectedTable, setSelectedTable] = useState<ResultTableName>('system_daily');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const result = useResultStore((state) => state.result);
  const frequency = useResultStore((state) => state.frequency);
  const setFrequency = useResultStore((state) => state.setFrequency);
  const edges = useProjectStore((state) => state.edges);
  const nodes = useProjectStore((state) => state.nodes);
  const timeseries = useProjectStore((state) => state.timeseries);
  const syncProject = useProjectStore((state) => state.syncProject);
  const systemRows = useMemo(
    () => aggregateRows(result?.tables.system_daily ?? [], frequency),
    [frequency, result],
  );

  if (!result) {
    return (
      <div className="page">
        <div className="empty-state large">
          <DropRegular />
          <h2>尚无仿真结果</h2>
          <p>先在“仿真运行”中通过检查并执行模型。</p>
        </div>
      </div>
    );
  }

  const summary = result.summary;
  const demand = result.tables.system_daily.reduce((sum, row) => sum + number(row, 'water_demand_ml'), 0);
  const delivered = result.tables.system_daily.reduce((sum, row) => sum + number(row, 'delivered_total_ml'), 0);
  const reliability = demand > 0 ? (100 * delivered) / demand : 100;
  const energy = result.tables.system_daily.reduce((sum, row) => sum + number(row, 'total_energy_kwh'), 0);
  const ghg = result.tables.system_daily.reduce((sum, row) => sum + number(row, 'ghg_net_kg_co2e'), 0);
  const cost = number(summary, 'present_total_cost_eur');

  const componentTotals = new Map<string, number>();
  result.tables.component_daily.forEach((row) => {
    const id = String(row.component_id ?? '');
    componentTotals.set(id, (componentTotals.get(id) ?? 0) + number(row, 'outflow_ml'));
  });
  const nodeKinds = new Map(nodes.map((node) => [node.id, node.data.nodeType]));
  const sankeyEdges = edges.filter(
    (edge) => nodeKinds.get(edge.source) !== 'reuse' && nodeKinds.get(edge.target) !== 'reuse',
  );
  const sankeyNodeIds = new Set(sankeyEdges.flatMap((edge) => [edge.source, edge.target]));
  const sankeyNodes = nodes
    .filter((node) => sankeyNodeIds.has(node.id))
    .map((node) => ({ name: node.id, itemStyle: { color: node.data.nodeType === 'wwtw' ? '#8b5e3c' : '#0f6cbd' } }));
  const sankeyLinks = sankeyEdges.map((edge) => ({
    source: edge.source,
    target: edge.target,
    value: Math.max(0.01, (componentTotals.get(edge.source) ?? 1) * (edge.data?.allocation ?? 1)),
  }));
  const sankeyOption = {
    animationDuration: 600,
    tooltip: { trigger: 'item' },
    series: [{
      type: 'sankey',
      data: sankeyNodes,
      links: sankeyLinks,
      emphasis: { focus: 'adjacency' },
      lineStyle: { color: 'gradient', curveness: 0.5, opacity: 0.42 },
      label: { color: 'var(--text)' },
      nodeWidth: 16,
      nodeGap: 12,
    }],
  };
  const timeOption = {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { data: ['需水量', '供水量', '缺水量', '储量'] },
    grid: { left: 55, right: 55, top: 48, bottom: 48 },
    xAxis: { type: 'category', data: systemRows.map((row) => row.date), axisLabel: { hideOverlap: true } },
    yAxis: [{ type: 'value', name: '流量 (ML)' }, { type: 'value', name: '储量 (ML)' }],
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 8 }],
    series: [
      { name: '需水量', type: 'line', showSymbol: false, data: systemRows.map((row) => number(row, 'water_demand_ml')), lineStyle: { color: '#d97706' } },
      { name: '供水量', type: 'line', showSymbol: false, data: systemRows.map((row) => number(row, 'delivered_total_ml')), areaStyle: { opacity: 0.12 }, lineStyle: { color: '#0f6cbd' } },
      { name: '缺水量', type: 'bar', stack: 'shortfall', data: systemRows.map((row) => number(row, 'unmet_demand_ml')), itemStyle: { color: '#d13438' } },
      { name: '储量', type: 'line', yAxisIndex: 1, showSymbol: false, data: systemRows.map((row) => number(row, 'storage_ml')), lineStyle: { color: '#2f9e44', type: 'dashed' } },
    ],
  };

  const cards = [
    { label: '供水可靠率', value: `${reliability.toFixed(3)}%`, icon: ShieldCheckmarkRegular, tone: 'blue' },
    { label: '总缺水量', value: formatMetric(number(summary, 'total_unmet_ml'), 'ML'), icon: DropRegular, tone: 'red' },
    { label: '总能耗', value: formatMetric(energy, 'kWh'), icon: FlashRegular, tone: 'amber' },
    { label: '净温室气体', value: formatMetric(ghg / 1000, 'tCO₂e'), icon: CloudRegular, tone: 'green' },
    { label: '现值总成本', value: formatMetric(cost, 'EUR'), icon: CurrencyDollarEuroRegular, tone: 'purple' },
    { label: '风险代码', value: `${result.tables.risk_summary.length} 类`, icon: ArrowTrendingRegular, tone: 'cyan' },
  ];
  const tableRows = result.tables[selectedTable];
  const tableColumns = Object.keys(tableRows[0] ?? {});
  const downloadTable = () => {
    const blob = new Blob([`\uFEFF${Papa.unparse(tableRows)}`], {
      type: 'text/csv;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${selectedTable}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };
  const downloadAll = async () => {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await exportSimulation({
        project: syncProject(),
        timeseries,
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'watermet2-results.zip';
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : '导出失败');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="page results-page">
      <div className="page-header">
        <div>
          <span className="eyebrow">RESULTS</span>
          <h1>综合结果仪表盘</h1>
          <p>从系统代谢流到风险与成本的多尺度结果视图。</p>
        </div>
        <div className="segmented">
          {(['daily', 'monthly', 'annual'] as const).map((item) => (
            <button key={item} className={frequency === item ? 'active' : ''} onClick={() => setFrequency(item)}>
              {{ daily: '日', monthly: '月', annual: '年' }[item]}
            </button>
          ))}
        </div>
        <Button
          variant="primary"
          icon={<ArrowDownloadRegular />}
          disabled={exporting}
          onClick={() => void downloadAll()}
        >
          {exporting ? '正在打包…' : '导出全部结果 ZIP'}
        </Button>
      </div>
      <div className="kpi-grid">
        {cards.map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.label} className={`kpi-card tone-${card.tone}`}>
              <Icon />
              <span>{card.label}</span>
              <strong>{card.value}</strong>
            </div>
          );
        })}
      </div>
      <div className="dashboard-grid">
        <section className="card sankey-card">
          <div className="section-heading"><div><h2>城市水代谢桑基图</h2><p>线宽表示模拟期累计通量；回用闭环在系统模型中单独展示。</p></div></div>
          <EChart option={sankeyOption} style={{ height: 390 }} />
        </section>
        <section className="card risk-card">
          <div className="section-heading"><div><h2>风险概览</h2><p>按累计风险分数排序。</p></div></div>
          <div className="risk-list">
            {[...result.tables.risk_summary]
              .sort((a, b) => number(b, 'cumulative_risk_score') - number(a, 'cumulative_risk_score'))
              .slice(0, 8)
              .map((row) => (
                <div key={String(row.risk_code)}>
                  <b>{String(row.risk_code)}</b>
                  <span>{String(row.description)}</span>
                  <strong>{number(row, 'probability').toLocaleString('zh-CN', { style: 'percent', maximumFractionDigits: 1 })}</strong>
                </div>
              ))}
          </div>
        </section>
        <section className="card time-card">
          <div className="section-heading"><div><h2>供需平衡与系统储量</h2><p>支持缩放和时间尺度切换。</p></div></div>
          <EChart option={timeOption} style={{ height: 410 }} />
        </section>
        <section className="card result-table-card">
          <div className="section-heading">
            <div>
              <h2>完整结果表</h2>
              <p>模型全部日尺度输出均可预览和导出；ZIP 还包含月/年聚合结果。</p>
            </div>
            <div className="table-actions">
              <select
                aria-label="结果表"
                value={selectedTable}
                onChange={(event) => setSelectedTable(event.target.value as ResultTableName)}
              >
                {(Object.keys(result.tables) as ResultTableName[]).map((name) => (
                  <option key={name} value={name}>
                    {name} ({result.tables[name].length.toLocaleString()} 行)
                  </option>
                ))}
              </select>
              <Button variant="secondary" icon={<ArrowDownloadRegular />} onClick={downloadTable}>
                导出当前表 CSV
              </Button>
            </div>
          </div>
          {exportError && <p className="danger-text" role="alert">{exportError}</p>}
          {tableRows.length === 0 ? (
            <div className="empty-state compact">此项目没有生成该类记录。</div>
          ) : (
            <div className="result-table-scroll">
              <table>
                <thead>
                  <tr>{tableColumns.map((column) => <th key={column}>{column}</th>)}</tr>
                </thead>
                <tbody>
                  {tableRows.slice(0, 50).map((row, rowIndex) => (
                    <tr key={`${selectedTable}-${rowIndex}`}>
                      {tableColumns.map((column) => (
                        <td key={column}>{String(row[column] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {tableRows.length > 50 && (
                <p className="table-note">页面显示前 50 行，导出的 CSV 包含全部 {tableRows.length.toLocaleString()} 行。</p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
