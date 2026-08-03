import { useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  ArrowUploadRegular,
  DataLineRegular,
  ArrowDownloadRegular,
  DocumentTableRegular,
} from '@fluentui/react-icons';
import Papa from 'papaparse';
import { useProjectStore } from '../../store/useProjectStore';
import type { ColumnMapping, Scalar, TimeSeriesRow } from '../../types/project';
import { EChart } from '../charts/EChart';
import { Button } from '../ui/Button';

const targets: ColumnMapping['target'][] = [
  'keep',
  'date',
  'rainfall_mm',
  'temperature_c',
  'population',
  'resource_inflow',
];

const canonicalColumns: Record<Exclude<ColumnMapping['target'], 'keep'>, string> = {
  date: 'date',
  rainfall_mm: 'rainfall_mm',
  temperature_c: 'temperature_c',
  population: 'population',
  resource_inflow: 'source_inflow_ml',
};

function scalar(value: unknown): Scalar {
  if (value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return value;
  }
  return String(value);
}

export function DataCenter() {
  const timeseries = useProjectStore((state) => state.timeseries);
  const setTimeseries = useProjectStore((state) => state.setTimeseries);
  const [mappings, setMappings] = useState<ColumnMapping[]>([]);
  const [fileName, setFileName] = useState<string>('内置时间序列');
  const parentRef = useRef<HTMLDivElement>(null);
  const columns = useMemo(
    () => Object.keys(timeseries[0] ?? {}),
    [timeseries],
  );
  const virtualizer = useVirtualizer({
    count: timeseries.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 34,
    overscan: 12,
  });
  const numericColumns = columns.filter((column) =>
    timeseries.some((row) => typeof row[column] === 'number'),
  );
  const previewColumn = numericColumns[0];
  const missing = timeseries.reduce(
    (total, row) =>
      total + columns.filter((column) => row[column] === null || row[column] === '').length,
    0,
  );

  const parseFile = (file: File) => {
    Papa.parse<Record<string, unknown>>(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (result.errors.length > 0 || result.data.length === 0) return;
        const rows = result.data.map((raw) => {
          const row: TimeSeriesRow = { date: String(raw.date ?? raw.Date ?? '') };
          Object.entries(raw).forEach(([key, value]) => {
            row[key] = scalar(value);
          });
          return row;
        });
        setTimeseries(rows);
        setFileName(file.name);
        setMappings(
          Object.keys(rows[0] ?? {}).map((source) => ({
            source,
            target:
              targets.find((target) =>
                source.toLowerCase().includes(target.split('_').at(0) ?? ''),
              ) ??
              (source.toLowerCase() === 'date' ? 'date' : 'keep'),
          })),
        );
      },
    });
  };

  const exportCsv = () => {
    const blob = new Blob([`\uFEFF${Papa.unparse(timeseries)}`], {
      type: 'text/csv;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'watermet2-timeseries.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const applyMappings = () => {
    const renamed = timeseries.map((row) => {
      const next: TimeSeriesRow = { ...row, date: String(row.date ?? '') };
      mappings.forEach(({ source, target }) => {
        if (target !== 'keep') next[canonicalColumns[target]] = row[source] ?? null;
      });
      next.date = String(next.date ?? '');
      return next;
    });
    setTimeseries(renamed);
    setMappings([]);
  };

  const chartOption = previewColumn
    ? {
        animation: false,
        grid: { left: 48, right: 18, top: 30, bottom: 45 },
        tooltip: { trigger: 'axis' },
        xAxis: {
          type: 'category',
          data: timeseries.slice(0, 365).map((row) => row.date),
          axisLabel: { hideOverlap: true },
        },
        yAxis: { type: 'value', name: previewColumn },
        series: [
          {
            type: 'line',
            showSymbol: false,
            smooth: true,
            data: timeseries.slice(0, 365).map((row) => row[previewColumn]),
            areaStyle: { opacity: 0.12 },
            lineStyle: { color: '#0f6cbd', width: 2 },
          },
        ],
      }
    : {};

  return (
    <div className="page data-page">
      <div className="page-header">
        <div>
          <span className="eyebrow">DATA CENTER</span>
          <h1>时间序列数据中心</h1>
          <p>导入、映射并预览驱动 WaterMet² 的日尺度数据。</p>
        </div>
        <div className="header-actions">
          <Button variant="secondary" icon={<ArrowDownloadRegular />} onClick={exportCsv}>
            导出当前 CSV
          </Button>
          <label className="file-button button button-primary">
            <ArrowUploadRegular />
            输入时间序列 CSV
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) parseFile(file);
              }}
            />
          </label>
        </div>
      </div>
      <div className="stats-grid">
        <div className="stat-card"><DocumentTableRegular /><span>文件</span><strong>{fileName}</strong></div>
        <div className="stat-card"><DataLineRegular /><span>记录数</span><strong>{timeseries.length.toLocaleString()}</strong></div>
        <div className="stat-card"><span>字段数</span><strong>{columns.length}</strong></div>
        <div className="stat-card"><span>缺失值</span><strong>{missing}</strong></div>
      </div>
      {mappings.length > 0 && (
        <section className="card mapping-card">
          <div className="section-heading">
            <div><h2>列映射</h2><p>确认外部字段与模型基础变量的对应关系。</p></div>
            <Button variant="ghost" onClick={applyMappings}>应用映射</Button>
          </div>
          <div className="mapping-grid">
            {mappings.map((mapping, index) => (
              <label key={`${mapping.source}-${index}`}>
                <span>{mapping.source}</span>
                <select
                  value={mapping.target}
                  onChange={(event) =>
                    setMappings((current) =>
                      current.map((item, itemIndex) =>
                        itemIndex === index
                          ? { ...item, target: event.target.value as ColumnMapping['target'] }
                          : item,
                      ),
                    )
                  }
                >
                  {targets.map((target) => (
                    <option key={target} value={target}>
                      {target === 'keep' ? '保留原列名' : target}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
        </section>
      )}
      <div className="data-grid-layout">
        <section className="card">
          <div className="section-heading"><h2>数据预览</h2><span>虚拟滚动</span></div>
          <div className="virtual-table" ref={parentRef}>
            <div className="table-header" style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(130px, 1fr))` }}>
              {columns.map((column) => <strong key={column}>{column}</strong>)}
            </div>
            <div style={{ height: virtualizer.getTotalSize(), position: 'relative', minWidth: columns.length * 130 }}>
              {virtualizer.getVirtualItems().map((virtualRow) => {
                const row = timeseries[virtualRow.index];
                return (
                  <div
                    key={virtualRow.key}
                    className="table-row"
                    style={{
                      gridTemplateColumns: `repeat(${columns.length}, minmax(130px, 1fr))`,
                      transform: `translateY(${virtualRow.start}px)`,
                    }}
                  >
                    {columns.map((column) => <span key={column}>{String(row?.[column] ?? '—')}</span>)}
                  </div>
                );
              })}
            </div>
          </div>
        </section>
        <section className="card chart-card">
          <div className="section-heading"><h2>快速预览</h2><span>{previewColumn ?? '无数值列'}</span></div>
          {previewColumn ? <EChart option={chartOption} style={{ height: 330 }} /> : <div className="empty-state">没有可绘制的数值列。</div>}
        </section>
      </div>
    </div>
  );
}
