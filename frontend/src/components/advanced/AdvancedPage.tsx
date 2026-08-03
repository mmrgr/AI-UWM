import { useMemo, useState } from 'react';
import { BeakerRegular, BranchCompareRegular, ChartMultipleRegular, PlayRegular } from '@fluentui/react-icons';
import {
  runCalibration,
  runDecisionSupport,
  runOptimization,
  runUncertainty,
} from '../../services/api';
import { useProjectStore } from '../../store/useProjectStore';
import type { ResultRow } from '../../types/results';
import { Button } from '../ui/Button';

type AnalysisMode = 'dss' | 'calibration' | 'uncertainty' | 'optimization';

const defaults: Record<AnalysisMode, unknown> = {
  dss: {
    scenarios: [{ name: 'reference' }],
    strategies: [
      { name: 'BAU' },
      {
        name: 'Leakage reduction',
        set: [{ path: 'components.DM1.leakage_fraction', value: 0.1 }],
      },
    ],
    metrics: {
      reliability_fraction: { goal: 'max' },
      mean_annual_ghg_net_kg_co2e: { goal: 'min' },
      present_total_cost_eur: { goal: 'min' },
    },
    preference_groups: {
      balanced_cp: {
        method: 'cp',
        weights: {
          reliability_fraction: 0.4,
          mean_annual_ghg_net_kg_co2e: 0.3,
          present_total_cost_eur: 0.3,
        },
      },
      reliability_ahp: {
        method: 'ahp',
        pairwise: [[1, 2, 2], [0.5, 1, 1], [0.5, 1, 1]],
      },
    },
  },
  calibration: {
    specification: {
      observed_column: 'observed',
      parameters: { 'components.DM1.leakage_fraction': [0.05, 0.1, 0.15] },
      result_table: 'system_daily',
      result_column: 'delivered_total_ml',
      objective: 'rmse',
    },
    observed: [
      { date: '2020-01-01', observed: 0 },
      { date: '2020-01-02', observed: 0 },
    ],
  },
  uncertainty: {
    samples: 20,
    seed: 42,
    parameters: [
      {
        path: 'components.DM1.leakage_fraction',
        distribution: 'uniform',
        low: 0.05,
        high: 0.15,
      },
    ],
  },
  optimization: {
    decisions: { 'components.DM1.leakage_fraction': [0.05, 0.1, 0.15] },
    objectives: {
      present_total_cost_eur: 'min',
      reliability_fraction: 'max',
    },
  },
};

const modeInfo: Record<AnalysisMode, { title: string; description: string }> = {
  dss: { title: '完整 DSS', description: '场景 × 策略 × 偏好组，支持 CP 与 AHP。' },
  calibration: { title: '校准与验证', description: '参数网格与 NSE、RSR、PBIAS、RMSE。' },
  uncertainty: { title: 'Monte Carlo', description: '概率分布抽样与 P05/P50/P95。' },
  optimization: { title: 'Pareto 优化', description: '离散决策组合与多目标非支配解。' },
};

function display(value: ResultRow[string] | undefined): string {
  return typeof value === 'number'
    ? value.toLocaleString('zh-CN', { maximumFractionDigits: 4 })
    : String(value ?? '');
}

export function AdvancedPage() {
  const timeseries = useProjectStore((state) => state.timeseries);
  const syncProject = useProjectStore((state) => state.syncProject);
  const [mode, setMode] = useState<AnalysisMode>('dss');
  const [editors, setEditors] = useState<Record<AnalysisMode, string>>(() =>
    Object.fromEntries(
      Object.entries(defaults).map(([key, value]) => [key, JSON.stringify(value, null, 2)]),
    ) as Record<AnalysisMode, string>,
  );
  const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<ResultRow[]>([]);
  const columns = useMemo(() => Object.keys(rows[0] ?? {}).slice(0, 10), [rows]);

  const run = async () => {
    setStatus('running');
    setError(null);
    try {
      const parsed = JSON.parse(editors[mode]) as Record<string, unknown>;
      const base = { project: syncProject(), timeseries };
      if (mode === 'dss') {
        const response = await runDecisionSupport({ ...base, specification: parsed });
        setRows(response.rankings);
      } else if (mode === 'calibration') {
        const specification = parsed.specification as Record<string, unknown>;
        const observed = parsed.observed as Record<string, unknown>[];
        if (!specification || !Array.isArray(observed)) {
          throw new Error('校准 JSON 必须包含 specification 和 observed 数组。');
        }
        const response = await runCalibration({ ...base, specification, observed });
        setRows(response.trials);
      } else if (mode === 'uncertainty') {
        const response = await runUncertainty({ ...base, specification: parsed });
        setRows(response.percentiles);
      } else {
        const response = await runOptimization({ ...base, specification: parsed });
        setRows(response.trials);
      }
      setStatus('success');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '高级分析失败');
      setRows([]);
      setStatus('error');
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">ADVANCED ANALYSIS</span>
          <h1>高级分析与决策支持</h1>
          <p>四类分析均直接调用 WaterMet² 计算引擎，并返回可检查的逐试验结果。</p>
        </div>
      </div>
      <div className="feature-grid advanced-mode-grid">
        {(Object.keys(modeInfo) as AnalysisMode[]).map((item) => {
          const Icon = item === 'calibration' ? BeakerRegular : item === 'uncertainty' ? ChartMultipleRegular : BranchCompareRegular;
          return (
            <button
              className={`feature-card advanced-mode-card${mode === item ? ' active' : ''}`}
              key={item}
              onClick={() => { setMode(item); setRows([]); setError(null); setStatus('idle'); }}
              type="button"
            >
              <Icon /><h2>{modeInfo[item].title}</h2><p>{modeInfo[item].description}</p>
            </button>
          );
        })}
      </div>
      <section className="card dss-workbench">
        <div className="section-heading">
          <div><h2>{modeInfo[mode].title}规范</h2><p>编辑 JSON 后运行；较大的网格和样本数会增加耗时。</p></div>
          <Button variant="primary" icon={<PlayRegular />} onClick={() => void run()} disabled={status === 'running'}>
            {status === 'running' ? '计算中…' : '运行分析'}
          </Button>
        </div>
        <textarea
          aria-label={`${mode} JSON specification`}
          className="dss-editor"
          value={editors[mode]}
          onChange={(event) => setEditors({ ...editors, [mode]: event.target.value })}
          spellCheck={false}
        />
        {error && <p className="danger-text" role="alert">{error}</p>}
      </section>
      <section className="card dss-results">
        <div className="section-heading"><h2>分析结果</h2><span>{rows.length ? `${rows.length} 条` : '尚未运行'}</span></div>
        {!rows.length ? (
          <div className="empty-state compact">运行后显示排序、校准试验、分位数或 Pareto 解。</div>
        ) : (
          <div className="result-table-scroll">
            <table>
              <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
              <tbody>
                {rows.slice(0, 100).map((row, index) => (
                  <tr key={index}>{columns.map((column) => <td key={column}>{display(row[column])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
