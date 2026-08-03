import { CheckmarkCircleRegular, PlayRegular, PulseRegular } from '@fluentui/react-icons';
import { runSimulation, validateProject } from '../../services/api';
import { useCheckerStore } from '../../store/useCheckerStore';
import { useProjectStore } from '../../store/useProjectStore';
import { useResultStore } from '../../store/useResultStore';
import { Button } from '../ui/Button';

interface SimulationPageProps {
  onShowResults: () => void;
}

export function SimulationPage({ onShowResults }: SimulationPageProps) {
  const project = useProjectStore((state) => state.project);
  const timeseries = useProjectStore((state) => state.timeseries);
  const syncProject = useProjectStore((state) => state.syncProject);
  const setDates = useProjectStore((state) => state.setSimulationDates);
  const applySimulationFlows = useProjectStore((state) => state.applySimulationFlows);
  const diagnostics = useCheckerStore((state) => state.diagnostics);
  const status = useResultStore((state) => state.status);
  const result = useResultStore((state) => state.result);
  const error = useResultStore((state) => state.error);
  const setStatus = useResultStore((state) => state.setStatus);
  const setResult = useResultStore((state) => state.setResult);
  const setError = useResultStore((state) => state.setError);
  const addLog = useResultStore((state) => state.addLog);
  const blocking = diagnostics.filter((item) => item.severity === 'error').length;

  const validate = async () => {
    setStatus('validating');
    try {
      const current = syncProject();
      const response = await validateProject({ project: current, timeseries });
      addLog(`后端校验通过：${response.components} 个组件，${response.days} 天。`);
      setStatus('idle');
    } catch (error) {
      setError(error instanceof Error ? error.message : '校验失败');
    }
  };

  const run = async () => {
    setStatus('running');
    addLog('开始运行 WaterMet² 日尺度质量平衡……');
    try {
      const current = syncProject();
      const response = await runSimulation({ project: current, timeseries });
      setResult(response);
      applySimulationFlows(response);
      onShowResults();
    } catch (error) {
      setError(error instanceof Error ? error.message : '仿真失败');
    }
  };

  return (
    <div className="page simulation-page">
      <div className="page-header">
        <div>
          <span className="eyebrow">SIMULATION</span>
          <h1>仿真运行</h1>
          <p>前端即时检查用于定位问题，Python 引擎校验与质量守恒是最终执行门槛。</p>
        </div>
      </div>
      <div className="simulation-grid">
        <section className="card setup-card">
          <div className="section-heading"><h2>运行设置</h2><PulseRegular /></div>
          <div className="date-grid">
            <label className="form-field">
              <span>开始日期</span>
              <input type="date" value={project.simulation.start} onChange={(event) => setDates(event.target.value, project.simulation.end)} />
            </label>
            <label className="form-field">
              <span>结束日期</span>
              <input type="date" value={project.simulation.end} onChange={(event) => setDates(project.simulation.start, event.target.value)} />
            </label>
          </div>
          <div className="run-summary">
            <div><span>时间步长</span><strong>日</strong></div>
            <div><span>数据行</span><strong>{timeseries.length}</strong></div>
            <div><span>组件数</span><strong>{Object.keys(project.components).length}</strong></div>
            <div><span>阻塞错误</span><strong className={blocking > 0 ? 'danger-text' : ''}>{blocking}</strong></div>
          </div>
          <div className="run-actions">
            <Button variant="secondary" icon={<CheckmarkCircleRegular />} onClick={() => void validate()} disabled={status === 'running' || status === 'validating'}>
              {status === 'validating' ? '校验中…' : '后端校验'}
            </Button>
            <Button variant="primary" icon={<PlayRegular />} onClick={() => void run()} disabled={status === 'running' || blocking > 0}>
              {status === 'running' ? '正在计算…' : '开始仿真'}
            </Button>
          </div>
          {error && <p className="danger-text" role="alert">{error}</p>}
        </section>
        <section className="card readiness-card">
          <div className="readiness-ring">
            <strong>{blocking === 0 ? 'READY' : blocking}</strong>
            <span>{blocking === 0 ? '可以运行' : '个错误待修复'}</span>
          </div>
          <h2>{blocking === 0 ? '模型已通过前端结构检查' : '模型尚未满足运行条件'}</h2>
          <p>运行将计算水量、污染物、能源、环境影响、生命周期成本、资产失效、洪涝与 23 类风险。</p>
          {result && (
            <Button variant="ghost" onClick={onShowResults}>查看最近结果</Button>
          )}
        </section>
      </div>
    </div>
  );
}
