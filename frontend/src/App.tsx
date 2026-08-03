import { useEffect, useRef, useState } from 'react';
import Papa from 'papaparse';
import { ReactFlowProvider } from '@xyflow/react';
import { ModelCanvas } from './components/canvas/ModelCanvas';
import { AdvancedPage } from './components/advanced/AdvancedPage';
import { DataCenter } from './components/datagrid/DataCenter';
import { ResultsDashboard } from './components/dashboard/ResultsDashboard';
import { DiagnosticsDrawer } from './components/layout/DiagnosticsDrawer';
import { NavigationRail, type StudioView } from './components/layout/NavigationRail';
import { Titlebar } from './components/layout/Titlebar';
import { PropertyPanel } from './components/property/PropertyPanel';
import { ScenarioPage } from './components/scenarios/ScenarioPage';
import { SimulationPage } from './components/simulation/SimulationPage';
import { runDataChecker } from './engine/checkerRules/runDataChecker';
import { loadDemoProject, runSimulation, validateProject } from './services/api';
import { useCheckerStore } from './store/useCheckerStore';
import { useProjectStore } from './store/useProjectStore';
import { useResultStore } from './store/useResultStore';

function download(name: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const demoLoaded = useRef(false);
  const projectInput = useRef<HTMLInputElement>(null);
  const [view, setView] = useState<StudioView>('model');
  const [dark, setDark] = useState(
    () => localStorage.getItem('watermet-theme') === 'dark',
  );
  const project = useProjectStore((state) => state.project);
  const timeseries = useProjectStore((state) => state.timeseries);
  const nodes = useProjectStore((state) => state.nodes);
  const edges = useProjectStore((state) => state.edges);
  const loadProject = useProjectStore((state) => state.loadProject);
  const syncProject = useProjectStore((state) => state.syncProject);
  const markSaved = useProjectStore((state) => state.markSaved);
  const applySimulationFlows = useProjectStore((state) => state.applySimulationFlows);
  const setDiagnostics = useCheckerStore((state) => state.setDiagnostics);
  const setDrawerOpen = useCheckerStore((state) => state.setDrawerOpen);
  const setStatus = useResultStore((state) => state.setStatus);
  const setResult = useResultStore((state) => state.setResult);
  const setError = useResultStore((state) => state.setError);
  const addLog = useResultStore((state) => state.addLog);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    localStorage.setItem('watermet-theme', dark ? 'dark' : 'light');
  }, [dark]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDiagnostics(runDataChecker(project, nodes, edges, timeseries));
    }, 120);
    return () => window.clearTimeout(timer);
  }, [edges, nodes, project, setDiagnostics, timeseries]);

  useEffect(() => {
    if (demoLoaded.current) return;
    demoLoaded.current = true;
    void loadDemoProject()
      .then(({ project: demo, timeseries: rows }) => {
        loadProject(demo, rows);
        addLog('已从本地 API 载入完整示例项目。');
      })
      .catch(() => {
        addLog('API 未连接，当前使用浏览器内置示例；启动 API 后可载入完整项目。');
      });
  }, [addLog, loadProject]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (!event.ctrlKey) return;
      if (event.key.toLowerCase() === 'z') {
        event.preventDefault();
        useProjectStore.temporal.getState().undo();
      } else if (event.key.toLowerCase() === 'y') {
        event.preventDefault();
        useProjectStore.temporal.getState().redo();
      } else if (event.key.toLowerCase() === 's') {
        event.preventDefault();
        const current = syncProject();
        download('watermet2-project.json', JSON.stringify(current, null, 2), 'application/json');
        markSaved();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [markSaved, syncProject]);

  const save = () => {
    const current = syncProject();
    download('watermet2-project.json', JSON.stringify(current, null, 2), 'application/json');
    download('timeseries.csv', Papa.unparse(timeseries), 'text/csv;charset=utf-8');
    markSaved();
    addLog('项目 JSON 与时间序列 CSV 已下载。');
  };

  const openProject = (file: File) => {
    void file.text()
      .then((content) => {
        const candidate = JSON.parse(content) as unknown;
        if (
          typeof candidate !== 'object' ||
          candidate === null ||
          !('components' in candidate) ||
          !('local_areas' in candidate) ||
          !('simulation' in candidate)
        ) {
          throw new Error('JSON 不是有效的 WaterMet² 项目。');
        }
        loadProject(candidate as typeof project, timeseries);
        setView('model');
        addLog(`已打开项目：${file.name}。请在数据中心导入与之配套的 CSV。`);
      })
      .catch((error: unknown) => {
        addLog(error instanceof Error ? `打开失败：${error.message}` : '打开项目失败。');
        setDrawerOpen(true);
      });
  };

  const run = async () => {
    const diagnostics = runDataChecker(project, nodes, edges, timeseries);
    setDiagnostics(diagnostics);
    if (diagnostics.some((item) => item.severity === 'error')) {
      setDrawerOpen(true);
      addLog('运行被阻止：请先修复结构错误。');
      return;
    }
    setStatus('running');
    try {
      const current = syncProject();
      const response = await runSimulation({ project: current, timeseries });
      setResult(response);
      applySimulationFlows(response);
      setView('results');
    } catch (error) {
      setError(error instanceof Error ? error.message : '仿真失败');
      setDrawerOpen(true);
    }
  };

  const check = async () => {
    const diagnostics = runDataChecker(project, nodes, edges, timeseries);
    setDiagnostics(diagnostics);
    setDrawerOpen(true);
    if (diagnostics.some((item) => item.severity === 'error')) {
      addLog(`前端检查发现 ${diagnostics.filter((item) => item.severity === 'error').length} 个阻塞错误。`);
      return;
    }
    try {
      const current = syncProject();
      const response = await validateProject({ project: current, timeseries });
      addLog(`完整检查通过：${response.components} 个组件、${response.days} 天数据。`);
    } catch (error) {
      const message = error instanceof Error ? error.message : '后端校验失败';
      setDiagnostics([
        ...diagnostics,
        {
          id: 'backend-validation',
          severity: 'error',
          code: 'BACKEND_VALIDATION',
          title: 'Python 引擎校验失败',
          message,
        },
      ]);
      addLog(`后端校验失败：${message}`);
    }
  };

  const content = {
    model: (
      <div className="model-view">
        <ModelCanvas />
        <PropertyPanel />
      </div>
    ),
    data: <DataCenter />,
    simulation: <SimulationPage onShowResults={() => setView('results')} />,
    results: <ResultsDashboard />,
    scenarios: <ScenarioPage />,
    advanced: <AdvancedPage />,
  }[view];

  return (
    <ReactFlowProvider>
      <div className="studio-shell">
        <Titlebar
          dark={dark}
          onToggleTheme={() => setDark((value) => !value)}
          onRun={() => void run()}
          onOpen={() => projectInput.current?.click()}
          onSave={save}
          onCheck={() => void check()}
        />
        <input
          ref={projectInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) openProject(file);
            event.currentTarget.value = '';
          }}
        />
        <div className="studio-main">
          <NavigationRail view={view} onChange={setView} />
          <main className="content-area">{content}</main>
        </div>
        <DiagnosticsDrawer />
      </div>
    </ReactFlowProvider>
  );
}
