import { ChevronDownRegular, ChevronUpRegular, LocationRegular } from '@fluentui/react-icons';
import { useCanvasStore } from '../../store/useCanvasStore';
import { useCheckerStore } from '../../store/useCheckerStore';
import { useProjectStore } from '../../store/useProjectStore';
import { useResultStore } from '../../store/useResultStore';

export function DiagnosticsDrawer() {
  const diagnostics = useCheckerStore((state) => state.diagnostics);
  const open = useCheckerStore((state) => state.drawerOpen);
  const activeTab = useCheckerStore((state) => state.activeTab);
  const setOpen = useCheckerStore((state) => state.setDrawerOpen);
  const setActiveTab = useCheckerStore((state) => state.setActiveTab);
  const focusNode = useCanvasStore((state) => state.focusNode);
  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const zoom = useCanvasStore((state) => state.zoom);
  const nodes = useProjectStore((state) => state.nodes);
  const edges = useProjectStore((state) => state.edges);
  const logs = useResultStore((state) => state.logs);
  const normalize = useProjectStore((state) => state.normalizeOutgoing);

  const visible =
    activeTab === 'logs'
      ? []
      : diagnostics.filter((item) => item.severity === activeTab);
  const counts = {
    error: diagnostics.filter((item) => item.severity === 'error').length,
    warning: diagnostics.filter((item) => item.severity === 'warning').length,
    missing: diagnostics.filter((item) => item.severity === 'missing').length,
  };

  return (
    <section className={open ? 'diagnostics-drawer open' : 'diagnostics-drawer'}>
      <div className="drawer-tabs">
        {([
          ['error', '错误', counts.error],
          ['warning', '警告', counts.warning],
          ['missing', '缺失数据', counts.missing],
          ['logs', '运行日志', logs.length],
        ] as const).map(([id, label, count]) => (
          <button
            key={id}
            className={activeTab === id ? 'drawer-tab active' : 'drawer-tab'}
            onClick={() => {
              setActiveTab(id);
              setOpen(true);
            }}
          >
            {label} <span>{count}</span>
          </button>
        ))}
        <div className="drawer-status">
          <span>选中：{selectedNodeId ?? '无'}</span>
          <span>缩放：{Math.round(zoom * 100)}%</span>
          <span>{nodes.length} 节点 / {edges.length} 连线</span>
        </div>
        <button
          className="drawer-toggle"
          aria-label={open ? '收起问题面板' : '展开问题面板'}
          onClick={() => setOpen(!open)}
        >
          {open ? <ChevronDownRegular /> : <ChevronUpRegular />}
        </button>
      </div>
      {open && (
        <div className="drawer-body">
          {activeTab === 'logs' ? (
            <div className="log-list">
              {logs.map((message, index) => (
                <div key={`${index}-${message}`}>{message}</div>
              ))}
            </div>
          ) : visible.length === 0 ? (
            <div className="empty-state compact">此分类没有问题。</div>
          ) : (
            visible.map((item) => (
              <div
                key={item.id}
                className={`diagnostic-row severity-${item.severity}`}
                role="button"
                tabIndex={0}
                onClick={() => item.nodeId && focusNode(item.nodeId, item.field)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && item.nodeId) focusNode(item.nodeId, item.field);
                }}
              >
                <LocationRegular />
                <div>
                  <strong>{item.title}</strong>
                  <p>{item.message}</p>
                </div>
                {item.fix === 'normalize-allocations' && item.nodeId && (
                  <button
                    className="inline-action"
                    onClick={(event) => {
                      event.stopPropagation();
                      normalize(item.nodeId ?? '');
                    }}
                  >
                    一键归一化
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </section>
  );
}
