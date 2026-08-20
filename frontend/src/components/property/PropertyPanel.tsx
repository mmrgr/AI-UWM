import { ArrowResetRegular } from '@fluentui/react-icons';
import { useCanvasStore } from '../../store/useCanvasStore';
import { useProjectStore } from '../../store/useProjectStore';
import type { ProjectObject } from '../../types/project';
import { Button } from '../ui/Button';

interface FieldSpec {
  key: string;
  label: string;
  unit?: string;
  required?: boolean;
  min?: number;
  max?: number;
  help: string;
}

const hydraulicFields: FieldSpec[] = [
  { key: 'capacity_ml', label: '储存容量', unit: 'ML', min: 0, help: '设施最大可用储量。' },
  { key: 'initial_ml', label: '初始储量', unit: 'ML', min: 0, help: '模拟起始日的储量。' },
  { key: 'daily_capacity_ml', label: '日处理能力', unit: 'ML/day', min: 0, help: '设施单日最大通量。' },
  { key: 'leakage_fraction', label: '漏损率', unit: '0–1', min: 0, max: 1, help: '输入流量中未到达下游的比例。' },
  { key: 'loss_fraction', label: '过程损失率', unit: '0–1', min: 0, max: 1, help: '处理或输送过程损失比例。' },
];

const impactFields: FieldSpec[] = [
  { key: 'electricity_kwh_m3', label: '单位电耗', unit: 'kWh/m³', min: 0, help: '每处理或输送一立方米水的用电量。' },
  { key: 'variable_cost_eur_m3', label: '单位运行成本', unit: 'EUR/m³', min: 0, help: '随活动量变化的运行成本。' },
  { key: 'fixed_cost_eur_year', label: '年度固定成本', unit: 'EUR/year', min: 0, help: '每年的固定运行维护成本。' },
];

const assetFields: FieldSpec[] = [
  { key: 'asset_age_years', label: '资产年龄', unit: 'year', min: 0, help: '设施或管道当前年龄。' },
  { key: 'capital_cost_eur', label: '资本成本', unit: 'EUR', min: 0, help: '建设或更换的初始资本成本。' },
  { key: 'lifetime_years', label: '设计寿命', unit: 'year', min: 0, help: '用于年化和更换周期计算。' },
  { key: 'maintenance_fraction_capital_year', label: '维护成本比例', unit: '1/year', min: 0, help: '每年维护费占资本成本的比例。' },
];

const aiFields: FieldSpec[] = [
  { key: 'installed_it_capacity_mw', label: 'Installed IT Capacity', unit: 'MW', required: true, min: 0, help: 'IT设备额定功率。' },
  { key: 'load_factor', label: '负荷率', unit: '0–1', required: true, min: 0, max: 1, help: '每日IT容量利用率。' },
  { key: 'base_pue', label: '基础 PUE', required: true, min: 1, help: '园区总电量与IT电量之比。' },
  { key: 'cooling_storage_ml', label: '冷却水储量', unit: 'ML', min: 0, help: '冷却系统可用调蓄容量。' },
  { key: 'offsite_electricity_water_intensity_l_kwh', label: '异地电力耗水', unit: 'L/kWh', min: 0, help: '仅计入水足迹，不进入城市物理水量平衡。' },
];

function readNumber(config: ProjectObject, key: string): string {
  const value = config[key];
  return typeof value === 'number' ? String(value) : '';
}

function NumberField({
  field,
  config,
  highlighted,
  onChange,
}: {
  field: FieldSpec;
  config: ProjectObject;
  highlighted: boolean;
  onChange: (key: string, value: number | undefined) => void;
}) {
  const value = readNumber(config, field.key);
  return (
    <label className={highlighted ? 'form-field highlighted' : 'form-field'} title={field.help}>
      <span>
        {field.label}{field.required && <b>*</b>}
        {field.unit && <small>{field.unit}</small>}
      </span>
      <input
        type="number"
        min={field.min}
        max={field.max}
        step="any"
        value={value}
        className={field.required && value === '' ? 'invalid' : ''}
        onChange={(event) =>
          onChange(
            field.key,
            event.target.value === '' ? undefined : Number(event.target.value),
          )
        }
      />
    </label>
  );
}

export function PropertyPanel() {
  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const selectedEdgeId = useCanvasStore((state) => state.selectedEdgeId);
  const focusField = useCanvasStore((state) => state.focusRequest?.field);
  const node = useProjectStore((state) =>
    state.nodes.find((candidate) => candidate.id === selectedNodeId),
  );
  const edge = useProjectStore((state) =>
    state.edges.find((candidate) => candidate.id === selectedEdgeId),
  );
  const updateConfig = useProjectStore((state) => state.updateNodeConfig);
  const updateLabel = useProjectStore((state) => state.updateNodeLabel);
  const updateAllocation = useProjectStore((state) => state.updateEdgeAllocation);

  if (edge?.data) {
    return (
      <aside className="property-panel">
        <div className="panel-heading">
          <strong>连线属性</strong>
          <span>{edge.source} → {edge.target}</span>
        </div>
        <div className="property-scroll">
          <details open>
            <summary>流量分配</summary>
            <label className="form-field">
              <span>分配比例 <small>0–1</small></span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={edge.data.allocation}
                onChange={(event) => updateAllocation(edge.id, Number(event.target.value))}
              />
            </label>
            <div className="read-only-row"><span>流类型</span><b>{edge.data.flowType}</b></div>
            <div className="read-only-row">
              <span>模拟平均流量</span>
              <b>
                {edge.data.flowMlDay === undefined
                  ? '运行后显示'
                  : `${edge.data.flowMlDay.toLocaleString('zh-CN', { maximumFractionDigits: 3 })} ML/day`}
              </b>
            </div>
          </details>
        </div>
      </aside>
    );
  }

  if (!node) {
    return (
      <aside className="property-panel">
        <div className="panel-heading"><strong>属性</strong></div>
        <div className="empty-state">选择节点或连线以编辑属性。</div>
      </aside>
    );
  }

  const setNumber = (key: string, value: number | undefined) => {
    const patch = { [key]: value };
    updateConfig(node.id, patch);
  };
  const sections = [
    ['水力与容量', hydraulicFields],
    ['能耗与成本', impactFields],
    ['资产与生命周期', assetFields],
  ] as const;

  return (
    <aside className="property-panel">
      <div className="panel-heading">
        <strong>属性编辑器</strong>
        <span>{node.id}</span>
      </div>
      <div className="property-scroll">
        <details open>
          <summary>基本信息</summary>
          <label className="form-field">
            <span>节点 ID</span>
            <input value={node.id} readOnly />
          </label>
          <label className="form-field">
            <span>显示名称 <b>*</b></span>
            <input
              value={node.data.label}
              className={!node.data.label ? 'invalid' : ''}
              onChange={(event) => {
                updateLabel(node.id, event.target.value);
                updateConfig(node.id, { name: event.target.value });
              }}
            />
          </label>
          <div className="read-only-row"><span>组件类型</span><b>{node.data.nodeType}</b></div>
        </details>
        {sections.map(([title, fields]) => (
          <details key={title} open={fields.some((field) => field.key === focusField)}>
            <summary>{title}</summary>
            {fields.map((field) => (
              <NumberField
                key={field.key}
                field={field}
                config={node.data.config}
                highlighted={field.key === focusField}
                onChange={setNumber}
              />
            ))}
          </details>
        ))}
        {node.data.nodeType === 'data_center' && (
          <details open>
            <summary>AI · Energy · Cooling · Water</summary>
            {aiFields.map((field) => (
              <NumberField key={field.key} field={field} config={node.data.config} highlighted={field.key === focusField} onChange={setNumber} />
            ))}
            <label className="form-field">
              <span>所属城市区域 <b>*</b></span>
              <input value={typeof node.data.config.local_area === 'string' ? node.data.config.local_area : ''} onChange={(event) => updateConfig(node.id, { local_area: event.target.value })} />
            </label>
            <label className="form-field">
              <span>PUE 模式</span>
              <select value={typeof node.data.config.pue_mode === 'string' ? node.data.config.pue_mode : 'dynamic'} onChange={(event) => updateConfig(node.id, { pue_mode: event.target.value })}>
                <option value="fixed">fixed_pue</option><option value="dynamic">dynamic_pue</option>
              </select>
            </label>
            <label className="form-field">
              <span>冷却技术</span>
              <select
                value={typeof (node.data.config.cooling as ProjectObject | undefined)?.technology === 'string' ? String((node.data.config.cooling as ProjectObject).technology) : 'evaporative'}
                onChange={(event) => updateConfig(node.id, { cooling: { ...((node.data.config.cooling as ProjectObject | undefined) ?? {}), technology: event.target.value } })}
              >
                {['evaporative', 'efficient_evaporative', 'hybrid', 'dry', 'liquid_to_air', 'liquid_to_water'].map((technology) => <option key={technology}>{technology}</option>)}
              </select>
            </label>
            <label className="form-field">
              <span>浓缩倍数 CoC</span>
              <input
                type="number" min="1.000001" step="0.1"
                value={typeof (node.data.config.cooling as ProjectObject | undefined)?.cycles_of_concentration === 'number' ? String((node.data.config.cooling as ProjectObject).cycles_of_concentration) : '5'}
                onChange={(event) => updateConfig(node.id, { cooling: { ...((node.data.config.cooling as ProjectObject | undefined) ?? {}), cycles_of_concentration: Number(event.target.value) } })}
              />
            </label>
            <label className="form-field">
              <span>再生水目标比例 <small>0–1</small></span>
              <input
                type="number" min="0" max="1" step="0.05"
                value={String((((node.data.config.water_sources as ProjectObject | undefined)?.reclaimed as ProjectObject | undefined)?.target_fraction as number | undefined) ?? 0.7)}
                onChange={(event) => {
                  const reclaimed = Number(event.target.value);
                  const sources = (node.data.config.water_sources as ProjectObject | undefined) ?? {};
                  updateConfig(node.id, { water_sources: { ...sources, reclaimed: { ...((sources.reclaimed as ProjectObject | undefined) ?? {}), target_fraction: reclaimed }, potable: { ...((sources.potable as ProjectObject | undefined) ?? {}), target_fraction: 1 - reclaimed } } });
                }}
              />
            </label>
          </details>
        )}
        <details open={focusField === 'base_population'}>
          <summary>时间序列与区域</summary>
          {node.data.nodeType === 'water_resource' && (
            <label className="form-field">
              <span>入流数据列</span>
              <input
                value={typeof node.data.config.inflow_column === 'string' ? node.data.config.inflow_column : ''}
                placeholder="source_inflow_ml"
                onChange={(event) => updateConfig(node.id, { inflow_column: event.target.value })}
              />
            </label>
          )}
          {node.data.nodeType === 'local_area' && (
            <>
              <NumberField
                field={{ key: 'base_population', label: '基准人口', required: true, min: 0, help: '未使用人口时间序列时的起始人口。' }}
                config={node.data.config}
                highlighted={focusField === 'base_population'}
                onChange={setNumber}
              />
              <NumberField
                field={{ key: 'area_ha', label: '区域面积', unit: 'ha', required: true, min: 0, help: '用于降雨径流计算的面积。' }}
                config={node.data.config}
                highlighted={focusField === 'area_ha'}
                onChange={setNumber}
              />
            </>
          )}
        </details>
        <Button
          variant="ghost"
          icon={<ArrowResetRegular />}
          onClick={() => {
            hydraulicFields.forEach((field) => setNumber(field.key, undefined));
          }}
        >
          恢复水力参数默认值
        </Button>
      </div>
    </aside>
  );
}
