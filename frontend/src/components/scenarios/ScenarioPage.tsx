import { useState } from 'react';
import { AddRegular, DeleteRegular } from '@fluentui/react-icons';
import { useResultStore } from '../../store/useResultStore';
import { useScenarioStore } from '../../store/useScenarioStore';
import { Button } from '../ui/Button';

const metrics = [
  ['reliability_fraction', '可靠率'],
  ['total_unmet_ml', '总缺水 ML'],
  ['mean_annual_ghg_net_kg_co2e', '年均净 GHG kgCO₂e'],
  ['present_total_cost_eur', '现值成本 EUR'],
] as const;

export function ScenarioPage() {
  const current = useResultStore((state) => state.result);
  const scenarios = useScenarioStore((state) => state.scenarios);
  const addScenario = useScenarioStore((state) => state.addScenario);
  const removeScenario = useScenarioStore((state) => state.removeScenario);
  const [name, setName] = useState(`方案 ${scenarios.length + 1}`);

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <span className="eyebrow">SCENARIOS</span>
          <h1>情景对比</h1>
          <p>保存当前仿真快照，比较干预方案的可靠性、环境影响和成本。</p>
        </div>
        <div className="scenario-add">
          <input value={name} onChange={(event) => setName(event.target.value)} />
          <Button
            variant="primary"
            icon={<AddRegular />}
            disabled={!current || !name.trim()}
            onClick={() => {
              if (current) {
                addScenario(name.trim(), current);
                setName(`方案 ${scenarios.length + 2}`);
              }
            }}
          >
            保存当前结果
          </Button>
        </div>
      </div>
      <section className="card scenario-table-card">
        {scenarios.length === 0 ? (
          <div className="empty-state large">
            <h2>尚未保存情景</h2>
            <p>修改模型参数、运行仿真，然后将结果保存为一个方案。</p>
          </div>
        ) : (
          <div className="scenario-table">
            <div className="scenario-row header">
              <strong>指标</strong>
              {scenarios.map((scenario) => <strong key={scenario.id}>{scenario.name}</strong>)}
            </div>
            {metrics.map(([key, label]) => (
              <div className="scenario-row" key={key}>
                <span>{label}</span>
                {scenarios.map((scenario) => {
                  const value = scenario.result.summary[key] ?? 0;
                  return <b key={scenario.id}>{value.toLocaleString('zh-CN', { maximumFractionDigits: 3 })}</b>;
                })}
              </div>
            ))}
            <div className="scenario-row actions">
              <span>管理</span>
              {scenarios.map((scenario) => (
                <Button key={scenario.id} variant="ghost" icon={<DeleteRegular />} onClick={() => removeScenario(scenario.id)}>
                  删除
                </Button>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
