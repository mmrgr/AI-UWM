import {
  BeakerRegular,
  BranchForkRegular,
  ChartMultipleRegular,
  BranchCompareRegular,
  DatabaseRegular,
  PlayCircleRegular,
} from '@fluentui/react-icons';

export type StudioView =
  | 'model'
  | 'data'
  | 'simulation'
  | 'results'
  | 'scenarios'
  | 'advanced';

const items: Array<{
  id: StudioView;
  label: string;
  icon: typeof BranchForkRegular;
}> = [
  { id: 'model', label: '系统模型', icon: BranchForkRegular },
  { id: 'data', label: '数据中心', icon: DatabaseRegular },
  { id: 'simulation', label: '仿真运行', icon: PlayCircleRegular },
  { id: 'results', label: '结果分析', icon: ChartMultipleRegular },
  { id: 'scenarios', label: '情景对比', icon: BranchCompareRegular },
  { id: 'advanced', label: '高级分析', icon: BeakerRegular },
];

interface NavigationRailProps {
  view: StudioView;
  onChange: (view: StudioView) => void;
}

export function NavigationRail({ view, onChange }: NavigationRailProps) {
  return (
    <nav className="navigation-rail" aria-label="主导航">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.id}
            className={view === item.id ? 'rail-item active' : 'rail-item'}
            title={item.label}
            aria-label={item.label}
            aria-current={view === item.id ? 'page' : undefined}
            onClick={() => onChange(item.id)}
          >
            <Icon />
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
