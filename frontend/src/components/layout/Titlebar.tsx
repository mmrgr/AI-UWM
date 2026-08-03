import {
  ArrowRedoRegular,
  ArrowUndoRegular,
  CheckmarkCircleRegular,
  DarkThemeRegular,
  FolderOpenRegular,
  PlayRegular,
  SaveRegular,
} from '@fluentui/react-icons';
import { useCheckerStore } from '../../store/useCheckerStore';
import { useProjectStore } from '../../store/useProjectStore';
import { useResultStore } from '../../store/useResultStore';
import { Button } from '../ui/Button';

interface TitlebarProps {
  dark: boolean;
  onToggleTheme: () => void;
  onRun: () => void;
  onOpen: () => void;
  onSave: () => void;
  onCheck: () => void;
}

export function Titlebar({
  dark,
  onToggleTheme,
  onRun,
  onOpen,
  onSave,
  onCheck,
}: TitlebarProps) {
  const projectName = useProjectStore((state) => state.project.name);
  const dirty = useProjectStore((state) => state.dirty);
  const status = useResultStore((state) => state.status);
  const diagnostics = useCheckerStore((state) => state.diagnostics);
  const errors = diagnostics.filter((item) => item.severity === 'error').length;
  const temporal = useProjectStore.temporal;

  return (
    <header className="titlebar">
      <div className="brand-mark" aria-hidden="true">
        W²
      </div>
      <div className="project-title">
        <strong>WaterMet² Studio</strong>
        <span>{projectName}{dirty ? ' *' : ''}</span>
      </div>
      <div className="titlebar-spacer" />
      <div className="command-group">
        <Button
          variant="ghost"
          icon={<ArrowUndoRegular />}
          title="撤销 (Ctrl+Z)"
          aria-label="撤销"
          onClick={() => temporal.getState().undo()}
        />
        <Button
          variant="ghost"
          icon={<ArrowRedoRegular />}
          title="重做 (Ctrl+Y)"
          aria-label="重做"
          onClick={() => temporal.getState().redo()}
        />
        <Button variant="ghost" icon={<SaveRegular />} onClick={onSave}>
          保存
        </Button>
        <Button variant="ghost" icon={<FolderOpenRegular />} onClick={onOpen}>
          打开
        </Button>
        <Button
          variant={errors > 0 ? 'danger' : 'secondary'}
          icon={<CheckmarkCircleRegular />}
          onClick={onCheck}
        >
          数据检查
          {diagnostics.length > 0 && <span className="badge">{diagnostics.length}</span>}
        </Button>
        <Button
          variant="primary"
          icon={<PlayRegular />}
          disabled={status === 'running'}
          onClick={onRun}
        >
          {status === 'running' ? '运行中…' : '运行仿真'}
        </Button>
        <Button
          variant="ghost"
          icon={<DarkThemeRegular />}
          title={dark ? '切换浅色主题' : '切换深色主题'}
          aria-label="切换主题"
          onClick={onToggleTheme}
        />
      </div>
    </header>
  );
}
