export type DiagnosticSeverity = 'error' | 'warning' | 'missing';

export interface Diagnostic {
  id: string;
  severity: DiagnosticSeverity;
  code: string;
  title: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
  field?: string;
  fix?: 'normalize-allocations';
}
