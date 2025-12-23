import type { Step } from '../lib/types';

type CodePanelProps = {
  code: string;
  step?: Step | null;
};

export default function CodePanel({ code, step }: CodePanelProps) {
  const lines = code.trimEnd().split('\n');
  const highlights = step?.codeLineHighlights ?? [];

  return (
    <div className="code-panel">
      {lines.map((line, index) => {
        const lineNumber = index + 1;
        const isHighlighted = highlights.includes(lineNumber);
        return (
          <div key={`${lineNumber}-${line}`} className={`code-line${isHighlighted ? ' highlight' : ''}`}>
            <span className="code-line-number">{lineNumber}</span>
            <span>{line || ' '}</span>
          </div>
        );
      })}
    </div>
  );
}
