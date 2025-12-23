import type { Pointer, VizState } from '../lib/types';

const pointerPalette = ['#ef476f', '#118ab2', '#06d6a0', '#ffd166', '#8d99ae'];

function normalizePointers(pointers?: Pointer[]) {
  if (!pointers) {
    return [];
  }
  return pointers.map((pointer, index) => ({
    ...pointer,
    color: pointer.color ?? pointerPalette[index % pointerPalette.length],
  }));
}

function ArrayViz({ array, window, pointers }: Pick<VizState, 'array' | 'window' | 'pointers'>) {
  if (!array) {
    return null;
  }
  const normalized = normalizePointers(pointers);
  return (
    <div>
      <div className="array-row">
        {array.map((value, index) => {
          const isInWindow = window ? index >= window.l && index <= window.r : false;
          const pointerHits = normalized.filter((pointer) => pointer.index === index);
          return (
            <div
              key={`cell-${index}`}
              className={`array-cell${isInWindow ? ' window' : ''}${pointerHits.length ? ' pointer' : ''}`}
            >
              {value}
              {pointerHits.map((pointer) => (
                <span
                  key={`${pointer.name}-${index}`}
                  className="pointer-tag"
                  style={{ borderColor: pointer.color, color: pointer.color }}
                >
                  {pointer.name}
                </span>
              ))}
            </div>
          );
        })}
      </div>
      {normalized.length > 0 && (
        <div className="pointer-legend">
          {normalized.map((pointer) => (
            <span key={`legend-${pointer.name}`} style={{ color: pointer.color }}>
              {pointer.name}: {pointer.index}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function IntervalsViz({ intervals }: Pick<VizState, 'intervals'>) {
  if (!intervals || intervals.length === 0) {
    return null;
  }
  const min = Math.min(...intervals.map((interval) => interval.start));
  const max = Math.max(...intervals.map((interval) => interval.end));
  const span = max - min || 1;
  return (
    <div>
      {intervals.map((interval, index) => {
        const left = ((interval.start - min) / span) * 100;
        const width = ((interval.end - interval.start) / span) * 100;
        return (
          <div key={`${interval.start}-${interval.end}-${index}`} className="interval-track">
            <div className="interval-bar" style={{ left: `${left}%`, width: `${Math.max(width, 6)}%` }}>
              {interval.start}
              {' -> '}
              {interval.end}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function GraphViz({ graph, pointers }: Pick<VizState, 'graph' | 'pointers'>) {
  if (!graph) {
    return null;
  }
  const nodes = graph.nodes;
  const normalized = normalizePointers(pointers);
  const radius = 90;
  const center = 120;
  const positions = nodes.map((_, index) => {
    const angle = (index / nodes.length) * Math.PI * 2 - Math.PI / 2;
    return {
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
    };
  });

  return (
    <svg className="graph-svg" viewBox="0 0 240 240">
      {graph.edges.map(([from, to], index) => {
        const fromIndex = nodes.indexOf(from);
        const toIndex = nodes.indexOf(to);
        const start = positions[fromIndex];
        const end = positions[toIndex];
        return (
          <line
            key={`edge-${index}`}
            x1={start.x}
            y1={start.y}
            x2={end.x}
            y2={end.y}
            stroke="#bfae9a"
            strokeWidth={2}
          />
        );
      })}
      {nodes.map((node, index) => {
        const pos = positions[index];
        const highlight = normalized.find((pointer) => pointer.index === index);
        return (
          <g key={`node-${node}`}>
            <circle
              cx={pos.x}
              cy={pos.y}
              r={20}
              fill={highlight?.color ?? '#fff6ea'}
              stroke="#bfae9a"
              strokeWidth={2}
            />
            <text x={pos.x} y={pos.y + 4} textAnchor="middle" fontSize={12} fontWeight={700}>
              {node}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function HeapViz({ heapLeft, heapRight }: Pick<VizState, 'heapLeft' | 'heapRight'>) {
  if (!heapLeft && !heapRight) {
    return null;
  }
  return (
    <div className="heap-panel">
      <div className="heap-column">
        <strong>Left (max-heap)</strong>
        <div className="chips">
          {(heapLeft ?? []).map((value, index) => (
            <span key={`left-${index}`} className="chip">
              {value}
            </span>
          ))}
        </div>
      </div>
      <div className="heap-column">
        <strong>Right (min-heap)</strong>
        <div className="chips">
          {(heapRight ?? []).map((value, index) => (
            <span key={`right-${index}`} className="chip">
              {value}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function MatrixViz({ matrix }: Pick<VizState, 'matrix'>) {
  if (!matrix || matrix.length === 0) {
    return null;
  }
  const columns = matrix[0].length;
  return (
    <div className="matrix-grid" style={{ gridTemplateColumns: `repeat(${columns}, minmax(36px, 1fr))` }}>
      {matrix.flat().map((value, index) => (
        <div key={`cell-${index}`} className="matrix-cell">
          {value}
        </div>
      ))}
    </div>
  );
}

export default function VizCanvas({ state }: { state: VizState }) {
  return (
    <div className="viz-canvas">
      <ArrayViz array={state.array} window={state.window} pointers={state.pointers} />
      <IntervalsViz intervals={state.intervals} />
      <GraphViz graph={state.graph} pointers={state.pointers} />
      <HeapViz heapLeft={state.heapLeft} heapRight={state.heapRight} />
      <MatrixViz matrix={state.matrix} />
      {state.notes && <div className="panel">{state.notes}</div>}
    </div>
  );
}
