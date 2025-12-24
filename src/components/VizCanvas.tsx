import type { Pointer, TreeNode, VizState } from '../lib/types';

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

function LinkedListViz({ linkedList, pointers }: Pick<VizState, 'linkedList' | 'pointers'>) {
  if (!linkedList || linkedList.length === 0) {
    return null;
  }
  const normalized = normalizePointers(pointers);
  return (
    <div className="list-row">
      {linkedList.map((value, index) => {
        const pointerHits = normalized.filter((pointer) => pointer.index === index);
        return (
          <div className="list-group" key={`list-${index}`}>
            <div className={`list-node${pointerHits.length ? ' pointer' : ''}`}>
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
            {index < linkedList.length - 1 && (
              <span className="list-arrow">{'->'}</span>
            )}
          </div>
        );
      })}
      <span className="list-null">null</span>
    </div>
  );
}

function StackViz({ stack, pointers }: Pick<VizState, 'stack' | 'pointers'>) {
  if (!stack || stack.length === 0) {
    return null;
  }
  const normalized = normalizePointers(pointers);
  return (
    <div className="stack-viz">
      <span className="stack-label">top</span>
      <div className="stack-column">
        {[...stack].reverse().map((value, revIndex) => {
          const index = stack.length - 1 - revIndex;
          const pointerHits = normalized.filter((pointer) => pointer.index === index);
          return (
            <div key={`stack-${index}`} className={`stack-cell${pointerHits.length ? ' pointer' : ''}`}>
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
    </div>
  );
}

function QueueViz({ queue, pointers }: Pick<VizState, 'queue' | 'pointers'>) {
  if (!queue || queue.length === 0) {
    return null;
  }
  const normalized = normalizePointers(pointers);
  return (
    <div className="queue-viz">
      <div className="queue-row">
        {queue.map((value, index) => {
          const pointerHits = normalized.filter((pointer) => pointer.index === index);
          return (
            <div key={`queue-${index}`} className={`queue-cell${pointerHits.length ? ' pointer' : ''}`}>
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
      <div className="queue-labels">
        <span>front</span>
        <span>back</span>
      </div>
    </div>
  );
}

function HashTableViz({ hashTable }: Pick<VizState, 'hashTable'>) {
  if (!hashTable) {
    return null;
  }
  return (
    <div className="hash-table">
      {hashTable.buckets.map((bucket, index) => (
        <div
          key={`bucket-${index}`}
          className={`hash-row${hashTable.activeBucket === index ? ' active' : ''}`}
        >
          <span className="hash-index">{index}</span>
          <div className="hash-bucket">
            {bucket.length === 0 ? (
              <span className="hash-empty">empty</span>
            ) : (
              bucket.map((entry, entryIndex) => (
                <span key={`entry-${index}-${entryIndex}`} className="chip">
                  {entry}
                </span>
              ))
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

type LayoutNode = {
  id: string;
  value: number | string;
  depth: number;
  xIndex: number;
};

const layoutTree = (root: TreeNode) => {
  const nodes: LayoutNode[] = [];
  const edges: { from: string; to: string }[] = [];
  let xCounter = 0;
  let idCounter = 0;
  let maxDepth = 0;

  const walk = (node: TreeNode, depth: number): string => {
    const leftId = node.left ? walk(node.left, depth + 1) : null;
    const id = `node-${idCounter}`;
    idCounter += 1;
    const xIndex = xCounter;
    xCounter += 1;
    nodes.push({ id, value: node.value, depth, xIndex });
    maxDepth = Math.max(maxDepth, depth);
    if (leftId) {
      edges.push({ from: id, to: leftId });
    }
    const rightId = node.right ? walk(node.right, depth + 1) : null;
    if (rightId) {
      edges.push({ from: id, to: rightId });
    }
    return id;
  };

  walk(root, 0);

  const maxX = nodes.reduce((acc, node) => Math.max(acc, node.xIndex), 0);
  return { nodes, edges, maxDepth, maxX };
};

function TreeViz({ tree }: Pick<VizState, 'tree'>) {
  if (!tree) {
    return null;
  }
  const { nodes, edges, maxDepth, maxX } = layoutTree(tree.root);
  const width = 360;
  const padding = 24;
  const rowHeight = 70;
  const height = padding * 2 + rowHeight * (maxDepth + 1);
  const span = maxX === 0 ? 1 : maxX;
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));

  const highlight = tree.highlight;
  const resolveX = (node: LayoutNode) => {
    if (maxX === 0) {
      return width / 2;
    }
    return padding + (node.xIndex / span) * (width - padding * 2);
  };

  return (
    <svg className="tree-svg" viewBox={`0 0 ${width} ${height}`}>
      {edges.map((edge) => {
        const from = nodeMap.get(edge.from);
        const to = nodeMap.get(edge.to);
        if (!from || !to) {
          return null;
        }
        const fromX = resolveX(from);
        const fromY = padding + from.depth * rowHeight;
        const toX = resolveX(to);
        const toY = padding + to.depth * rowHeight;
        return (
          <line
            key={`${edge.from}-${edge.to}`}
            x1={fromX}
            y1={fromY}
            x2={toX}
            y2={toY}
            stroke="#bfae9a"
            strokeWidth={2}
          />
        );
      })}
      {nodes.map((node) => {
        const x = resolveX(node);
        const y = padding + node.depth * rowHeight;
        const isHighlighted =
          highlight !== undefined && String(node.value) === String(highlight);
        return (
          <g key={node.id}>
            <circle
              cx={x}
              cy={y}
              r={18}
              fill={isHighlighted ? '#f08a5d' : '#fff6ea'}
              stroke="#bfae9a"
              strokeWidth={2}
            />
            <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fontWeight={700}>
              {node.value}
            </text>
          </g>
        );
      })}
    </svg>
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
      <LinkedListViz linkedList={state.linkedList} pointers={state.pointers} />
      <StackViz stack={state.stack} pointers={state.pointers} />
      <QueueViz queue={state.queue} pointers={state.pointers} />
      <HashTableViz hashTable={state.hashTable} />
      <TreeViz tree={state.tree} />
      <IntervalsViz intervals={state.intervals} />
      <GraphViz graph={state.graph} pointers={state.pointers} />
      <HeapViz heapLeft={state.heapLeft} heapRight={state.heapRight} />
      <MatrixViz matrix={state.matrix} />
      {state.notes && <div className="panel">{state.notes}</div>}
    </div>
  );
}
