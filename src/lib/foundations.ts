import { exampleArrays, exampleGraph } from './examples';
import type { LessonDefinition, Step, TreeNode, TreeState, VizState } from './types';
import { createSeededRandom, randomInt, range, shuffle } from './utils';

const makeStep = (title: string, state: VizState, codeLineHighlights?: number[]): Step => ({
  title,
  state,
  codeLineHighlights,
});

const arrayCode = `def array_ops(nums, insert_index, insert_value, delete_index):
    read = nums[insert_index]
    nums.append(insert_value)
    nums.insert(insert_index, insert_value)
    nums.pop(delete_index)
    return read, nums`;

const linkedListCode = `class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def insert_after(head, index, value):
    cur = head
    for _ in range(index):
        cur = cur.next
    node = Node(value, cur.next)
    cur.next = node
    return head`;

const stackCode = `stack = [1, 2, 3]
stack.append(4)   # push
top = stack.pop()  # pop`;

const queueCode = `from collections import deque

q = deque([1, 2, 3])
q.append(4)       # enqueue
front = q.popleft()  # dequeue`;

const hashTableCode = `def put(table, key, value):
    idx = key % len(table)
    table[idx].append((key, value))

def get(table, key):
    idx = key % len(table)
    for k, v in table[idx]:
        if k == key:
            return v
    return None`;

const treeCode = `class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def search(root, target):
    cur = root
    while cur:
        if target == cur.val:
            return True
        if target < cur.val:
            cur = cur.left
        else:
            cur = cur.right
    return False`;

const heapCode = `def heap_push(heap, value):
    heap.append(value)
    i = len(heap) - 1
    while i > 0:
        p = (i - 1) // 2
        if heap[p] <= heap[i]:
            break
        heap[p], heap[i] = heap[i], heap[p]
        i = p`;

const graphCode = `def add_edge(graph, u, v):
    graph[u].append(v)
    graph[v].append(u)`;

const linearSearchCode = `def linear_search(nums, target):
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1`;

const binarySearchCode = `def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1`;

const mergeSortCode = `def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)

def merge(left, right):
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out`;

const quickSortCode = `def quick_sort(nums):
    if len(nums) <= 1:
        return nums
    pivot = nums[-1]
    left = [x for x in nums[:-1] if x <= pivot]
    right = [x for x in nums[:-1] if x > pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)`;

const bfsCode = `from collections import deque

def bfs(graph, start):
    q = deque([start])
    seen = {start}
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return order`;

const dfsCode = `def dfs(graph, start):
    stack = [start]
    seen = set()
    order = []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for nxt in reversed(graph[node]):
            if nxt not in seen:
                stack.append(nxt)
    return order`;

type ArrayOpsInput = {
  array: number[];
  insertIndex: number;
  insertValue: number;
  deleteIndex: number;
};

const makeArraySteps = ({ array, insertIndex, insertValue, deleteIndex }: ArrayOpsInput) => {
  const steps: Step[] = [];
  steps.push(
    makeStep(
      'Index access in O(1)',
      {
        array,
        pointers: [{ name: 'i', index: insertIndex }],
        notes: `Read array[${insertIndex}] instantly.`,
      },
      [2]
    )
  );

  const afterAppend = [...array, insertValue];
  steps.push(
    makeStep(
      'Append at the end',
      {
        array: afterAppend,
        pointers: [{ name: 'end', index: afterAppend.length - 1 }],
        notes: 'Append is amortized O(1).',
      },
      [3]
    )
  );

  const afterInsert = [...array.slice(0, insertIndex), insertValue, ...array.slice(insertIndex)];
  steps.push(
    makeStep(
      'Insert needs shifting',
      {
        array: afterInsert,
        pointers: [{ name: 'insert', index: insertIndex }],
        notes: 'Elements to the right shift by one.',
      },
      [4]
    )
  );

  const safeDeleteIndex = Math.min(Math.max(deleteIndex, 0), afterInsert.length - 1);
  const afterDelete = afterInsert.filter((_, index) => index !== safeDeleteIndex);
  steps.push(
    makeStep(
      'Delete collapses the gap',
      {
        array: afterDelete,
        pointers: [{ name: 'del', index: Math.min(safeDeleteIndex, afterDelete.length - 1) }],
        notes: 'Delete shifts remaining items left.',
      },
      [5]
    )
  );
  return steps;
};

type LinkedListInput = {
  values: number[];
  insertIndex: number;
  insertValue: number;
};

const makeLinkedListSteps = ({ values, insertIndex, insertValue }: LinkedListInput) => {
  const steps: Step[] = [];
  steps.push(
    makeStep('Start at head', {
      linkedList: values,
      pointers: [{ name: 'head', index: 0 }],
      notes: 'Nodes are connected by next pointers.',
    })
  );
  steps.push(
    makeStep('Traverse to insertion point', {
      linkedList: values,
      pointers: [{ name: 'cur', index: insertIndex }],
      notes: 'Move one node at a time.',
    })
  );
  const afterInsert = [
    ...values.slice(0, insertIndex + 1),
    insertValue,
    ...values.slice(insertIndex + 1),
  ];
  steps.push(
    makeStep('Rewire next pointers', {
      linkedList: afterInsert,
      pointers: [{ name: 'new', index: insertIndex + 1 }],
      notes: 'Insert is O(1) once the node is found.',
    })
  );
  return steps;
};

type StackInput = { stack: number[]; pushValue: number };
const makeStackSteps = ({ stack, pushValue }: StackInput) => {
  const steps: Step[] = [];
  const topIndex = Math.max(stack.length - 1, 0);
  steps.push(
    makeStep('Top of stack', {
      stack,
      pointers: stack.length ? [{ name: 'top', index: topIndex }] : undefined,
      notes: 'Last in, first out.',
    })
  );
  const afterPush = [...stack, pushValue];
  steps.push(
    makeStep('Push onto stack', {
      stack: afterPush,
      pointers: [{ name: 'top', index: afterPush.length - 1 }],
      notes: `Push ${pushValue} to the top.`,
    })
  );
  const afterPop = afterPush.slice(0, -1);
  steps.push(
    makeStep('Pop from stack', {
      stack: afterPop,
      pointers: afterPop.length ? [{ name: 'top', index: afterPop.length - 1 }] : undefined,
      notes: 'Pop removes the most recent item.',
    })
  );
  return steps;
};

type QueueInput = { queue: number[]; enqueueValue: number };
const makeQueueSteps = ({ queue, enqueueValue }: QueueInput) => {
  const steps: Step[] = [];
  steps.push(
    makeStep('Front of the line', {
      queue,
      pointers: queue.length
        ? [
            { name: 'front', index: 0 },
            { name: 'back', index: queue.length - 1 },
          ]
        : undefined,
      notes: 'First in, first out.',
    })
  );
  const afterEnqueue = [...queue, enqueueValue];
  steps.push(
    makeStep('Enqueue to the back', {
      queue: afterEnqueue,
      pointers: [
        { name: 'front', index: 0 },
        { name: 'back', index: afterEnqueue.length - 1 },
      ],
      notes: `Add ${enqueueValue} to the back.`,
    })
  );
  const afterDequeue = afterEnqueue.slice(1);
  steps.push(
    makeStep('Dequeue from the front', {
      queue: afterDequeue,
      pointers: afterDequeue.length
        ? [
            { name: 'front', index: 0 },
            { name: 'back', index: afterDequeue.length - 1 },
          ]
        : undefined,
      notes: 'Remove the oldest item.',
    })
  );
  return steps;
};

type HashTableInput = { buckets: (string | number)[][]; key: number; value: number };
const makeHashTableSteps = ({ buckets, key, value }: HashTableInput) => {
  const steps: Step[] = [];
  const bucketIndex = key % buckets.length;
  steps.push(
    makeStep('Hash key to bucket', {
      hashTable: { buckets, activeBucket: bucketIndex },
      notes: `Index = ${key} % ${buckets.length} = ${bucketIndex}.`,
    })
  );
  const updatedBuckets = buckets.map((bucket, index) =>
    index === bucketIndex ? [...bucket, `${key}:${value}`] : [...bucket]
  );
  steps.push(
    makeStep('Insert into bucket', {
      hashTable: { buckets: updatedBuckets, activeBucket: bucketIndex },
      notes: 'Collisions live in the same bucket.',
    })
  );
  steps.push(
    makeStep('Lookup scans the bucket', {
      hashTable: { buckets: updatedBuckets, activeBucket: bucketIndex },
      notes: 'Check key-value pairs in this bucket.',
    })
  );
  return steps;
};

type TreeSearchInput = { tree: TreeState; target: number };
const makeTreeSteps = ({ tree, target }: TreeSearchInput) => {
  const steps: Step[] = [];
  let node: TreeNode | undefined = tree.root;
  while (node) {
    steps.push(
      makeStep(`Visit ${node.value}`, {
        tree: { root: tree.root, highlight: node.value },
        notes: `Compare ${target} with ${node.value}.`,
      })
    );
    if (node.value === target) {
      steps.push(
        makeStep('Target found', {
          tree: { root: tree.root, highlight: node.value },
          notes: `${target} is in the tree.`,
        })
      );
      return steps;
    }
    node = target < Number(node.value) ? node.left : node.right;
  }
  steps.push(
    makeStep('Target not found', {
      tree: { root: tree.root },
      notes: `${target} is not in this tree.`,
    })
  );
  return steps;
};

type HeapInput = { heap: number[]; insertValue: number };
const makeHeapSteps = ({ heap, insertValue }: HeapInput) => {
  const steps: Step[] = [];
  const working = [...heap];
  steps.push(
    makeStep('Valid min-heap array', {
      array: working,
      notes: 'Parent values are smaller than their children.',
    })
  );
  working.push(insertValue);
  let child = working.length - 1;
  steps.push(
    makeStep('Insert at the end', {
      array: working,
      pointers: [{ name: 'child', index: child }],
      notes: `Place ${insertValue} at the next leaf.`,
    })
  );
  while (child > 0) {
    const parent = Math.floor((child - 1) / 2);
    if (working[parent] <= working[child]) {
      break;
    }
    [working[parent], working[child]] = [working[child], working[parent]];
    steps.push(
      makeStep('Sift up', {
        array: working,
        pointers: [
          { name: 'parent', index: parent },
          { name: 'child', index: child },
        ],
        notes: 'Swap to restore heap order.',
      })
    );
    child = parent;
  }
  return steps;
};

type GraphInput = { graph: { nodes: string[]; edges: [string, string][] }; newEdge: [string, string] };
const makeGraphSteps = ({ graph, newEdge }: GraphInput) => {
  const steps: Step[] = [];
  steps.push(
    makeStep('Nodes and edges', {
      graph,
      notes: 'Edges describe connections between nodes.',
    })
  );
  const fromIndex = graph.nodes.indexOf(newEdge[0]);
  const toIndex = graph.nodes.indexOf(newEdge[1]);
  steps.push(
    makeStep('Add a connection', {
      graph,
      pointers: [
        { name: 'u', index: fromIndex },
        { name: 'v', index: toIndex },
      ],
      notes: `Connect ${newEdge[0]} to ${newEdge[1]}.`,
    })
  );
  const updatedGraph = { nodes: graph.nodes, edges: [...graph.edges, newEdge] };
  steps.push(
    makeStep('Graph updated', {
      graph: updatedGraph,
      notes: 'Adjacency lists now include the new edge.',
    })
  );
  return steps;
};

const buildAdjacency = (nodes: string[], edges: [string, string][]) => {
  const adjacency: Record<string, string[]> = {};
  nodes.forEach((node) => {
    adjacency[node] = [];
  });
  edges.forEach(([from, to]) => {
    adjacency[from].push(to);
    adjacency[to].push(from);
  });
  return adjacency;
};

type LinearSearchInput = { array: number[]; target: number };
const makeLinearSearchSteps = ({ array, target }: LinearSearchInput) => {
  const steps: Step[] = [];
  for (let i = 0; i < array.length; i += 1) {
    const hit = array[i] === target;
    steps.push(
      makeStep(
        `Check index ${i}`,
        {
          array,
          pointers: [{ name: 'i', index: i }],
          notes: `Compare ${array[i]} vs ${target}.`,
        },
        [2, 3]
      )
    );
    if (hit) {
      steps.push(
        makeStep(
          'Target found',
          {
            array,
            pointers: [{ name: 'i', index: i }],
            notes: `Return index ${i}.`,
          },
          [3]
        )
      );
      break;
    }
  }
  return steps;
};

type BinarySearchInput = { array: number[]; target: number };
const makeBinarySearchSteps = ({ array, target }: BinarySearchInput) => {
  const steps: Step[] = [];
  let l = 0;
  let r = array.length - 1;
  while (l <= r) {
    const mid = Math.floor((l + r) / 2);
    steps.push(
      makeStep(
        `Probe mid @ ${mid}`,
        {
          array,
          window: { l, r },
          pointers: [
            { name: 'l', index: l },
            { name: 'mid', index: mid },
            { name: 'r', index: r },
          ],
          notes: `Compare ${array[mid]} with ${target}.`,
        },
        [3, 4, 5]
      )
    );
    if (array[mid] === target) {
      steps.push(
        makeStep(
          'Target found',
          {
            array,
            window: { l, r },
            pointers: [{ name: 'mid', index: mid }],
            notes: `Return index ${mid}.`,
          },
          [5, 6]
        )
      );
      return steps;
    }
    if (array[mid] < target) {
      l = mid + 1;
      steps.push(
        makeStep(
          'Discard left half',
          {
            array,
            window: { l, r },
            pointers: [{ name: 'l', index: l }],
            notes: 'Target is larger than mid.',
          },
          [7, 8]
        )
      );
    } else {
      r = mid - 1;
      steps.push(
        makeStep(
          'Discard right half',
          {
            array,
            window: { l, r },
            pointers: [{ name: 'r', index: r }],
            notes: 'Target is smaller than mid.',
          },
          [9, 10]
        )
      );
    }
  }
  steps.push(
    makeStep('Target not found', {
      array,
      notes: 'Search space exhausted.',
    })
  );
  return steps;
};

type MergeSortInput = { array: number[] };
const makeMergeSortSteps = ({ array }: MergeSortInput) => {
  const steps: Step[] = [];
  const mid = Math.floor(array.length / 2);
  const left = array.slice(0, mid);
  const right = array.slice(mid);
  const leftSorted = [...left].sort((a, b) => a - b);
  const rightSorted = [...right].sort((a, b) => a - b);
  const merged = [...array].sort((a, b) => a - b);
  steps.push(
    makeStep('Split into halves', {
      matrix: [left, right],
      notes: 'Divide the array recursively.',
    })
  );
  steps.push(
    makeStep('Sort each half', {
      matrix: [leftSorted, rightSorted],
      notes: 'Merge sort orders each half.',
    })
  );
  steps.push(
    makeStep('Merge the halves', {
      array: merged,
      notes: 'Merge in sorted order.',
    })
  );
  return steps;
};

type QuickSortInput = { array: number[] };
const makeQuickSortSteps = ({ array }: QuickSortInput) => {
  const steps: Step[] = [];
  const pivot = array[array.length - 1];
  const left = array.slice(0, -1).filter((x) => x <= pivot);
  const right = array.slice(0, -1).filter((x) => x > pivot);
  const partitioned = [...left, pivot, ...right];
  const pivotIndex = left.length;
  steps.push(
    makeStep('Choose pivot', {
      array,
      pointers: [{ name: 'pivot', index: array.length - 1 }],
      notes: `Pivot = ${pivot}.`,
    })
  );
  steps.push(
    makeStep('Partition around pivot', {
      array: partitioned,
      pointers: [{ name: 'pivot', index: pivotIndex }],
      notes: 'Lower values left, higher values right.',
    })
  );
  steps.push(
    makeStep('Recurse on partitions', {
      matrix: [left, right],
      notes: 'Quick sort repeats on each side.',
    })
  );
  return steps;
};

type TraversalInput = { graph: { nodes: string[]; edges: [string, string][] }; start: string };
const makeBfsSteps = ({ graph, start }: TraversalInput) => {
  const steps: Step[] = [];
  const adjacency = buildAdjacency(graph.nodes, graph.edges);
  const queue: string[] = [start];
  const seen = new Set([start]);
  steps.push(
    makeStep('Initialize queue', {
      graph,
      queue: [...queue],
      pointers: [{ name: 'start', index: graph.nodes.indexOf(start) }],
      notes: 'Seed the queue with the start node.',
    })
  );
  while (queue.length > 0) {
    const node = queue.shift() as string;
    steps.push(
      makeStep('Dequeue next node', {
        graph,
        queue: [...queue],
        pointers: [{ name: 'node', index: graph.nodes.indexOf(node) }],
        notes: `Visit ${node}.`,
      })
    );
    adjacency[node].forEach((neighbor) => {
      if (!seen.has(neighbor)) {
        seen.add(neighbor);
        queue.push(neighbor);
        steps.push(
          makeStep('Enqueue neighbor', {
            graph,
            queue: [...queue],
            pointers: [{ name: 'nbr', index: graph.nodes.indexOf(neighbor) }],
            notes: `Discovered ${neighbor}.`,
          })
        );
      }
    });
  }
  return steps;
};

const makeDfsSteps = ({ graph, start }: TraversalInput) => {
  const steps: Step[] = [];
  const adjacency = buildAdjacency(graph.nodes, graph.edges);
  const stack: string[] = [start];
  const seen = new Set<string>();
  steps.push(
    makeStep('Initialize stack', {
      graph,
      stack: [...stack],
      pointers: [{ name: 'start', index: graph.nodes.indexOf(start) }],
      notes: 'Seed the stack with the start node.',
    })
  );
  while (stack.length > 0) {
    const node = stack.pop() as string;
    if (seen.has(node)) {
      continue;
    }
    seen.add(node);
    steps.push(
      makeStep('Pop node', {
        graph,
        stack: [...stack],
        pointers: [{ name: 'node', index: graph.nodes.indexOf(node) }],
        notes: `Visit ${node}.`,
      })
    );
    adjacency[node]
      .slice()
      .reverse()
      .forEach((neighbor) => {
        if (!seen.has(neighbor)) {
          stack.push(neighbor);
          steps.push(
            makeStep('Push neighbor', {
              graph,
              stack: [...stack],
              pointers: [{ name: 'nbr', index: graph.nodes.indexOf(neighbor) }],
              notes: `Stack ${neighbor} for later.`,
            })
          );
        }
      });
  }
  return steps;
};

const baseTree: TreeState = {
  root: {
    value: 8,
    left: {
      value: 3,
      left: { value: 1 },
      right: {
        value: 6,
        left: { value: 4 },
        right: { value: 7 },
      },
    },
    right: {
      value: 10,
      right: {
        value: 14,
        left: { value: 13 },
      },
    },
  },
};

export const dataStructures: LessonDefinition[] = [
  {
    slug: 'array',
    title: 'Arrays',
    summary: 'Contiguous storage with constant-time indexing.',
    kind: 'data-structure',
    description:
      'Arrays store items in a single block of memory. Index reads are fast, but inserts and deletes in the middle require shifting elements.',
    sections: [
      {
        title: 'Core operations',
        items: ['read by index O(1)', 'append amortized O(1)', 'insert/delete O(n)'],
      },
      {
        title: 'When to use',
        items: ['fast indexing', 'cache-friendly scans', 'tight memory layout'],
      },
      {
        title: 'Pitfalls',
        items: ['expensive middle edits', 'resizing overhead for dynamic arrays'],
      },
    ],
    complexity: { time: 'Read O(1), insert/delete O(n)', space: 'O(n)' },
    pythonCode: arrayCode,
    exampleProblems: [
      { title: 'Two Sum (sorted)' },
      { title: 'Rotate Array' },
      { title: 'Prefix Sum Queries' },
    ],
    presets: [
      {
        label: 'Insert + delete',
        input: { array: [3, 5, 8, 12], insertIndex: 2, insertValue: 6, deleteIndex: 1 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(5).map(() => randomInt(rng, 1, 12));
      return {
        array,
        insertIndex: randomInt(rng, 1, Math.max(1, array.length - 2)),
        insertValue: randomInt(rng, 1, 15),
        deleteIndex: randomInt(rng, 0, array.length - 1),
      };
    },
    makeSteps: makeArraySteps,
  },
  {
    slug: 'linked-list',
    title: 'Linked Lists',
    summary: 'Nodes connected by next pointers.',
    kind: 'data-structure',
    description:
      'Linked lists trade random access for cheap insertions and deletions once you have a pointer to a node.',
    sections: [
      {
        title: 'Core operations',
        items: ['insert/delete O(1) after finding node', 'search O(n)', 'append O(n)'],
      },
      {
        title: 'When to use',
        items: ['frequent inserts', 'unknown size', 'memory flexibility'],
      },
      {
        title: 'Pitfalls',
        items: ['no random access', 'extra pointer storage per node'],
      },
    ],
    complexity: { time: 'Search O(n), insert/delete O(1) after locate', space: 'O(n)' },
    pythonCode: linkedListCode,
    exampleProblems: [
      { title: 'Reverse Linked List' },
      { title: 'Merge Two Lists' },
      { title: 'Detect Cycle' },
    ],
    presets: [
      {
        label: 'Insert node',
        input: { values: [2, 5, 7, 9], insertIndex: 1, insertValue: 6 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const values = range(5).map(() => randomInt(rng, 1, 12));
      return {
        values,
        insertIndex: randomInt(rng, 0, values.length - 2),
        insertValue: randomInt(rng, 1, 12),
      };
    },
    makeSteps: makeLinkedListSteps,
  },
  {
    slug: 'stack',
    title: 'Stacks',
    summary: 'Last in, first out access.',
    kind: 'data-structure',
    description:
      'Stacks push and pop from one end. They are ideal for backtracking, undo stacks, and depth-first traversals.',
    sections: [
      {
        title: 'Core operations',
        items: ['push O(1)', 'pop O(1)', 'peek O(1)'],
      },
      {
        title: 'When to use',
        items: ['backtracking', 'expression evaluation', 'DFS traversal'],
      },
      {
        title: 'Pitfalls',
        items: ['no random access', 'stack overflow with deep recursion'],
      },
    ],
    complexity: { time: 'Push/pop O(1)', space: 'O(n)' },
    pythonCode: stackCode,
    exampleProblems: [
      { title: 'Valid Parentheses' },
      { title: 'Daily Temperatures' },
      { title: 'Decode String' },
    ],
    presets: [
      {
        label: 'Push + pop',
        input: { stack: [3, 7, 9], pushValue: 4 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const stack = range(4).map(() => randomInt(rng, 1, 10));
      return { stack, pushValue: randomInt(rng, 1, 10) };
    },
    makeSteps: makeStackSteps,
  },
  {
    slug: 'queue',
    title: 'Queues',
    summary: 'First in, first out ordering.',
    kind: 'data-structure',
    description:
      'Queues add to the back and remove from the front. They power breadth-first traversals and task scheduling.',
    sections: [
      {
        title: 'Core operations',
        items: ['enqueue O(1)', 'dequeue O(1)', 'peek O(1)'],
      },
      {
        title: 'When to use',
        items: ['BFS traversal', 'task scheduling', 'rate limiting'],
      },
      {
        title: 'Pitfalls',
        items: ['array-based queues need head tracking', 'unbounded growth if not drained'],
      },
    ],
    complexity: { time: 'Enqueue/dequeue O(1)', space: 'O(n)' },
    pythonCode: queueCode,
    exampleProblems: [
      { title: 'Number of Islands' },
      { title: 'Binary Tree Level Order' },
      { title: 'Shortest Path in Grid' },
    ],
    presets: [
      {
        label: 'Enqueue + dequeue',
        input: { queue: [1, 3, 5], enqueueValue: 8 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const queue = range(4).map(() => randomInt(rng, 1, 10));
      return { queue, enqueueValue: randomInt(rng, 1, 10) };
    },
    makeSteps: makeQueueSteps,
  },
  {
    slug: 'hash-table',
    title: 'Hash Tables',
    summary: 'Fast key-value access with hashing.',
    kind: 'data-structure',
    description:
      'Hash tables map keys to buckets using a hash function. Lookups are O(1) on average with good hashing.',
    sections: [
      {
        title: 'Core operations',
        items: ['put/get O(1) average', 'rehash when load factor grows', 'handle collisions'],
      },
      {
        title: 'When to use',
        items: ['fast lookups', 'deduping', 'caching'],
      },
      {
        title: 'Pitfalls',
        items: ['poor hash leads to O(n)', 'rehashing can be expensive'],
      },
    ],
    complexity: { time: 'Average O(1), worst O(n)', space: 'O(n)' },
    pythonCode: hashTableCode,
    exampleProblems: [
      { title: 'Two Sum' },
      { title: 'LRU Cache' },
      { title: 'Word Pattern' },
    ],
    presets: [
      {
        label: 'Insert key/value',
        input: {
          buckets: [['2:9'], [], ['7:1'], []],
          key: 10,
          value: 5,
        },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const buckets = range(4).map(() => [] as (string | number)[]);
      const seedKey = randomInt(rng, 1, 12);
      const seedValue = randomInt(rng, 1, 9);
      buckets[seedKey % buckets.length].push(`${seedKey}:${seedValue}`);
      const key = randomInt(rng, 1, 12);
      const value = randomInt(rng, 1, 9);
      return { buckets, key, value };
    },
    makeSteps: makeHashTableSteps,
  },
  {
    slug: 'binary-tree',
    title: 'Binary Search Trees',
    summary: 'Hierarchical search with ordered nodes.',
    kind: 'data-structure',
    description:
      'Binary search trees keep left values smaller and right values larger, enabling log-time searches when balanced.',
    sections: [
      {
        title: 'Core operations',
        items: ['search/insert/delete O(h)', 'inorder traversal yields sorted order', 'balance matters'],
      },
      {
        title: 'When to use',
        items: ['ordered sets', 'range queries', 'dynamic sorted data'],
      },
      {
        title: 'Pitfalls',
        items: ['skewed trees degrade to O(n)', 'needs rebalancing'],
      },
    ],
    complexity: { time: 'Average O(log n), worst O(n)', space: 'O(n)' },
    pythonCode: treeCode,
    exampleProblems: [
      { title: 'Validate BST' },
      { title: 'Kth Smallest in BST' },
      { title: 'Lowest Common Ancestor' },
    ],
    presets: [
      {
        label: 'Search path',
        input: { tree: baseTree, target: 7 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const target = randomInt(rng, 1, 14);
      return { tree: baseTree, target };
    },
    makeSteps: makeTreeSteps,
  },
  {
    slug: 'heap',
    title: 'Binary Heaps',
    summary: 'Priority queues backed by arrays.',
    kind: 'data-structure',
    description:
      'Heaps keep the smallest (or largest) element at the root and support fast insert and extract operations.',
    sections: [
      {
        title: 'Core operations',
        items: ['insert O(log n)', 'extract-min O(log n)', 'peek O(1)'],
      },
      {
        title: 'When to use',
        items: ['priority queues', 'scheduling', 'top-k problems'],
      },
      {
        title: 'Pitfalls',
        items: ['not fully sorted', 'index math for parent/child'],
      },
    ],
    complexity: { time: 'Insert/extract O(log n)', space: 'O(n)' },
    pythonCode: heapCode,
    exampleProblems: [
      { title: 'Kth Largest Element' },
      { title: 'Merge K Sorted Lists' },
      { title: 'Task Scheduler' },
    ],
    presets: [
      {
        label: 'Insert into heap',
        input: { heap: [1, 4, 7, 9, 6], insertValue: 3 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const heap = shuffle(rng, range(5, 1)).sort((a, b) => a - b);
      return { heap, insertValue: randomInt(rng, 1, 12) };
    },
    makeSteps: makeHeapSteps,
  },
  {
    slug: 'graph',
    title: 'Graphs',
    summary: 'Nodes connected by edges.',
    kind: 'data-structure',
    description:
      'Graphs model relationships with nodes and edges. They can be directed or undirected, weighted or unweighted.',
    sections: [
      {
        title: 'Core operations',
        items: ['add/remove edges', 'traverse neighbors', 'store via adjacency list'],
      },
      {
        title: 'When to use',
        items: ['networks', 'dependencies', 'routing problems'],
      },
      {
        title: 'Pitfalls',
        items: ['cycles', 'disconnected components', 'memory for dense graphs'],
      },
    ],
    complexity: { time: 'Edge ops O(1) average', space: 'O(V + E)' },
    pythonCode: graphCode,
    exampleProblems: [
      { title: 'Course Schedule' },
      { title: 'Clone Graph' },
      { title: 'Connected Components' },
    ],
    presets: [
      {
        label: 'Add edge',
        input: { graph: exampleGraph, newEdge: ['B', 'E'] },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const nodes = shuffle(rng, exampleGraph.nodes).slice(0, 5);
      const edges = exampleGraph.edges.filter(
        ([from, to]) => nodes.includes(from) && nodes.includes(to)
      );
      const newEdge = [nodes[0], nodes[nodes.length - 1]] as [string, string];
      return { graph: { nodes, edges }, newEdge };
    },
    makeSteps: makeGraphSteps,
  },
];

export const algorithms: LessonDefinition[] = [
  {
    slug: 'linear-search',
    title: 'Linear Search',
    summary: 'Scan left to right until you find the target.',
    kind: 'algorithm',
    description: 'Linear search checks each element in order. It is simple and works on unsorted data.',
    sections: [
      {
        title: 'Key steps',
        items: ['scan each item', 'compare to target', 'stop when found'],
      },
      {
        title: 'When to use',
        items: ['small arrays', 'unsorted data', 'one-off lookups'],
      },
      {
        title: 'Pitfalls',
        items: ['slow on large data', 'no early pruning'],
      },
    ],
    complexity: { time: 'O(n)', space: 'O(1)' },
    pythonCode: linearSearchCode,
    exampleProblems: [
      { title: 'Find First Occurrence' },
      { title: 'Unsorted Two Sum' },
      { title: 'Contains Duplicate' },
    ],
    presets: [
      {
        label: 'Find target',
        input: { array: [4, 7, 2, 9, 5], target: 9 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(6).map(() => randomInt(rng, 1, 12));
      const target = array[randomInt(rng, 0, array.length - 1)];
      return { array, target };
    },
    makeSteps: makeLinearSearchSteps,
  },
  {
    slug: 'binary-search',
    title: 'Binary Search',
    summary: 'Halve the search space every step.',
    kind: 'algorithm',
    description:
      'Binary search works on sorted arrays by comparing the middle element and discarding half of the range.',
    sections: [
      {
        title: 'Key steps',
        items: ['midpoint probe', 'discard half', 'repeat until found'],
      },
      {
        title: 'When to use',
        items: ['sorted arrays', 'monotonic predicates', 'log-time lookup'],
      },
      {
        title: 'Pitfalls',
        items: ['requires sorted data', 'off-by-one bounds'],
      },
    ],
    complexity: { time: 'O(log n)', space: 'O(1)' },
    pythonCode: binarySearchCode,
    exampleProblems: [
      { title: 'Search Insert Position' },
      { title: 'First Bad Version' },
      { title: 'Find Peak Element' },
    ],
    presets: [
      {
        label: 'Search sorted array',
        input: { array: exampleArrays.sorted, target: 5 },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(7).map(() => randomInt(rng, -3, 12)).sort((a, b) => a - b);
      const target = array[randomInt(rng, 0, array.length - 1)];
      return { array, target };
    },
    makeSteps: makeBinarySearchSteps,
  },
  {
    slug: 'merge-sort',
    title: 'Merge Sort',
    summary: 'Divide, sort, and merge.',
    kind: 'algorithm',
    description:
      'Merge sort splits the array, recursively sorts each half, and merges the results into a fully sorted list.',
    sections: [
      {
        title: 'Key steps',
        items: ['split the array', 'sort halves', 'merge with two pointers'],
      },
      {
        title: 'When to use',
        items: ['stable sorting', 'large data sets', 'linked list sorting'],
      },
      {
        title: 'Pitfalls',
        items: ['extra memory for merging', 'recursive overhead'],
      },
    ],
    complexity: { time: 'O(n log n)', space: 'O(n)' },
    pythonCode: mergeSortCode,
    exampleProblems: [
      { title: 'Sort List' },
      { title: 'Count Inversions' },
      { title: 'Merge K Sorted Lists' },
    ],
    presets: [
      {
        label: 'Split and merge',
        input: { array: [9, 4, 7, 3, 2, 8] },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(6).map(() => randomInt(rng, 1, 12));
      return { array };
    },
    makeSteps: makeMergeSortSteps,
  },
  {
    slug: 'quick-sort',
    title: 'Quick Sort',
    summary: 'Partition around a pivot.',
    kind: 'algorithm',
    description:
      'Quick sort chooses a pivot, partitions the array into smaller and larger values, and recurses on each side.',
    sections: [
      {
        title: 'Key steps',
        items: ['choose a pivot', 'partition elements', 'recurse on partitions'],
      },
      {
        title: 'When to use',
        items: ['fast average performance', 'in-place sorting', 'large arrays'],
      },
      {
        title: 'Pitfalls',
        items: ['worst-case O(n^2)', 'pivot choice matters'],
      },
    ],
    complexity: { time: 'Average O(n log n)', space: 'O(log n)' },
    pythonCode: quickSortCode,
    exampleProblems: [
      { title: 'Kth Largest Element' },
      { title: 'Sort Colors' },
      { title: 'Top K Frequent Elements' },
    ],
    presets: [
      {
        label: 'Partition step',
        input: { array: [8, 3, 7, 4, 9, 2] },
      },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(6).map(() => randomInt(rng, 1, 12));
      return { array };
    },
    makeSteps: makeQuickSortSteps,
  },
  {
    slug: 'bfs',
    title: 'Breadth-First Search',
    summary: 'Explore layer by layer with a queue.',
    kind: 'algorithm',
    description:
      'BFS visits nodes in expanding rings from a start node. It uses a queue to ensure the shallowest nodes are processed first.',
    sections: [
      {
        title: 'Key steps',
        items: ['enqueue start', 'dequeue + visit', 'enqueue neighbors'],
      },
      {
        title: 'When to use',
        items: ['shortest path in unweighted graphs', 'level order traversal', 'flood fill'],
      },
      {
        title: 'Pitfalls',
        items: ['track visited nodes', 'queue growth on dense graphs'],
      },
    ],
    complexity: { time: 'O(V + E)', space: 'O(V)' },
    pythonCode: bfsCode,
    exampleProblems: [
      { title: 'Shortest Path in Binary Matrix' },
      { title: 'Word Ladder' },
      { title: 'Minimum Depth of Tree' },
    ],
    presets: [
      {
        label: 'Traverse graph',
        input: { graph: exampleGraph, start: 'A' },
      },
    ],
    randomInput: () => ({
      graph: exampleGraph,
      start: 'A',
    }),
    makeSteps: makeBfsSteps,
  },
  {
    slug: 'dfs',
    title: 'Depth-First Search',
    summary: 'Dive deep with a stack.',
    kind: 'algorithm',
    description:
      'DFS explores as far as possible down a branch before backtracking. It uses recursion or an explicit stack.',
    sections: [
      {
        title: 'Key steps',
        items: ['push start', 'pop + visit', 'push neighbors'],
      },
      {
        title: 'When to use',
        items: ['cycle detection', 'topological sorting', 'path exploration'],
      },
      {
        title: 'Pitfalls',
        items: ['stack depth with recursion', 'track visited nodes'],
      },
    ],
    complexity: { time: 'O(V + E)', space: 'O(V)' },
    pythonCode: dfsCode,
    exampleProblems: [
      { title: 'Number of Islands' },
      { title: 'Clone Graph' },
      { title: 'Path Sum' },
    ],
    presets: [
      {
        label: 'Traverse graph',
        input: { graph: exampleGraph, start: 'A' },
      },
    ],
    randomInput: () => ({
      graph: exampleGraph,
      start: 'A',
    }),
    makeSteps: makeDfsSteps,
  },
];
