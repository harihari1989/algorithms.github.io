import { exampleArrays, exampleDag, exampleGraph, exampleIntervals, exampleLists } from './examples';
import type { PatternDefinition, Step, VizState } from './types';
import { createSeededRandom, randomInt, range, shuffle } from './utils';

const makeStep = (title: string, state: VizState, codeLineHighlights?: number[]): Step => ({
  title,
  state,
  codeLineHighlights,
});

const slidingWindowCode = `def sliding_window(nums, k):
    l = 0
    window_sum = 0
    best = 0

    for r, x in enumerate(nums):
        window_sum += x

        # shrink while invalid
        while r - l + 1 > k:
            window_sum -= nums[l]
            l += 1

        best = max(best, window_sum)

    return best`;

const twoPointersCode = `def two_pointers(nums, target):
    i, j = 0, len(nums) - 1
    while i < j:
        total = nums[i] + nums[j]
        if total == target:
            return [i, j]
        if total < target:
            i += 1
        else:
            j -= 1
    return []`;

const fastSlowCode = `class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: Node) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`;

const mergeIntervalsCode = `def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged`;

const cyclicSortCode = `def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        j = nums[i] - 1
        if nums[i] != nums[j]:
            nums[i], nums[j] = nums[j], nums[i]
        else:
            i += 1
    return nums`;

const reverseListCode = `def reverse_list(head):
    prev = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    return prev`;

const bfsCode = `from collections import deque

def bfs(start, neighbors):
    q = deque([start])
    seen = {start}

    while q:
        node = q.popleft()
        for nxt in neighbors(node):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)`;

const dfsCode = `def dfs(node, neighbors, seen=None):
    if seen is None:
        seen = set()
    seen.add(node)
    for nxt in neighbors(node):
        if nxt not in seen:
            dfs(nxt, neighbors, seen)
    return seen`;

const twoHeapsCode = `import heapq

class MedianFinder:
    def __init__(self):
        self.left = []   # max-heap via negatives
        self.right = []  # min-heap

    def add(self, num: int):
        if not self.left or num <= -self.left[0]:
            heapq.heappush(self.left, -num)
        else:
            heapq.heappush(self.right, num)

        if len(self.left) > len(self.right) + 1:
            heapq.heappush(self.right, -heapq.heappop(self.left))
        elif len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))

    def median(self) -> float:
        if len(self.left) > len(self.right):
            return float(-self.left[0])
        return (-self.left[0] + self.right[0]) / 2.0`;

const subsetsCode = `def subsets(nums):
    res = [[]]
    for x in nums:
        res += [cur + [x] for cur in res]
    return res`;

const modifiedBinarySearchCode = `def lower_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l`;

const topKCode = `import heapq

def kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)

    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)

    return heap[0]`;

const kWayMergeCode = `import heapq

def merge_k_sorted(lists):
    heap = []
    for i, arr in enumerate(lists):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    out = []
    while heap:
        val, i, j = heapq.heappop(heap)
        out.append(val)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))

    return out`;

const knapsackCode = `def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        w, v = weights[i], values[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)

    return dp[capacity]`;

const topoSortCode = `from collections import deque

def topo_sort(nodes, edges):
    graph = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}

    for u, v in edges:
        graph[u].append(v)
        indeg[v] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return order`;

const backtrackingCode = `def backtrack(path, choices, is_valid, on_solution):
    if on_solution(path):
        return

    for c in choices:
        if not is_valid(path, c):
            continue
        path.append(c)
        backtrack(path, choices, is_valid, on_solution)
        path.pop()`;

type SlidingWindowInput = { array: number[]; k: number };
const makeSlidingWindowSteps = ({ array, k }: SlidingWindowInput) => {
  const steps: Step[] = [];
  let l = 0;
  let sum = 0;
  for (let r = 0; r < array.length; r += 1) {
    sum += array[r];
    steps.push(
      makeStep(
        `Expand r to ${r}`,
        {
          array,
          window: { l, r },
          pointers: [
            { name: 'l', index: l, color: '#ef476f' },
            { name: 'r', index: r, color: '#118ab2' },
          ],
          notes: `Add ${array[r]} -> sum = ${sum}.`,
        },
        [5, 6, 7]
      )
    );
    while (r - l + 1 > k) {
      sum -= array[l];
      steps.push(
        makeStep(
          `Shrink from l ${l}`,
          {
            array,
            window: { l: l + 1, r },
            pointers: [
              { name: 'l', index: l + 1, color: '#ef476f' },
              { name: 'r', index: r, color: '#118ab2' },
            ],
            notes: `Remove ${array[l]} -> sum = ${sum}.`,
          },
          [9, 10, 11]
        )
      );
      l += 1;
    }
  }
  steps.push(
    makeStep('Scan complete', {
      array,
      window: { l, r: array.length - 1 },
      pointers: [
        { name: 'l', index: l, color: '#ef476f' },
        { name: 'r', index: array.length - 1, color: '#118ab2' },
      ],
      notes: 'Track the best window during the scan.',
    })
  );
  return steps;
};

type TwoPointersInput = { array: number[]; target: number };
const makeTwoPointersSteps = ({ array, target }: TwoPointersInput) => {
  const steps: Step[] = [];
  let i = 0;
  let j = array.length - 1;
  while (i < j) {
    const total = array[i] + array[j];
    steps.push(
      makeStep(
        `Compare ${array[i]} + ${array[j]}`,
        {
          array,
          pointers: [
            { name: 'i', index: i, color: '#ef476f' },
            { name: 'j', index: j, color: '#118ab2' },
          ],
          notes: `Sum = ${total}, target = ${target}.`,
        },
        [2, 3, 4]
      )
    );
    if (total === target) {
      steps.push(
        makeStep('Pair found', {
          array,
          pointers: [
            { name: 'i', index: i, color: '#ef476f' },
            { name: 'j', index: j, color: '#118ab2' },
          ],
          notes: 'Pointers meet the target sum.',
        }, [5])
      );
      break;
    }
    if (total < target) {
      i += 1;
      steps.push(
        makeStep('Move i right', {
          array,
          pointers: [
            { name: 'i', index: i, color: '#ef476f' },
            { name: 'j', index: j, color: '#118ab2' },
          ],
          notes: 'Sum is too small, grow the left pointer.',
        }, [7])
      );
    } else {
      j -= 1;
      steps.push(
        makeStep('Move j left', {
          array,
          pointers: [
            { name: 'i', index: i, color: '#ef476f' },
            { name: 'j', index: j, color: '#118ab2' },
          ],
          notes: 'Sum is too large, shrink the right pointer.',
        }, [8])
      );
    }
  }
  return steps;
};

type FastSlowInput = { array: number[] };
const makeFastSlowSteps = ({ array }: FastSlowInput) => {
  const steps: Step[] = [];
  let slow = 0;
  let fast = 0;
  while (fast < array.length && fast + 1 < array.length) {
    steps.push(
      makeStep(
        `Slow @ ${slow}, Fast @ ${fast}`,
        {
          array,
          pointers: [
            { name: 'slow', index: slow, color: '#ffd166' },
            { name: 'fast', index: fast, color: '#06d6a0' },
          ],
          notes: 'Fast moves twice as quickly as slow.',
        },
        [7, 8]
      )
    );
    slow += 1;
    fast += 2;
  }
  steps.push(
    makeStep('Fast reached the end', {
      array,
      pointers: [
        { name: 'slow', index: Math.min(slow, array.length - 1), color: '#ffd166' },
        { name: 'fast', index: Math.min(fast, array.length - 1), color: '#06d6a0' },
      ],
      notes: 'If slow and fast ever meet, a cycle exists.',
    })
  );
  return steps;
};

type MergeIntervalsInput = { intervals: { start: number; end: number }[] };
const makeMergeIntervalsSteps = ({ intervals }: MergeIntervalsInput) => {
  const sorted = [...intervals].sort((a, b) => a.start - b.start);
  const merged: { start: number; end: number }[] = [];
  const steps: Step[] = [];
  sorted.forEach((interval, index) => {
    if (!merged.length || interval.start > merged[merged.length - 1].end) {
      merged.push({ ...interval });
      steps.push(
        makeStep(
          `Start new interval at ${interval.start}`,
          {
            intervals: sorted,
            notes: `Merged: ${merged.map((m) => `[${m.start}, ${m.end}]`).join(', ')}`,
            pointers: [{ name: 'cur', index, color: '#ef476f' }],
          },
          [4, 5, 6]
        )
      );
    } else {
      merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, interval.end);
      steps.push(
        makeStep(
          `Merge with previous (${interval.start}, ${interval.end})`,
          {
            intervals: sorted,
            notes: `Merged: ${merged.map((m) => `[${m.start}, ${m.end}]`).join(', ')}`,
            pointers: [{ name: 'cur', index, color: '#118ab2' }],
          },
          [7, 8]
        )
      );
    }
  });
  return steps;
};

type CyclicSortInput = { array: number[] };
const makeCyclicSortSteps = ({ array }: CyclicSortInput) => {
  const steps: Step[] = [];
  const nums = [...array];
  let i = 0;
  while (i < nums.length) {
    const j = nums[i] - 1;
    steps.push(
      makeStep(
        `Inspect index ${i}`,
        {
          array: nums,
          pointers: [
            { name: 'i', index: i, color: '#ef476f' },
            { name: 'target', index: j, color: '#06d6a0' },
          ],
          notes: `Value ${nums[i]} should move to index ${j}.`,
        },
        [3, 4]
      )
    );
    if (nums[i] !== nums[j]) {
      [nums[i], nums[j]] = [nums[j], nums[i]];
      steps.push(
        makeStep(
          `Swap index ${i} with ${j}`,
          {
            array: nums,
            pointers: [
              { name: 'i', index: i, color: '#ef476f' },
              { name: 'target', index: j, color: '#06d6a0' },
            ],
            notes: 'Swap until each value lands in its correct slot.',
          },
          [5]
        )
      );
    } else {
      i += 1;
    }
  }
  steps.push(
    makeStep('All numbers placed', {
      array: nums,
      notes: 'Array is now sorted by index mapping.',
    })
  );
  return steps;
};

type ReverseListInput = { list: number[] };
const makeReverseListSteps = ({ list }: ReverseListInput) => {
  const steps: Step[] = [];
  let prev = -1;
  let cur = 0;
  while (cur < list.length) {
    const next = cur + 1;
    const pointers = [
      ...(prev >= 0 ? [{ name: 'prev', index: prev, color: '#ef476f' }] : []),
      { name: 'cur', index: cur, color: '#118ab2' },
      ...(next < list.length ? [{ name: 'next', index: next, color: '#06d6a0' }] : []),
    ];
    steps.push(
      makeStep(
        `Rewire node ${cur}`,
        {
          array: list,
          pointers,
          notes: prev < 0 ? 'Prev is None; start reversing.' : `Point ${list[cur]} back to ${list[prev]}.`,
        },
        [2, 3, 4]
      )
    );
    prev = cur;
    cur = next;
  }
  steps.push(
    makeStep('List reversed', {
      array: list,
      notes: 'Prev now points to the new head.',
    })
  );
  return steps;
};

type GraphInput = { graph: { nodes: string[]; edges: [string, string][] }; start: string };
const makeBfsSteps = ({ graph, start }: GraphInput) => {
  const steps: Step[] = [];
  const queue = [start];
  const seen = new Set([start]);
  while (queue.length) {
    const node = queue.shift() as string;
    steps.push(
      makeStep(
        `Visit ${node}`,
        {
          graph,
          pointers: [{ name: 'current', index: graph.nodes.indexOf(node), color: '#ef476f' }],
          notes: `Queue: ${queue.join(', ') || 'empty'} | Seen: ${[...seen].join(', ')}`,
        },
        [6, 7]
      )
    );
    const neighbors = graph.edges.filter(([from]) => from === node).map(([, to]) => to);
    neighbors.forEach((next) => {
      if (!seen.has(next)) {
        seen.add(next);
        queue.push(next);
        steps.push(
          makeStep(
            `Enqueue ${next}`,
            {
              graph,
              pointers: [{ name: 'enqueue', index: graph.nodes.indexOf(next), color: '#06d6a0' }],
              notes: `Queue: ${queue.join(', ')}`,
            },
            [8, 9, 10]
          )
        );
      }
    });
  }
  return steps;
};

const makeDfsSteps = ({ graph, start }: GraphInput) => {
  const steps: Step[] = [];
  const stack = [start];
  const seen = new Set<string>();
  while (stack.length) {
    const node = stack.pop() as string;
    if (seen.has(node)) {
      continue;
    }
    seen.add(node);
    steps.push(
      makeStep(
        `Depth-first visit ${node}`,
        {
          graph,
          pointers: [{ name: 'current', index: graph.nodes.indexOf(node), color: '#118ab2' }],
          notes: `Stack: ${stack.join(', ') || 'empty'} | Seen: ${[...seen].join(', ')}`,
        },
        [4, 5, 6]
      )
    );
    const neighbors = graph.edges
      .filter(([from]) => from === node)
      .map(([, to]) => to)
      .reverse();
    neighbors.forEach((next) => {
      if (!seen.has(next)) {
        stack.push(next);
      }
    });
  }
  return steps;
};

type TwoHeapsInput = { stream: number[] };
const makeTwoHeapsSteps = ({ stream }: TwoHeapsInput) => {
  const steps: Step[] = [];
  let left: number[] = [];
  let right: number[] = [];

  const rebalance = () => {
    if (left.length > right.length + 1) {
      const maxValue = Math.max(...left);
      const maxIndex = left.indexOf(maxValue);
      right = [...right, maxValue];
      left = left.filter((_, index) => index !== maxIndex);
    } else if (right.length > left.length) {
      const minValue = Math.min(...right);
      const minIndex = right.indexOf(minValue);
      left = [...left, minValue];
      right = right.filter((_, index) => index !== minIndex);
    }
  };

  stream.forEach((num) => {
    if (!left.length || num <= Math.max(...left)) {
      left = [...left, num];
    } else {
      right = [...right, num];
    }
    rebalance();
    left = [...left].sort((a, b) => b - a);
    right = [...right].sort((a, b) => a - b);
    const median = left.length === right.length
      ? (left[0] + right[0]) / 2
      : left[0];
    steps.push(
      makeStep(
        `Add ${num}`,
        {
          heapLeft: left,
          heapRight: right,
          notes: `Median now ${median}.`,
        },
        [8, 9, 10, 12]
      )
    );
  });
  return steps;
};

type SubsetsInput = { array: number[] };
const makeSubsetsSteps = ({ array }: SubsetsInput) => {
  const steps: Step[] = [];
  let subsets: number[][] = [[]];
  steps.push(
    makeStep('Start with empty subset', {
      array,
      notes: `Subsets: ${JSON.stringify(subsets)}`,
    }, [1])
  );
  array.forEach((value, index) => {
    const additions = subsets.map((current) => [...current, value]);
    subsets = [...subsets, ...additions];
    steps.push(
      makeStep(
        `Add ${value} to existing subsets`,
        {
          array,
          pointers: [{ name: 'item', index, color: '#ef476f' }],
          notes: `Total subsets: ${subsets.length}`,
        },
        [2, 3]
      )
    );
  });
  return steps;
};

type ModifiedBinarySearchInput = { array: number[]; target: number };
const makeModifiedBinarySearchSteps = ({ array, target }: ModifiedBinarySearchInput) => {
  const steps: Step[] = [];
  let l = 0;
  let r = array.length;
  while (l < r) {
    const mid = Math.floor((l + r) / 2);
    steps.push(
      makeStep(
        `Check mid ${mid}`,
        {
          array,
          window: { l, r: r - 1 },
          pointers: [{ name: 'mid', index: mid, color: '#06d6a0' }],
          notes: `nums[mid] = ${array[mid]}, target = ${target}.`,
        },
        [2, 3, 4]
      )
    );
    if (array[mid] < target) {
      l = mid + 1;
      steps.push(
        makeStep('Shift left bound right', {
          array,
          window: { l, r: r - 1 },
          pointers: [{ name: 'mid', index: mid, color: '#06d6a0' }],
          notes: 'Target is in the right half.',
        }, [5, 6])
      );
    } else {
      r = mid;
      steps.push(
        makeStep('Shift right bound left', {
          array,
          window: { l, r: r - 1 },
          pointers: [{ name: 'mid', index: mid, color: '#06d6a0' }],
          notes: 'Target is in the left half.',
        }, [7, 8])
      );
    }
  }
  steps.push(
    makeStep('Lower bound found', {
      array,
      window: { l, r: l },
      pointers: [{ name: 'l', index: l, color: '#ef476f' }],
      notes: `Insert position = ${l}.`,
    })
  );
  return steps;
};

type TopKInput = { array: number[]; k: number };
const makeTopKSteps = ({ array, k }: TopKInput) => {
  const steps: Step[] = [];
  let heap = [...array.slice(0, k)].sort((a, b) => a - b);
  steps.push(
    makeStep('Seed heap with first k elements', {
      array,
      heapRight: heap,
      notes: `Heap size ${k}.`,
    }, [3, 4])
  );
  array.slice(k).forEach((value) => {
    if (value > heap[0]) {
      heap = [...heap.slice(1), value].sort((a, b) => a - b);
      steps.push(
        makeStep('Replace smallest in heap', {
          array,
          heapRight: heap,
          notes: `Inserted ${value}.`,
        }, [6, 7])
      );
    } else {
      steps.push(
        makeStep('Skip smaller element', {
          array,
          heapRight: heap,
          notes: `${value} is below current top-k threshold.`,
        }, [6])
      );
    }
  });
  return steps;
};

type KWayMergeInput = { lists: number[][] };
const makeKWayMergeSteps = ({ lists }: KWayMergeInput) => {
  const steps: Step[] = [];
  const indices = lists.map(() => 0);
  const output: number[] = [];
  const total = lists.reduce((sum, list) => sum + list.length, 0);

  for (let step = 0; step < Math.min(total, 6); step += 1) {
    let minValue = Number.POSITIVE_INFINITY;
    let minList = -1;
    lists.forEach((list, i) => {
      const idx = indices[i];
      if (idx < list.length && list[idx] < minValue) {
        minValue = list[idx];
        minList = i;
      }
    });
    output.push(minValue);
    indices[minList] += 1;
    steps.push(
      makeStep(
        `Pop ${minValue} from list ${minList + 1}`,
        {
          matrix: lists,
          notes: `Output: ${output.join(', ')} | Indices: ${indices.join(', ')}`,
        },
        [7, 8, 9]
      )
    );
  }
  steps.push(
    makeStep('Merge continues...', {
      matrix: lists,
      notes: 'Repeat pop + push until the heap is empty.',
    })
  );
  return steps;
};

type KnapsackInput = { weights: number[]; values: number[]; capacity: number };
const makeKnapsackSteps = ({ weights, values, capacity }: KnapsackInput) => {
  const steps: Step[] = [];
  const dp = Array.from({ length: weights.length + 1 }, () => Array(capacity + 1).fill(0));
  steps.push(
    makeStep('Initialize DP table', {
      matrix: dp.map((row) => [...row]),
      notes: 'Row 0 and column 0 stay zero.',
    }, [2, 3])
  );
  for (let i = 1; i <= weights.length; i += 1) {
    const w = weights[i - 1];
    const v = values[i - 1];
    for (let c = 1; c <= capacity; c += 1) {
      dp[i][c] = dp[i - 1][c];
      if (c >= w) {
        dp[i][c] = Math.max(dp[i][c], dp[i - 1][c - w] + v);
      }
    }
    steps.push(
      makeStep(
        `Process item ${i} (w=${w}, v=${v})`,
        {
          matrix: dp.map((row) => [...row]),
          notes: 'Decide pick vs skip for each capacity.',
        },
        [5, 6, 7]
      )
    );
  }
  return steps;
};

type TopoInput = { nodes: string[]; edges: [string, string][] };
const makeTopoSteps = ({ nodes, edges }: TopoInput) => {
  const steps: Step[] = [];
  const indeg = new Map(nodes.map((node) => [node, 0]));
  edges.forEach(([, to]) => indeg.set(to, (indeg.get(to) ?? 0) + 1));
  let queue = nodes.filter((node) => (indeg.get(node) ?? 0) === 0);
  const order: string[] = [];
  while (queue.length) {
    const node = queue[0];
    queue = queue.slice(1);
    order.push(node);
    steps.push(
      makeStep(
        `Pull ${node} (in-degree 0)`,
        {
          graph: { nodes, edges },
          pointers: [{ name: 'next', index: nodes.indexOf(node), color: '#ef476f' }],
          notes: `Queue: ${queue.join(', ') || 'empty'} | Order: ${order.join(' -> ')}`,
        },
        [10, 11, 12]
      )
    );
    edges.forEach(([from, to]) => {
      if (from === node) {
        indeg.set(to, (indeg.get(to) ?? 0) - 1);
        if (indeg.get(to) === 0) {
          queue = [...queue, to];
        }
      }
    });
  }
  return steps;
};

type BacktrackingInput = { choices: string[]; targetLength: number };
const makeBacktrackingSteps = ({ choices, targetLength }: BacktrackingInput) => {
  const steps: Step[] = [];
  const path: string[] = [];

  const walk = (depth: number) => {
    if (depth === targetLength) {
      steps.push(
        makeStep(
          'Solution found',
          {
            array: choices.map((_, index) => index + 1),
            notes: `Path: ${path.join(' -> ')}`,
          },
          [1, 2]
        )
      );
      return;
    }
    choices.forEach((choice, index) => {
      path.push(choice);
      steps.push(
        makeStep(
          `Choose ${choice}`,
          {
            array: choices.map((_, idx) => idx + 1),
            pointers: [{ name: 'depth', index, color: '#06d6a0' }],
            notes: `Path: ${path.join(' -> ')}`,
          },
          [5, 6]
        )
      );
      walk(depth + 1);
      path.pop();
      steps.push(
        makeStep(
          'Backtrack',
          {
            array: choices.map((_, idx) => idx + 1),
            notes: `Undo to: ${path.join(' -> ') || 'start'}`,
          },
          [8]
        )
      );
    });
  };

  walk(0);
  return steps.slice(0, 8);
};

export const patterns: PatternDefinition<any>[] = [
  {
    slug: 'sliding-window',
    title: 'Sliding Window',
    summary: 'Scan contiguous ranges while maintaining a live window invariant.',
    description:
      'Maintain a valid window [l..r] while expanding right and shrinking left as needed. Great for subarray and substring constraints.',
    signals: [
      'Contiguous subarray or substring',
      'Longest/shortest window',
      'Constraint like at most K distinct or sum >= target',
    ],
    invariants: ['Window remains valid after each adjustment', 'Only move left when the window is invalid'],
    pitfalls: ['Forgetting to update running sum/map on shrink', 'Mixing inclusive and exclusive bounds'],
    complexity: { time: 'O(n)', space: 'O(1) to O(k)' },
    pythonCode: slidingWindowCode,
    exampleProblems: [
      { title: 'Longest Substring with K Distinct Characters' },
      { title: 'Max Sum Subarray of Size K' },
    ],
    presets: [
      { label: 'Window size 3', input: { array: exampleArrays.window, k: 3 } },
      { label: 'Window size 4', input: { array: exampleArrays.window, k: 4 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const length = randomInt(rng, 6, 9);
      const array = range(length, 1).map(() => randomInt(rng, 1, 9));
      return { array, k: randomInt(rng, 2, Math.min(5, length)) };
    },
    makeSteps: makeSlidingWindowSteps,
  },
  {
    slug: 'two-pointers',
    title: 'Two Pointers',
    summary: 'Use two indices to converge on a target in a structured array.',
    description:
      'Move pointers from ends or along a sequence to satisfy ordering or sum conditions. Excellent for sorted arrays and partitioning.',
    signals: ['Sorted array or string', 'Pair sum / palindrome checks', 'Partitioning around a pivot'],
    invariants: ['Pointers move monotonically', 'Discard impossible pairs each step'],
    pitfalls: ['Moving the wrong pointer', 'Missing equality condition when target is found'],
    complexity: { time: 'O(n)', space: 'O(1)' },
    pythonCode: twoPointersCode,
    exampleProblems: [
      { title: 'Two Sum II (sorted input)' },
      { title: 'Valid Palindrome' },
    ],
    presets: [
      { label: 'Target 6', input: { array: exampleArrays.sorted, target: 6 } },
      { label: 'Target 4', input: { array: exampleArrays.sorted, target: 4 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = shuffle(rng, range(7, -3));
      array.sort((a, b) => a - b);
      return { array, target: randomInt(rng, -2, 10) };
    },
    makeSteps: makeTwoPointersSteps,
  },
  {
    slug: 'fast-slow-pointers',
    title: 'Fast & Slow Pointers',
    summary: 'Detect cycles or midpoints with two speeds.',
    description:
      'Advance one pointer twice as fast as the other to detect cycles or locate list midpoints without extra space.',
    signals: ['Linked list or index jumping', 'Cycle detection or midpoint'],
    invariants: ['Fast moves 2x slow', 'Meeting point implies a cycle'],
    pitfalls: ['Not checking fast and fast.next', 'Assuming arrays instead of actual links'],
    complexity: { time: 'O(n)', space: 'O(1)' },
    pythonCode: fastSlowCode,
    exampleProblems: [
      { title: 'Linked List Cycle' },
      { title: 'Middle of the Linked List' },
    ],
    presets: [
      { label: 'List of 7', input: { array: exampleArrays.medium } },
      { label: 'List of 6', input: { array: exampleArrays.window } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      return { array: range(randomInt(rng, 6, 9), 1).map(() => randomInt(rng, 1, 9)) };
    },
    makeSteps: makeFastSlowSteps,
  },
  {
    slug: 'merge-intervals',
    title: 'Merge Intervals',
    summary: 'Sort and merge overlaps to condense ranges.',
    description:
      'Sort by start time, then merge overlapping ranges to build a compact schedule or coverage list.',
    signals: ['Overlapping ranges', 'Scheduling or range compression'],
    invariants: ['Intervals processed in sorted order', 'Merged list stays disjoint'],
    pitfalls: ['Skipping sort step', 'Not updating end when overlap occurs'],
    complexity: { time: 'O(n log n)', space: 'O(n)' },
    pythonCode: mergeIntervalsCode,
    exampleProblems: [
      { title: 'Merge Intervals' },
      { title: 'Meeting Rooms II' },
    ],
    presets: [
      { label: 'Mixed overlaps', input: { intervals: exampleIntervals } },
      { label: 'Already sorted', input: { intervals: [...exampleIntervals].sort((a, b) => a.start - b.start) } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const intervals = range(5).map(() => {
        const start = randomInt(rng, 1, 12);
        return { start, end: start + randomInt(rng, 1, 4) };
      });
      return { intervals };
    },
    makeSteps: makeMergeIntervalsSteps,
  },
  {
    slug: 'cyclic-sort',
    title: 'Cyclic Sort',
    summary: 'Place each value in its correct index by swapping.',
    description:
      'When numbers are in a known range 1..n, swap each number into its target index to sort or spot missing values.',
    signals: ['Values are within 1..n', 'Need to find missing/duplicate numbers'],
    invariants: ['Each swap places at least one value correctly'],
    pitfalls: ['Forgetting to increment i only when correct', 'Assuming values outside range'],
    complexity: { time: 'O(n)', space: 'O(1)' },
    pythonCode: cyclicSortCode,
    exampleProblems: [
      { title: 'Cyclic Sort' },
      { title: 'Find the Missing Number' },
    ],
    presets: [
      { label: 'Simple shuffle', input: { array: [3, 1, 5, 4, 2] } },
      { label: 'Longer shuffle', input: { array: [2, 6, 4, 3, 1, 5] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const length = randomInt(rng, 5, 7);
      const array = shuffle(rng, range(length, 1));
      return { array };
    },
    makeSteps: makeCyclicSortSteps,
  },
  {
    slug: 'in-place-reversal',
    title: 'In-place Reversal of Linked List',
    summary: 'Reverse pointers without extra memory.',
    description:
      'Iterate through a list and reverse each pointer with prev/cur/next trackers. Used for reversing full or partial lists.',
    signals: ['Linked list reversal', 'Reverse every k nodes'],
    invariants: ['Prev is always the head of the reversed section', 'Cur moves forward one node per step'],
    pitfalls: ['Losing the next pointer', 'Not returning the new head'],
    complexity: { time: 'O(n)', space: 'O(1)' },
    pythonCode: reverseListCode,
    exampleProblems: [
      { title: 'Reverse Linked List' },
      { title: 'Reverse Nodes in k-Group' },
    ],
    presets: [
      { label: 'List length 5', input: { list: [1, 2, 3, 4, 5] } },
      { label: 'List length 4', input: { list: [10, 20, 30, 40] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      return { list: range(randomInt(rng, 4, 7), 1).map(() => randomInt(rng, 1, 9)) };
    },
    makeSteps: makeReverseListSteps,
  },
  {
    slug: 'bfs',
    title: 'BFS (Tree/Graph)',
    summary: 'Level-order traversal with a queue.',
    description:
      'Breadth-first search explores neighbors layer by layer, producing shortest paths in unweighted graphs.',
    signals: ['Shortest path in unweighted graph', 'Level order traversal'],
    invariants: ['Queue holds the frontier', 'Visited set prevents cycles'],
    pitfalls: ['Forgetting to mark visited on enqueue', 'Revisiting nodes'],
    complexity: { time: 'O(V + E)', space: 'O(V)' },
    pythonCode: bfsCode,
    exampleProblems: [
      { title: 'Binary Tree Level Order Traversal' },
      { title: 'Rotting Oranges' },
    ],
    presets: [
      { label: 'Graph from A', input: { graph: exampleGraph, start: 'A' } },
      { label: 'Graph from C', input: { graph: exampleGraph, start: 'C' } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const nodes = ['A', 'B', 'C', 'D', 'E'];
      const edges: [string, string][] = nodes
        .flatMap((node, index) => nodes.slice(index + 1).map((other) => [node, other] as [string, string]))
        .filter(() => rng() > 0.55);
      return { graph: { nodes, edges }, start: 'A' };
    },
    makeSteps: makeBfsSteps,
  },
  {
    slug: 'dfs',
    title: 'DFS (Tree/Graph)',
    summary: 'Go deep before you go wide.',
    description:
      'Depth-first search explores as far as possible along each branch before backtracking. Useful for connectivity and paths.',
    signals: ['Need to explore all paths', 'Connected components'],
    invariants: ['Stack/recursion keeps current path', 'Visited prevents cycles'],
    pitfalls: ['Missing base cases', 'Stack overflow on deep recursion'],
    complexity: { time: 'O(V + E)', space: 'O(V)' },
    pythonCode: dfsCode,
    exampleProblems: [
      { title: 'Number of Islands' },
      { title: 'Clone Graph' },
    ],
    presets: [
      { label: 'Graph from A', input: { graph: exampleGraph, start: 'A' } },
      { label: 'Graph from C', input: { graph: exampleGraph, start: 'C' } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const nodes = ['A', 'B', 'C', 'D', 'E'];
      const edges: [string, string][] = nodes
        .flatMap((node, index) => nodes.slice(index + 1).map((other) => [node, other] as [string, string]))
        .filter(() => rng() > 0.6);
      return { graph: { nodes, edges }, start: 'A' };
    },
    makeSteps: makeDfsSteps,
  },
  {
    slug: 'two-heaps',
    title: 'Two Heaps',
    summary: 'Balance two heaps to track medians.',
    description:
      'Maintain a max-heap for the lower half and a min-heap for the upper half to answer running median queries.',
    signals: ['Running median', 'Stream of numbers', 'Need both min and max halves'],
    invariants: ['Heap sizes differ by at most 1', 'All left <= all right'],
    pitfalls: ['Forgetting to rebalance after insert', 'Mixing heap order'],
    complexity: { time: 'O(log n) per insert', space: 'O(n)' },
    pythonCode: twoHeapsCode,
    exampleProblems: [
      { title: 'Find Median from Data Stream' },
      { title: 'Sliding Window Median' },
    ],
    presets: [
      { label: 'Stream A', input: { stream: [5, 2, 8, 1, 3] } },
      { label: 'Stream B', input: { stream: [10, 4, 6, 7, 2] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      return { stream: range(6, 1).map(() => randomInt(rng, 1, 12)) };
    },
    makeSteps: makeTwoHeapsSteps,
  },
  {
    slug: 'subsets',
    title: 'Subsets',
    summary: 'Generate power sets via BFS/DFS expansion.',
    description:
      'Build up subsets by either branching (DFS) or expanding layer by layer (BFS). Each step doubles the set size.',
    signals: ['Generate all combinations/subsets', 'Power set requested'],
    invariants: ['Every element is either included or excluded', 'Count doubles per element'],
    pitfalls: ['Mutating subset list in-place', 'Missing base empty subset'],
    complexity: { time: 'O(2^n)', space: 'O(2^n)' },
    pythonCode: subsetsCode,
    exampleProblems: [
      { title: 'Subsets' },
      { title: 'Subsets II' },
    ],
    presets: [
      { label: 'Array [1,2,3]', input: { array: [1, 2, 3] } },
      { label: 'Array [2,4,6]', input: { array: [2, 4, 6] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      return { array: range(randomInt(rng, 3, 4), 1).map(() => randomInt(rng, 1, 6)) };
    },
    makeSteps: makeSubsetsSteps,
  },
  {
    slug: 'modified-binary-search',
    title: 'Modified Binary Search',
    summary: 'Binary search with custom conditions.',
    description:
      'Adjust mid checks to find first/last occurrence, rotation pivots, or boundaries like lower/upper bound.',
    signals: ['Sorted array with twist', 'Need first/last occurrence'],
    invariants: ['Search space halves each step', 'Bounds define candidate range'],
    pitfalls: ['Infinite loops with mid calculation', 'Off-by-one in bounds'],
    complexity: { time: 'O(log n)', space: 'O(1)' },
    pythonCode: modifiedBinarySearchCode,
    exampleProblems: [
      { title: 'Find First and Last Position of Element in Sorted Array' },
      { title: 'Search in Rotated Sorted Array' },
    ],
    presets: [
      { label: 'Target 5', input: { array: [1, 2, 4, 4, 5, 7, 9], target: 5 } },
      { label: 'Target 4', input: { array: [1, 2, 4, 4, 5, 7, 9], target: 4 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(8, 1).map(() => randomInt(rng, 1, 12)).sort((a, b) => a - b);
      return { array, target: array[randomInt(rng, 0, array.length - 1)] };
    },
    makeSteps: makeModifiedBinarySearchSteps,
  },
  {
    slug: 'top-k-elements',
    title: 'Top K Elements',
    summary: 'Track the k largest or smallest items with a heap.',
    description:
      'Maintain a min-heap of size k. Any element larger than the heap root replaces it.',
    signals: ['Top K largest/smallest', 'Streaming data'],
    invariants: ['Heap size stays at k', 'Heap root is the kth element'],
    pitfalls: ['Using max-heap instead of min-heap', 'Not trimming heap'],
    complexity: { time: 'O(n log k)', space: 'O(k)' },
    pythonCode: topKCode,
    exampleProblems: [
      { title: 'Kth Largest Element in an Array' },
      { title: 'Top K Frequent Elements' },
    ],
    presets: [
      { label: 'k = 3', input: { array: [3, 2, 1, 5, 6, 4], k: 3 } },
      { label: 'k = 2', input: { array: [10, 7, 11, 5, 2, 13], k: 2 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const array = range(7, 1).map(() => randomInt(rng, 1, 15));
      return { array, k: randomInt(rng, 2, 4) };
    },
    makeSteps: makeTopKSteps,
  },
  {
    slug: 'k-way-merge',
    title: 'K-way Merge',
    summary: 'Merge multiple sorted lists using a heap.',
    description:
      'Use a min-heap to always pull the smallest current head among k lists, then push the next from that list.',
    signals: ['Merge multiple sorted sequences', 'Need ordered output'],
    invariants: ['Heap contains current heads', 'Output grows monotonically'],
    pitfalls: ['Forgetting to push the next item', 'Dropping list index info'],
    complexity: { time: 'O(n log k)', space: 'O(k)' },
    pythonCode: kWayMergeCode,
    exampleProblems: [
      { title: 'Merge k Sorted Lists' },
      { title: 'Smallest Range Covering Elements from K Lists' },
    ],
    presets: [
      { label: 'Three lists', input: { lists: exampleLists } },
      { label: 'Short lists', input: { lists: [[1, 3], [2, 4, 6], [0, 9]] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const lists = range(3).map(() => {
        const base = randomInt(rng, 1, 5);
        return [base, base + randomInt(rng, 2, 4), base + randomInt(rng, 5, 7)];
      });
      return { lists };
    },
    makeSteps: makeKWayMergeSteps,
  },
  {
    slug: 'knapsack',
    title: '0/1 Knapsack (DP)',
    summary: 'Choose items without exceeding capacity.',
    description:
      'Dynamic programming tracks the best value for each capacity as items are considered once.',
    signals: ['Pick items with weights/values', 'Capacity constraint'],
    invariants: ['Each item used at most once', 'DP row builds on previous row'],
    pitfalls: ['Iterating capacity forward (causes reuse)', 'Not initializing base row'],
    complexity: { time: 'O(n * C)', space: 'O(C)' },
    pythonCode: knapsackCode,
    exampleProblems: [
      { title: 'Partition Equal Subset Sum' },
      { title: '0/1 Knapsack' },
    ],
    presets: [
      { label: 'Capacity 6', input: { weights: [1, 2, 3], values: [6, 10, 12], capacity: 6 } },
      { label: 'Capacity 5', input: { weights: [2, 3, 4], values: [4, 5, 7], capacity: 5 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const weights = range(3).map(() => randomInt(rng, 1, 4));
      const values = range(3).map(() => randomInt(rng, 3, 9));
      return { weights, values, capacity: randomInt(rng, 4, 7) };
    },
    makeSteps: makeKnapsackSteps,
  },
  {
    slug: 'topological-sort',
    title: 'Topological Sort',
    summary: 'Order tasks in a DAG by dependencies.',
    description:
      'Compute in-degrees, enqueue zero-degree nodes, and peel them off to get a valid ordering.',
    signals: ['Dependencies in a DAG', 'Need valid ordering'],
    invariants: ['Queue always holds zero in-degree nodes', 'Edges removed exactly once'],
    pitfalls: ['Missing cycles', 'Not updating in-degrees'],
    complexity: { time: 'O(V + E)', space: 'O(V)' },
    pythonCode: topoSortCode,
    exampleProblems: [
      { title: 'Course Schedule II' },
      { title: 'Alien Dictionary' },
    ],
    presets: [
      { label: 'Study plan', input: exampleDag },
      { label: 'Simple chain', input: { nodes: ['A', 'B', 'C'], edges: [['A', 'B'], ['B', 'C']] } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const nodes = ['A', 'B', 'C', 'D'];
      const edges: [string, string][] = [
        ['A', 'B'],
        ['A', 'C'],
        ['B', 'D'],
        ['C', 'D'],
      ].filter(() => rng() > 0.3) as [string, string][];
      return { nodes, edges };
    },
    makeSteps: makeTopoSteps,
  },
  {
    slug: 'backtracking',
    title: 'Backtracking',
    summary: 'Explore choices with undo to satisfy constraints.',
    description:
      'Build candidates incrementally and backtrack whenever a constraint fails or a solution is found.',
    signals: ['Search all combinations with pruning', 'Need to explore permutations'],
    invariants: ['Path holds current decision', 'Undo after exploring a branch'],
    pitfalls: ['Missing prune checks', 'Mutating shared state without undo'],
    complexity: { time: 'O(b^d)', space: 'O(d)' },
    pythonCode: backtrackingCode,
    exampleProblems: [
      { title: 'N-Queens' },
      { title: 'Sudoku Solver' },
    ],
    presets: [
      { label: 'Choices ABC', input: { choices: ['A', 'B', 'C'], targetLength: 2 } },
      { label: 'Choices 123', input: { choices: ['1', '2', '3'], targetLength: 2 } },
    ],
    randomInput: (seed) => {
      const rng = createSeededRandom(seed);
      const choices = shuffle(rng, ['A', 'B', 'C', 'D']).slice(0, 3);
      return { choices, targetLength: 2 };
    },
    makeSteps: makeBacktrackingSteps,
  },
];

export const patternBySlug = Object.fromEntries(
  patterns.map((pattern) => [pattern.slug, pattern])
) as Record<string, PatternDefinition<any>>;
