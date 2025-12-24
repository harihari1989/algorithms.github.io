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

const bfsGraphCode = `from collections import deque

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

const dfsGraphCode = `def dfs(graph, start):
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

const bfsNeighborsCode = `from collections import deque

def bfs(start, neighbors):
    q = deque([start])
    seen = {start}

    while q:
        node = q.popleft()
        for nxt in neighbors(node):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)`;

const dfsNeighborsCode = `def dfs(node, neighbors, seen=None):
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

const dataStructures = [
  {
    slug: "array",
    title: "Arrays",
    summary: "Contiguous storage with constant-time indexing.",
    kind: "data-structure",
    description:
      "Arrays store items in a single block of memory. Index reads are fast, but inserts and deletes in the middle require shifting elements.",
    sections: [
      {
        title: "Core operations",
        items: ["read by index O(1)", "append amortized O(1)", "insert/delete O(n)"],
      },
      {
        title: "When to use",
        items: ["fast indexing", "cache-friendly scans", "tight memory layout"],
      },
      {
        title: "Pitfalls",
        items: ["expensive middle edits", "resizing overhead for dynamic arrays"],
      },
    ],
    complexity: { time: "Read O(1), insert/delete O(n)", space: "O(n)" },
    pythonCode: arrayCode,
    exampleProblems: ["Two Sum (sorted)", "Rotate Array", "Prefix Sum Queries"],
  },
  {
    slug: "linked-list",
    title: "Linked Lists",
    summary: "Nodes connected by next pointers.",
    kind: "data-structure",
    description:
      "Linked lists trade random access for cheap insertions and deletions once you have a pointer to a node.",
    sections: [
      {
        title: "Core operations",
        items: ["insert/delete O(1) after finding node", "search O(n)", "append O(n)"],
      },
      {
        title: "When to use",
        items: ["frequent inserts", "unknown size", "memory flexibility"],
      },
      {
        title: "Pitfalls",
        items: ["no random access", "extra pointer storage per node"],
      },
    ],
    complexity: { time: "Search O(n), insert/delete O(1) after locate", space: "O(n)" },
    pythonCode: linkedListCode,
    exampleProblems: ["Reverse Linked List", "Merge Two Lists", "Detect Cycle"],
  },
  {
    slug: "stack",
    title: "Stacks",
    summary: "Last in, first out access.",
    kind: "data-structure",
    description:
      "Stacks push and pop from one end. They are ideal for backtracking, undo stacks, and depth-first traversals.",
    sections: [
      {
        title: "Core operations",
        items: ["push O(1)", "pop O(1)", "peek O(1)"],
      },
      {
        title: "When to use",
        items: ["backtracking", "expression evaluation", "DFS traversal"],
      },
      {
        title: "Pitfalls",
        items: ["no random access", "stack overflow with deep recursion"],
      },
    ],
    complexity: { time: "Push/pop O(1)", space: "O(n)" },
    pythonCode: stackCode,
    exampleProblems: ["Valid Parentheses", "Daily Temperatures", "Decode String"],
  },
  {
    slug: "queue",
    title: "Queues",
    summary: "First in, first out ordering.",
    kind: "data-structure",
    description:
      "Queues add to the back and remove from the front. They power breadth-first traversals and task scheduling.",
    sections: [
      {
        title: "Core operations",
        items: ["enqueue O(1)", "dequeue O(1)", "peek O(1)"],
      },
      {
        title: "When to use",
        items: ["BFS traversal", "task scheduling", "rate limiting"],
      },
      {
        title: "Pitfalls",
        items: ["array-based queues need head tracking", "unbounded growth if not drained"],
      },
    ],
    complexity: { time: "Enqueue/dequeue O(1)", space: "O(n)" },
    pythonCode: queueCode,
    exampleProblems: ["Number of Islands", "Binary Tree Level Order", "Shortest Path in Grid"],
  },
  {
    slug: "hash-table",
    title: "Hash Tables",
    summary: "Fast key-value access with hashing.",
    kind: "data-structure",
    description:
      "Hash tables map keys to buckets using a hash function. Lookups are O(1) on average with good hashing.",
    sections: [
      {
        title: "Core operations",
        items: ["put/get O(1) average", "rehash when load factor grows", "handle collisions"],
      },
      {
        title: "When to use",
        items: ["fast lookups", "deduping", "caching"],
      },
      {
        title: "Pitfalls",
        items: ["poor hash leads to O(n)", "rehashing can be expensive"],
      },
    ],
    complexity: { time: "Average O(1), worst O(n)", space: "O(n)" },
    pythonCode: hashTableCode,
    exampleProblems: ["Two Sum", "LRU Cache", "Word Pattern"],
  },
  {
    slug: "binary-tree",
    title: "Binary Search Trees",
    summary: "Hierarchical search with ordered nodes.",
    kind: "data-structure",
    description:
      "Binary search trees keep left values smaller and right values larger, enabling log-time searches when balanced.",
    sections: [
      {
        title: "Core operations",
        items: ["search/insert/delete O(h)", "inorder traversal yields sorted order", "balance matters"],
      },
      {
        title: "When to use",
        items: ["ordered sets", "range queries", "dynamic sorted data"],
      },
      {
        title: "Pitfalls",
        items: ["skewed trees degrade to O(n)", "needs rebalancing"],
      },
    ],
    complexity: { time: "Average O(log n), worst O(n)", space: "O(n)" },
    pythonCode: treeCode,
    exampleProblems: ["Validate BST", "Kth Smallest in BST", "Lowest Common Ancestor"],
  },
  {
    slug: "heap",
    title: "Binary Heaps",
    summary: "Priority queues backed by arrays.",
    kind: "data-structure",
    description:
      "Heaps keep the smallest (or largest) element at the root and support fast insert and extract operations.",
    sections: [
      {
        title: "Core operations",
        items: ["insert O(log n)", "extract-min O(log n)", "peek O(1)"],
      },
      {
        title: "When to use",
        items: ["priority queues", "scheduling", "top-k problems"],
      },
      {
        title: "Pitfalls",
        items: ["not fully sorted", "index math for parent/child"],
      },
    ],
    complexity: { time: "Insert/extract O(log n)", space: "O(n)" },
    pythonCode: heapCode,
    exampleProblems: ["Kth Largest Element", "Merge K Sorted Lists", "Task Scheduler"],
  },
  {
    slug: "graph",
    title: "Graphs",
    summary: "Nodes connected by edges.",
    kind: "data-structure",
    description:
      "Graphs model relationships with nodes and edges. They can be directed or undirected, weighted or unweighted.",
    sections: [
      {
        title: "Core operations",
        items: ["add/remove edges", "traverse neighbors", "store via adjacency list"],
      },
      {
        title: "When to use",
        items: ["networks", "dependencies", "routing problems"],
      },
      {
        title: "Pitfalls",
        items: ["cycles", "disconnected components", "memory for dense graphs"],
      },
    ],
    complexity: { time: "Edge ops O(1) average", space: "O(V + E)" },
    pythonCode: graphCode,
    exampleProblems: ["Course Schedule", "Clone Graph", "Connected Components"],
  },
];

const algorithms = [
  {
    slug: "linear-search",
    title: "Linear Search",
    summary: "Scan left to right until you find the target.",
    kind: "algorithm",
    description: "Linear search checks each element in order. It is simple and works on unsorted data.",
    sections: [
      { title: "Key steps", items: ["scan each item", "compare to target", "stop when found"] },
      { title: "When to use", items: ["small arrays", "unsorted data", "one-off lookups"] },
      { title: "Pitfalls", items: ["slow on large data", "no early pruning"] },
    ],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: linearSearchCode,
    exampleProblems: ["Find First Occurrence", "Unsorted Two Sum", "Contains Duplicate"],
  },
  {
    slug: "binary-search",
    title: "Binary Search",
    summary: "Halve the search space every step.",
    kind: "algorithm",
    description: "Binary search works on sorted arrays by comparing the middle element and discarding half of the range.",
    sections: [
      { title: "Key steps", items: ["midpoint probe", "discard half", "repeat until found"] },
      { title: "When to use", items: ["sorted arrays", "monotonic predicates", "log-time lookup"] },
      { title: "Pitfalls", items: ["requires sorted data", "off-by-one bounds"] },
    ],
    complexity: { time: "O(log n)", space: "O(1)" },
    pythonCode: binarySearchCode,
    exampleProblems: ["Search Insert Position", "First Bad Version", "Find Peak Element"],
  },
  {
    slug: "merge-sort",
    title: "Merge Sort",
    summary: "Divide, sort, and merge.",
    kind: "algorithm",
    description: "Merge sort splits the array, recursively sorts each half, and merges the results into a fully sorted list.",
    sections: [
      { title: "Key steps", items: ["split the array", "sort halves", "merge with two pointers"] },
      { title: "When to use", items: ["stable sorting", "large data sets", "linked list sorting"] },
      { title: "Pitfalls", items: ["extra memory for merging", "recursive overhead"] },
    ],
    complexity: { time: "O(n log n)", space: "O(n)" },
    pythonCode: mergeSortCode,
    exampleProblems: ["Sort List", "Count Inversions", "Merge K Sorted Lists"],
  },
  {
    slug: "quick-sort",
    title: "Quick Sort",
    summary: "Partition around a pivot.",
    kind: "algorithm",
    description: "Quick sort chooses a pivot, partitions the array into smaller and larger values, and recurses on each side.",
    sections: [
      { title: "Key steps", items: ["choose a pivot", "partition elements", "recurse on partitions"] },
      { title: "When to use", items: ["fast average performance", "in-place sorting", "large arrays"] },
      { title: "Pitfalls", items: ["worst-case O(n^2)", "pivot choice matters"] },
    ],
    complexity: { time: "Average O(n log n)", space: "O(log n)" },
    pythonCode: quickSortCode,
    exampleProblems: ["Kth Largest Element", "Sort Colors", "Top K Frequent Elements"],
  },
  {
    slug: "bfs",
    title: "Breadth-First Search",
    summary: "Explore layer by layer with a queue.",
    kind: "algorithm",
    description: "BFS visits nodes in expanding rings from a start node. It uses a queue to ensure the shallowest nodes are processed first.",
    sections: [
      { title: "Key steps", items: ["enqueue start", "dequeue + visit", "enqueue neighbors"] },
      { title: "When to use", items: ["shortest path in unweighted graphs", "level order traversal", "flood fill"] },
      { title: "Pitfalls", items: ["track visited nodes", "queue growth on dense graphs"] },
    ],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: bfsGraphCode,
    exampleProblems: ["Shortest Path in Binary Matrix", "Word Ladder", "Minimum Depth of Tree"],
  },
  {
    slug: "dfs",
    title: "Depth-First Search",
    summary: "Dive deep with a stack.",
    kind: "algorithm",
    description: "DFS explores as far as possible down a branch before backtracking. It uses recursion or an explicit stack.",
    sections: [
      { title: "Key steps", items: ["push start", "pop + visit", "push neighbors"] },
      { title: "When to use", items: ["cycle detection", "topological sorting", "path exploration"] },
      { title: "Pitfalls", items: ["stack depth with recursion", "track visited nodes"] },
    ],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: dfsGraphCode,
    exampleProblems: ["Number of Islands", "Clone Graph", "Path Sum"],
  },
];

const patterns = [
  {
    slug: "sliding-window",
    title: "Sliding Window",
    summary: "Scan contiguous ranges while maintaining a live window invariant.",
    description:
      "Maintain a valid window [l..r] while expanding right and shrinking left as needed. Great for subarray and substring constraints.",
    signals: [
      "Contiguous subarray or substring",
      "Longest/shortest window",
      "Constraint like at most K distinct or sum >= target",
    ],
    invariants: ["Window remains valid after each adjustment", "Only move left when the window is invalid"],
    pitfalls: ["Forgetting to update running sum/map on shrink", "Mixing inclusive and exclusive bounds"],
    complexity: { time: "O(n)", space: "O(1) to O(k)" },
    pythonCode: slidingWindowCode,
    exampleProblems: ["Longest Substring with K Distinct Characters", "Max Sum Subarray of Size K"],
  },
  {
    slug: "two-pointers",
    title: "Two Pointers",
    summary: "Use two indices to converge on a target in a structured array.",
    description:
      "Move pointers from ends or along a sequence to satisfy ordering or sum conditions. Excellent for sorted arrays and partitioning.",
    signals: ["Sorted array or string", "Pair sum / palindrome checks", "Partitioning around a pivot"],
    invariants: ["Pointers move monotonically", "Discard impossible pairs each step"],
    pitfalls: ["Moving the wrong pointer", "Missing equality condition when target is found"],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: twoPointersCode,
    exampleProblems: ["Two Sum II (sorted input)", "Valid Palindrome"],
  },
  {
    slug: "fast-slow-pointers",
    title: "Fast and Slow Pointers",
    summary: "Detect cycles or midpoints with two speeds.",
    description:
      "Advance one pointer twice as fast as the other to detect cycles or locate list midpoints without extra space.",
    signals: ["Linked list or index jumping", "Cycle detection or midpoint"],
    invariants: ["Fast moves 2x slow", "Meeting point implies a cycle"],
    pitfalls: ["Not checking fast and fast.next", "Assuming arrays instead of actual links"],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: fastSlowCode,
    exampleProblems: ["Linked List Cycle", "Middle of the Linked List"],
  },
  {
    slug: "merge-intervals",
    title: "Merge Intervals",
    summary: "Sort and merge overlaps to condense ranges.",
    description:
      "Sort by start time, then merge overlapping ranges to build a compact schedule or coverage list.",
    signals: ["Overlapping ranges", "Scheduling or range compression"],
    invariants: ["Intervals processed in sorted order", "Merged list stays disjoint"],
    pitfalls: ["Skipping sort step", "Not updating end when overlap occurs"],
    complexity: { time: "O(n log n)", space: "O(n)" },
    pythonCode: mergeIntervalsCode,
    exampleProblems: ["Merge Intervals", "Meeting Rooms II"],
  },
  {
    slug: "cyclic-sort",
    title: "Cyclic Sort",
    summary: "Place each value in its correct index by swapping.",
    description:
      "When numbers are in a known range 1..n, swap each number into its target index to sort or spot missing values.",
    signals: ["Values are within 1..n", "Need to find missing/duplicate numbers"],
    invariants: ["Each swap places at least one value correctly"],
    pitfalls: ["Forgetting to increment i only when correct", "Assuming values outside range"],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: cyclicSortCode,
    exampleProblems: ["Cyclic Sort", "Find the Missing Number"],
  },
  {
    slug: "in-place-reversal",
    title: "In-place Reversal of Linked List",
    summary: "Reverse pointers without extra memory.",
    description:
      "Iterate through a list and reverse each pointer with prev/cur/next trackers. Used for reversing full or partial lists.",
    signals: ["Linked list reversal", "Reverse every k nodes"],
    invariants: ["Prev is always the head of the reversed section", "Cur moves forward one node per step"],
    pitfalls: ["Losing the next pointer", "Not returning the new head"],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: reverseListCode,
    exampleProblems: ["Reverse Linked List", "Reverse Nodes in k-Group"],
  },
  {
    slug: "bfs-pattern",
    title: "BFS (Tree/Graph)",
    summary: "Level-order traversal with a queue.",
    description:
      "Breadth-first search explores neighbors layer by layer, producing shortest paths in unweighted graphs.",
    signals: ["Shortest path in unweighted graph", "Level order traversal"],
    invariants: ["Queue holds the frontier", "Visited set prevents cycles"],
    pitfalls: ["Forgetting to mark visited on enqueue", "Revisiting nodes"],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: bfsNeighborsCode,
    exampleProblems: ["Binary Tree Level Order Traversal", "Rotting Oranges"],
  },
  {
    slug: "dfs-pattern",
    title: "DFS (Tree/Graph)",
    summary: "Go deep before you go wide.",
    description:
      "Depth-first search explores as far as possible along each branch before backtracking. Useful for connectivity and paths.",
    signals: ["Need to explore all paths", "Connected components"],
    invariants: ["Stack/recursion keeps current path", "Visited prevents cycles"],
    pitfalls: ["Missing base cases", "Stack overflow on deep recursion"],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: dfsNeighborsCode,
    exampleProblems: ["Number of Islands", "Clone Graph"],
  },
  {
    slug: "two-heaps",
    title: "Two Heaps",
    summary: "Balance two heaps to track medians.",
    description:
      "Maintain a max-heap for the lower half and a min-heap for the upper half to answer running median queries.",
    signals: ["Running median", "Stream of numbers", "Need both min and max halves"],
    invariants: ["Heap sizes differ by at most 1", "All left <= all right"],
    pitfalls: ["Forgetting to rebalance after insert", "Mixing heap order"],
    complexity: { time: "O(log n) per insert", space: "O(n)" },
    pythonCode: twoHeapsCode,
    exampleProblems: ["Find Median from Data Stream", "Sliding Window Median"],
  },
  {
    slug: "subsets",
    title: "Subsets",
    summary: "Generate power sets via BFS/DFS expansion.",
    description:
      "Build up subsets by either branching (DFS) or expanding layer by layer (BFS). Each step doubles the set size.",
    signals: ["Generate all combinations/subsets", "Power set requested"],
    invariants: ["Every element is either included or excluded", "Count doubles per element"],
    pitfalls: ["Mutating subset list in-place", "Missing base empty subset"],
    complexity: { time: "O(2^n)", space: "O(2^n)" },
    pythonCode: subsetsCode,
    exampleProblems: ["Subsets", "Subsets II"],
  },
  {
    slug: "modified-binary-search",
    title: "Modified Binary Search",
    summary: "Binary search with custom conditions.",
    description:
      "Adjust mid checks to find first/last occurrence, rotation pivots, or boundaries like lower/upper bound.",
    signals: ["Sorted array with twist", "Need first/last occurrence"],
    invariants: ["Search space halves each step", "Bounds define candidate range"],
    pitfalls: ["Infinite loops with mid calculation", "Off-by-one in bounds"],
    complexity: { time: "O(log n)", space: "O(1)" },
    pythonCode: modifiedBinarySearchCode,
    exampleProblems: [
      "Find First and Last Position of Element in Sorted Array",
      "Search in Rotated Sorted Array",
    ],
  },
  {
    slug: "top-k-elements",
    title: "Top K Elements",
    summary: "Track the k largest or smallest items with a heap.",
    description: "Maintain a min-heap of size k. Any element larger than the heap root replaces it.",
    signals: ["Top K largest/smallest", "Streaming data"],
    invariants: ["Heap size stays at k", "Heap root is the kth element"],
    pitfalls: ["Using max-heap instead of min-heap", "Not trimming heap"],
    complexity: { time: "O(n log k)", space: "O(k)" },
    pythonCode: topKCode,
    exampleProblems: ["Kth Largest Element in an Array", "Top K Frequent Elements"],
  },
  {
    slug: "k-way-merge",
    title: "K-way Merge",
    summary: "Merge multiple sorted lists using a heap.",
    description:
      "Use a min-heap to always pull the smallest current head among k lists, then push the next from that list.",
    signals: ["Merge multiple sorted sequences", "Need ordered output"],
    invariants: ["Heap contains current heads", "Output grows monotonically"],
    pitfalls: ["Forgetting to push the next item", "Dropping list index info"],
    complexity: { time: "O(n log k)", space: "O(k)" },
    pythonCode: kWayMergeCode,
    exampleProblems: [
      "Merge k Sorted Lists",
      "Smallest Range Covering Elements from K Lists",
    ],
  },
  {
    slug: "knapsack",
    title: "0/1 Knapsack (DP)",
    summary: "Choose items without exceeding capacity.",
    description:
      "Dynamic programming tracks the best value for each capacity as items are considered once.",
    signals: ["Pick items with weights/values", "Capacity constraint"],
    invariants: ["Each item used at most once", "DP row builds on previous row"],
    pitfalls: ["Iterating capacity forward (causes reuse)", "Not initializing base row"],
    complexity: { time: "O(n * C)", space: "O(C)" },
    pythonCode: knapsackCode,
    exampleProblems: ["Partition Equal Subset Sum", "0/1 Knapsack"],
  },
  {
    slug: "topological-sort",
    title: "Topological Sort",
    summary: "Order tasks in a DAG by dependencies.",
    description:
      "Compute in-degrees, enqueue zero-degree nodes, and peel them off to get a valid ordering.",
    signals: ["Dependencies in a DAG", "Need valid ordering"],
    invariants: ["Queue always holds zero in-degree nodes", "Edges removed exactly once"],
    pitfalls: ["Missing cycles", "Not updating in-degrees"],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: topoSortCode,
    exampleProblems: ["Course Schedule II", "Alien Dictionary"],
  },
  {
    slug: "backtracking",
    title: "Backtracking",
    summary: "Explore choices with undo to satisfy constraints.",
    description:
      "Build candidates incrementally and backtrack whenever a constraint fails or a solution is found.",
    signals: ["Search all combinations with pruning", "Need to explore permutations"],
    invariants: ["Path holds current decision", "Undo after exploring a branch"],
    pitfalls: ["Missing prune checks", "Mutating shared state without undo"],
    complexity: { time: "O(b^d)", space: "O(d)" },
    pythonCode: backtrackingCode,
    exampleProblems: ["N-Queens", "Sudoku Solver"],
  },
];

const baseTraversalGraph = {
  nodes: ["A", "B", "C", "D", "E", "F", "G"],
  edges: {
    A: ["B", "C"],
    B: ["D", "E"],
    C: ["F"],
    D: ["G"],
    E: ["G"],
    F: ["G"],
    G: [],
  },
  start: "A",
};

const bfsSimulation = {
  title: "BFS traversal",
  description: "Visit nodes level by level using a queue.",
  type: "graph-traversal",
  mode: "bfs",
  ...baseTraversalGraph,
};

const dfsSimulation = {
  title: "DFS traversal",
  description: "Go deep before backtracking with a stack.",
  type: "graph-traversal",
  mode: "dfs",
  ...baseTraversalGraph,
};

const simulationConfigs = {
  "sliding-window": {
    title: "Max sum window",
    description: "Find the max sum of any 3 consecutive elements.",
    type: "sliding-window",
    array: [2, 1, 5, 1, 3, 2, 9, 7],
    windowSize: 3,
  },
  "two-pointers": {
    title: "Two sum (sorted)",
    description: "Move pointers inward to hit the target sum.",
    type: "two-pointers",
    array: [1, 2, 3, 4, 6, 8, 11],
    target: 10,
  },
  "modified-binary-search": {
    title: "Binary search",
    description: "Halve the search range until the target is found.",
    type: "binary-search",
    array: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    target: 6,
  },
  "cyclic-sort": {
    title: "Cyclic swap",
    description: "Swap each value into its correct index.",
    type: "cyclic-sort",
    array: [3, 1, 5, 4, 2],
  },
  "merge-sort": {
    title: "Merge sort",
    description: "Split, then merge sorted halves.",
    type: "merge-sort",
    array: [7, 2, 5, 3, 1, 6, 4],
  },
  "quick-sort": {
    title: "Quick sort",
    description: "Partition around a pivot and recurse.",
    type: "quick-sort",
    array: [7, 2, 5, 3, 1, 6, 4],
  },
  "bfs-pattern": bfsSimulation,
  "dfs-pattern": dfsSimulation,
  "topological-sort": {
    title: "Topological order",
    description: "Peel off zero in-degree nodes to form a valid ordering.",
    type: "topological-sort",
    nodes: ["A", "B", "C", "D", "E", "F"],
    edges: [
      ["A", "D"],
      ["A", "E"],
      ["B", "E"],
      ["C", "F"],
      ["D", "F"],
      ["E", "F"],
    ],
  },
  "two-heaps": {
    title: "Streaming median",
    description: "Balance two heaps as numbers arrive.",
    type: "two-heaps",
    stream: [5, 2, 10, 3, 8, 1],
  },
  "linear-search": {
    title: "Linear scan",
    description: "Check each value until the target is found.",
    type: "linear-search",
    array: [4, 1, 7, 3, 9, 2],
    target: 9,
  },
  "binary-search": {
    title: "Binary search",
    description: "Halve the search range until the target is found.",
    type: "binary-search",
    array: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    target: 6,
  },
  bfs: bfsSimulation,
  dfs: dfsSimulation,
};

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;");

const renderList = (items) => items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

const renderSectionBlock = (title, items) => `
  <div>
    <h5>${escapeHtml(title)}</h5>
    <ul class="card-list">${renderList(items)}</ul>
  </div>
`;

const renderMeta = (complexity) => `
  <div class="card-meta">
    <div>
      <strong>Time</strong>
      <span>${escapeHtml(complexity.time)}</span>
    </div>
    <div>
      <strong>Space</strong>
      <span>${escapeHtml(complexity.space)}</span>
    </div>
  </div>
`;

const renderCodeBlock = (code) => `
  <details class="code-block">
    <summary>Python sketch</summary>
    <pre><code>${escapeHtml(code)}</code></pre>
  </details>
`;

const renderExamples = (examples) => `
  <div class="card-examples">
    <strong>Example problems</strong>
    <ul>${renderList(examples)}</ul>
  </div>
`;

const renderSimulationPanel = (item) => {
  const config = simulationConfigs[item.slug];
  if (!config) {
    return "";
  }

  return `
    <section class="simulator" data-sim="${escapeHtml(item.slug)}">
      <div class="simulator-header">
        <div>
          <p class="eyebrow">Simulation</p>
          <h4>${escapeHtml(config.title)}</h4>
          <p class="simulator-subtitle">${escapeHtml(config.description)}</p>
        </div>
        <div class="simulator-controls">
          <button class="sim-button ghost" type="button" data-sim-action="reset">Reset</button>
          <button class="sim-button" type="button" data-sim-action="step">Step</button>
          <button class="sim-button accent" type="button" data-sim-action="play">Play</button>
        </div>
      </div>
      <div class="simulator-track" data-sim-track></div>
      <div class="simulator-state" data-sim-state></div>
    </section>
  `;
};

const renderFoundationCard = (item) => {
  const sectionsHtml = item.sections.map((section) => renderSectionBlock(section.title, section.items)).join("");
  const codeHtml = item.pythonCode ? renderCodeBlock(item.pythonCode) : "";
  const examplesHtml = item.exampleProblems?.length ? renderExamples(item.exampleProblems) : "";

  return `
    <details class="card" data-slug="${escapeHtml(item.slug)}">
      <summary>
        <h4>${escapeHtml(item.title)}</h4>
        <p>${escapeHtml(item.summary)}</p>
        <div class="card-summary-tags">
          <span class="tag accent">${item.kind === "algorithm" ? "Algorithm" : "Data structure"}</span>
        </div>
      </summary>
      <div class="card-body">
        <p>${escapeHtml(item.description)}</p>
        ${sectionsHtml}
        ${renderMeta(item.complexity)}
        ${codeHtml}
        ${examplesHtml}
      </div>
    </details>
  `;
};

const renderFoundationListItem = (item) => `
  <button class="card pattern-card" type="button" data-slug="${escapeHtml(item.slug)}">
    <div class="pattern-card-body">
      <h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.summary)}</p>
      <div class="card-summary-tags">
        <span class="tag accent">${item.kind === "algorithm" ? "Algorithm" : "Data structure"}</span>
      </div>
    </div>
  </button>
`;

const renderFoundationDetail = (item) => {
  const sectionsHtml = item.sections.map((section) => renderSectionBlock(section.title, section.items)).join("");
  const codeHtml = item.pythonCode ? renderCodeBlock(item.pythonCode) : "";
  const examplesHtml = item.exampleProblems?.length ? renderExamples(item.exampleProblems) : "";
  const simulationHtml = item.kind === "algorithm" ? renderSimulationPanel(item) : "";

  return `
    <div class="side-panel-header">
      <p class="eyebrow">${item.kind === "algorithm" ? "Algorithm" : "Data structure"}</p>
      <h3>${escapeHtml(item.title)}</h3>
      <p class="side-panel-summary">${escapeHtml(item.summary)}</p>
    </div>
    ${simulationHtml}
    <div class="side-panel-body">
      <p>${escapeHtml(item.description)}</p>
      ${sectionsHtml}
      ${renderMeta(item.complexity)}
      ${codeHtml}
      ${examplesHtml}
    </div>
  `;
};

const renderPatternCard = (item) => {
  const signalsHtml = item.signals?.length ? renderSectionBlock("Signals", item.signals) : "";
  const invariantsHtml = item.invariants?.length ? renderSectionBlock("Invariants", item.invariants) : "";
  const pitfallsHtml = item.pitfalls?.length ? renderSectionBlock("Pitfalls", item.pitfalls) : "";
  const codeHtml = item.pythonCode ? renderCodeBlock(item.pythonCode) : "";
  const examplesHtml = item.exampleProblems?.length ? renderExamples(item.exampleProblems) : "";

  return `
    <details class="card" data-slug="${escapeHtml(item.slug)}">
      <summary>
        <h4>${escapeHtml(item.title)}</h4>
        <p>${escapeHtml(item.summary)}</p>
        <div class="card-summary-tags">
          <span class="tag accent">Pattern</span>
        </div>
      </summary>
      <div class="card-body">
        <p>${escapeHtml(item.description)}</p>
        ${signalsHtml}
        ${invariantsHtml}
        ${pitfallsHtml}
        ${renderMeta(item.complexity)}
        ${codeHtml}
        ${examplesHtml}
      </div>
    </details>
  `;
};

const renderPatternPreviewCard = (item) => `
  <article class="card preview-card">
    <div class="preview-card-body">
      <h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.summary)}</p>
      <div class="card-summary-tags">
        <span class="tag accent">Pattern</span>
        <span class="tag">${escapeHtml(item.complexity.time)}</span>
      </div>
      <a class="preview-link" href="patterns.html">View details</a>
    </div>
  </article>
`;

const renderPatternListItem = (item) => `
  <button class="card pattern-card" type="button" data-slug="${escapeHtml(item.slug)}">
    <div class="pattern-card-body">
      <h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.summary)}</p>
      <div class="card-summary-tags">
        <span class="tag accent">Pattern</span>
      </div>
    </div>
  </button>
`;

const renderPatternDetail = (item) => {
  const signalsHtml = item.signals?.length ? renderSectionBlock("Signals", item.signals) : "";
  const invariantsHtml = item.invariants?.length ? renderSectionBlock("Invariants", item.invariants) : "";
  const pitfallsHtml = item.pitfalls?.length ? renderSectionBlock("Pitfalls", item.pitfalls) : "";
  const codeHtml = item.pythonCode ? renderCodeBlock(item.pythonCode) : "";
  const examplesHtml = item.exampleProblems?.length ? renderExamples(item.exampleProblems) : "";

  return `
    <div class="side-panel-header">
      <p class="eyebrow">Pattern</p>
      <h3>${escapeHtml(item.title)}</h3>
      <p class="side-panel-summary">${escapeHtml(item.summary)}</p>
    </div>
    ${renderSimulationPanel(item)}
    <div class="side-panel-body">
      <p>${escapeHtml(item.description)}</p>
      ${signalsHtml}
      ${invariantsHtml}
      ${pitfallsHtml}
      ${renderMeta(item.complexity)}
      ${codeHtml}
      ${examplesHtml}
    </div>
  `;
};

let activeSimulation = null;

const buildSlidingWindowSteps = (config) => {
  const steps = [];
  const values = config.array;
  const windowSize = config.windowSize;
  let left = 0;
  let windowSum = 0;
  let maxSum = Number.NEGATIVE_INFINITY;

  for (let right = 0; right < values.length; right += 1) {
    windowSum += values[right];
    if (right - left + 1 > windowSize) {
      windowSum -= values[left];
      left += 1;
    }
    if (right - left + 1 === windowSize) {
      maxSum = Math.max(maxSum, windowSum);
    }
    steps.push({ left, right, windowSum, maxSum, windowSize });
  }

  return steps;
};

const buildTwoPointersSteps = (config) => {
  const values = config.array;
  const target = config.target;
  const steps = [];
  let left = 0;
  let right = values.length - 1;

  while (left <= right) {
    const sum = values[left] + values[right];
    const found = sum === target && left !== right;
    steps.push({ left, right, sum, target, found });
    if (found) {
      break;
    }
    if (sum < target) {
      left += 1;
    } else {
      right -= 1;
    }
  }

  return steps;
};

const buildBinarySearchSteps = (config) => {
  const values = config.array;
  const target = config.target;
  const steps = [];
  let left = 0;
  let right = values.length - 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const value = values[mid];
    const found = value === target;
    const comparison = found ? "found" : value < target ? "go right" : "go left";
    steps.push({ left, right, mid, value, target, found, comparison });
    if (found) {
      break;
    }
    if (value < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return steps;
};

const buildCyclicSortSteps = (config) => {
  const steps = [];
  const values = [...config.array];
  let index = 0;

  while (index < values.length) {
    const targetIndex = values[index] - 1;
    if (values[index] !== values[targetIndex]) {
      const swapped = [...values];
      const temp = swapped[index];
      swapped[index] = swapped[targetIndex];
      swapped[targetIndex] = temp;
      steps.push({ array: swapped, index, swapIndex: targetIndex, action: "swap" });
      values[index] = swapped[index];
      values[targetIndex] = swapped[targetIndex];
    } else {
      steps.push({ array: [...values], index, swapIndex: null, action: "advance" });
      index += 1;
    }
  }

  return steps;
};

const buildMergeSortSteps = (config) => {
  const steps = [];
  const values = [...config.array];
  const aux = [...values];

  const merge = (start, mid, end) => {
    let left = start;
    let right = mid;
    let index = start;

    while (left < mid && right < end) {
      if (values[left] <= values[right]) {
        aux[index] = values[left];
        left += 1;
      } else {
        aux[index] = values[right];
        right += 1;
      }
      index += 1;
    }

    while (left < mid) {
      aux[index] = values[left];
      left += 1;
      index += 1;
    }

    while (right < end) {
      aux[index] = values[right];
      right += 1;
      index += 1;
    }

    for (let i = start; i < end; i += 1) {
      values[i] = aux[i];
    }

    steps.push({
      array: [...values],
      start,
      end: end - 1,
      mid,
      action: "merge",
    });
  };

  const sort = (start, end) => {
    if (end - start <= 1) {
      return;
    }
    const mid = Math.floor((start + end) / 2);
    sort(start, mid);
    sort(mid, end);
    merge(start, mid, end);
  };

  sort(0, values.length);
  return steps;
};

const buildQuickSortSteps = (config) => {
  const steps = [];
  const values = [...config.array];

  const partition = (start, end) => {
    const pivotValue = values[end];
    let i = start;

    for (let j = start; j < end; j += 1) {
      if (values[j] <= pivotValue) {
        const temp = values[i];
        values[i] = values[j];
        values[j] = temp;
        i += 1;
      }
    }

    const temp = values[i];
    values[i] = values[end];
    values[end] = temp;

    steps.push({
      array: [...values],
      start,
      end,
      pivot: i,
      pivotValue,
      action: "partition",
    });

    return i;
  };

  const sort = (start, end) => {
    if (start >= end) {
      return;
    }
    const pivotIndex = partition(start, end);
    sort(start, pivotIndex - 1);
    sort(pivotIndex + 1, end);
  };

  sort(0, values.length - 1);
  return steps;
};

const buildLinearSearchSteps = (config) => {
  const values = config.array;
  const target = config.target;
  const steps = [];

  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const found = value === target;
    steps.push({ index, value, target, found });
    if (found) {
      break;
    }
  }

  return steps;
};

const buildGraphTraversalSteps = (config) => {
  const steps = [];
  const visited = new Set();
  const frontier = [config.start];
  const order = [];

  while (frontier.length > 0) {
    const current = config.mode === "bfs" ? frontier.shift() : frontier.pop();
    if (!current || visited.has(current)) {
      continue;
    }
    visited.add(current);
    order.push(current);
    const neighbors = config.edges[current] || [];
    const nextNodes = neighbors.filter((node) => !visited.has(node));

    if (config.mode === "bfs") {
      frontier.push(...nextNodes);
    } else {
      frontier.push(...nextNodes.slice().reverse());
    }

    steps.push({
      current,
      visited: Array.from(visited),
      frontier: [...frontier],
      order: [...order],
    });
  }

  return steps;
};

const buildTopologicalSortSteps = (config) => {
  const steps = [];
  const nodes = config.nodes;
  const edges = config.edges;
  const inDegree = {};
  const adjacency = {};

  nodes.forEach((node) => {
    inDegree[node] = 0;
    adjacency[node] = [];
  });

  edges.forEach(([from, to]) => {
    adjacency[from].push(to);
    inDegree[to] += 1;
  });

  const queue = nodes.filter((node) => inDegree[node] === 0);
  const order = [];

  while (queue.length > 0) {
    const current = queue.shift();
    order.push(current);

    adjacency[current].forEach((neighbor) => {
      inDegree[neighbor] -= 1;
      if (inDegree[neighbor] === 0) {
        queue.push(neighbor);
      }
    });

    steps.push({
      current,
      queue: [...queue],
      order: [...order],
      inDegree: { ...inDegree },
    });
  }

  return steps;
};

const buildTwoHeapsSteps = (config) => {
  const steps = [];
  const left = [];
  const right = [];

  const maxOf = (arr) => (arr.length ? Math.max(...arr) : null);
  const minOf = (arr) => (arr.length ? Math.min(...arr) : null);

  config.stream.forEach((value) => {
    const leftMax = maxOf(left);
    if (leftMax === null || value <= leftMax) {
      left.push(value);
    } else {
      right.push(value);
    }

    if (left.length > right.length + 1) {
      const move = maxOf(left);
      left.splice(left.indexOf(move), 1);
      right.push(move);
    } else if (right.length > left.length) {
      const move = minOf(right);
      right.splice(right.indexOf(move), 1);
      left.push(move);
    }

    const leftSorted = [...left].sort((a, b) => b - a);
    const rightSorted = [...right].sort((a, b) => a - b);
    const median =
      leftSorted.length === rightSorted.length
        ? (leftSorted[0] + rightSorted[0]) / 2
        : leftSorted[0];

    steps.push({
      value,
      left: leftSorted,
      right: rightSorted,
      median,
    });
  });

  return steps;
};

const buildSimulationSteps = (config) => {
  switch (config.type) {
    case "sliding-window":
      return buildSlidingWindowSteps(config);
    case "two-pointers":
      return buildTwoPointersSteps(config);
    case "binary-search":
      return buildBinarySearchSteps(config);
    case "cyclic-sort":
      return buildCyclicSortSteps(config);
    case "merge-sort":
      return buildMergeSortSteps(config);
    case "quick-sort":
      return buildQuickSortSteps(config);
    case "linear-search":
      return buildLinearSearchSteps(config);
    case "graph-traversal":
      return buildGraphTraversalSteps(config);
    case "topological-sort":
      return buildTopologicalSortSteps(config);
    case "two-heaps":
      return buildTwoHeapsSteps(config);
    default:
      return [];
  }
};

const getPointerLabel = (labels) => {
  if (!labels.length) {
    return { text: "", className: "" };
  }

  const priority = ["is-pivot", "is-mid", "is-right", "is-left", "is-index", "is-swap"];
  const available = new Set(labels.map((label) => label.className));
  const className = priority.find((name) => available.has(name)) || "";
  const text = labels.map((label) => label.text).join("");
  return { text, className };
};

const renderSimulationTrack = (track, config, step) => {
  if (config.type === "graph-traversal") {
    const nodesHtml = config.nodes
      .map((node) => {
        const classes = ["sim-node"];
        if (step.visited.includes(node)) {
          classes.push("is-visited");
        }
        if (step.frontier.includes(node)) {
          classes.push("is-frontier");
        }
        if (node === step.current) {
          classes.push("is-active");
        }

        return `
          <div class="sim-node-wrap">
            <div class="${classes.join(" ")}">${escapeHtml(node)}</div>
          </div>
        `;
      })
      .join("");

    track.innerHTML = `<div class="sim-graph">${nodesHtml}</div>`;
    return;
  }

  if (config.type === "topological-sort") {
    const nodesHtml = config.nodes
      .map((node) => {
        const classes = ["sim-node"];
        if (step.order.includes(node)) {
          classes.push("is-visited");
        }
        if (step.queue.includes(node)) {
          classes.push("is-frontier");
        }
        if (node === step.current) {
          classes.push("is-active");
        }

        return `
          <div class="sim-node-wrap">
            <div class="${classes.join(" ")}">${escapeHtml(node)}</div>
            <span class="sim-degree">deg ${step.inDegree[node]}</span>
          </div>
        `;
      })
      .join("");

    track.innerHTML = `<div class="sim-graph">${nodesHtml}</div>`;
    return;
  }

  if (config.type === "two-heaps") {
    const leftHtml = step.left.length
      ? step.left.map((value) => `<span class="sim-heap-value">${escapeHtml(value)}</span>`).join("")
      : `<span class="sim-heap-empty">Empty</span>`;
    const rightHtml = step.right.length
      ? step.right.map((value) => `<span class="sim-heap-value">${escapeHtml(value)}</span>`).join("")
      : `<span class="sim-heap-empty">Empty</span>`;

    track.innerHTML = `
      <div class="sim-heaps">
        <div class="sim-heap">
          <h5>Max heap</h5>
          <div class="sim-heap-values">${leftHtml}</div>
        </div>
        <div class="sim-heap">
          <h5>Min heap</h5>
          <div class="sim-heap-values">${rightHtml}</div>
        </div>
      </div>
    `;
    return;
  }

  const values = step.array ? step.array : config.array;
  const itemsHtml = values
    .map((value, index) => {
      const classes = ["sim-cell"];
      const labels = [];

      if (config.type === "sliding-window") {
        const inWindow = index >= step.left && index <= step.right;
        if (inWindow) {
          classes.push("is-window");
        }
        if (index === step.left) {
          labels.push({ text: "L", className: "is-left" });
        }
        if (index === step.right) {
          classes.push("is-active");
          labels.push({ text: "R", className: "is-right" });
        }
      }

      if (config.type === "two-pointers") {
        if (index === step.left) {
          classes.push("is-left");
          labels.push({ text: "L", className: "is-left" });
        }
        if (index === step.right) {
          classes.push("is-right");
          labels.push({ text: "R", className: "is-right" });
        }
      }

      if (config.type === "binary-search") {
        if (index === step.left) {
          classes.push("is-left");
          labels.push({ text: "L", className: "is-left" });
        }
        if (index === step.right) {
          classes.push("is-right");
          labels.push({ text: "R", className: "is-right" });
        }
        if (index === step.mid) {
          classes.push("is-active", "is-mid");
          labels.push({ text: "M", className: "is-mid" });
        }
      }

      if (config.type === "cyclic-sort") {
        if (index === step.index) {
          classes.push("is-index");
          labels.push({ text: "I", className: "is-index" });
        }
        if (step.swapIndex !== null && index === step.swapIndex) {
          classes.push("is-swap");
          labels.push({ text: "J", className: "is-swap" });
        }
      }

      if (config.type === "linear-search") {
        if (index === step.index) {
          classes.push("is-active");
          labels.push({ text: "I", className: "is-index" });
        }
      }

      if (config.type === "merge-sort") {
        if (index >= step.start && index <= step.end) {
          classes.push("is-window");
        }
        if (index === step.start) {
          labels.push({ text: "L", className: "is-left" });
        }
        if (index === step.end) {
          labels.push({ text: "R", className: "is-right" });
        }
        if (index === step.mid) {
          labels.push({ text: "M", className: "is-mid" });
        }
      }

      if (config.type === "quick-sort") {
        if (index >= step.start && index <= step.end) {
          classes.push("is-window");
        }
        if (index === step.start) {
          labels.push({ text: "L", className: "is-left" });
        }
        if (index === step.end) {
          labels.push({ text: "R", className: "is-right" });
        }
        if (index === step.pivot) {
          classes.push("is-active", "is-pivot");
          labels.push({ text: "P", className: "is-pivot" });
        }
      }

      const pointerLabel = getPointerLabel(labels);
      const pointerHtml = pointerLabel.text
        ? `<span class="sim-pointer ${pointerLabel.className}">${pointerLabel.text}</span>`
        : "";

      return `
        <div class="sim-item">
          ${pointerHtml}
          <div class="${classes.join(" ")}">${escapeHtml(value)}</div>
          <span class="sim-index">${index}</span>
        </div>
      `;
    })
    .join("");

  track.innerHTML = `<div class="sim-array">${itemsHtml}</div>`;
};

const renderSimulationState = (state, config, step, stepIndex, totalSteps) => {
  const baseItems = [
    { label: "Step", value: `${stepIndex + 1} / ${totalSteps}` },
  ];

  if (config.type === "sliding-window") {
    const windowSize = step.right - step.left + 1;
    const maxSum = step.maxSum === Number.NEGATIVE_INFINITY ? "-" : String(step.maxSum);
    baseItems.push(
      { label: "Window", value: `[${step.left}, ${step.right}]` },
      { label: "Window size", value: `${windowSize} / ${config.windowSize}` },
      { label: "Window sum", value: String(step.windowSum) },
      { label: "Best sum", value: maxSum }
    );
  }

  if (config.type === "two-pointers") {
    const status = step.found
      ? "Match found"
      : stepIndex === totalSteps - 1
        ? "No match"
        : "Searching";
    baseItems.push(
      { label: "Left", value: String(step.left) },
      { label: "Right", value: String(step.right) },
      { label: "Sum", value: String(step.sum) },
      { label: "Target", value: String(step.target) },
      { label: "Status", value: status }
    );
  }

  if (config.type === "binary-search") {
    const status = step.found
      ? "Found"
      : stepIndex === totalSteps - 1
        ? "Not found"
        : "Searching";
    baseItems.push(
      { label: "Range", value: `[${step.left}, ${step.right}]` },
      { label: "Mid", value: String(step.mid) },
      { label: "Mid value", value: String(step.value) },
      { label: "Target", value: String(step.target) },
      { label: "Decision", value: step.comparison },
      { label: "Status", value: status }
    );
  }

  if (config.type === "cyclic-sort") {
    const targetIndex = step.swapIndex === null ? "-" : String(step.swapIndex);
    baseItems.push(
      { label: "Index", value: String(step.index) },
      { label: "Swap target", value: targetIndex },
      { label: "Action", value: step.action }
    );
  }

  if (config.type === "merge-sort") {
    baseItems.push(
      { label: "Range", value: `[${step.start}, ${step.end}]` },
      { label: "Mid", value: String(step.mid) },
      { label: "Action", value: step.action }
    );
  }

  if (config.type === "quick-sort") {
    baseItems.push(
      { label: "Range", value: `[${step.start}, ${step.end}]` },
      { label: "Pivot index", value: String(step.pivot) },
      { label: "Pivot value", value: String(step.pivotValue) },
      { label: "Action", value: step.action }
    );
  }

  if (config.type === "linear-search") {
    const status = step.found ? "Found" : stepIndex === totalSteps - 1 ? "Not found" : "Scanning";
    baseItems.push(
      { label: "Index", value: String(step.index) },
      { label: "Value", value: String(step.value) },
      { label: "Target", value: String(step.target) },
      { label: "Status", value: status }
    );
  }

  if (config.type === "graph-traversal") {
    const frontierLabel = config.mode === "bfs" ? "Queue" : "Stack";
    baseItems.push(
      { label: "Mode", value: config.mode.toUpperCase() },
      { label: "Current", value: step.current },
      { label: frontierLabel, value: step.frontier.join(" -> ") || "-" },
      { label: "Visited", value: step.order.join(" -> ") || "-" }
    );
  }

  if (config.type === "topological-sort") {
    baseItems.push(
      { label: "Current", value: step.current },
      { label: "Ready", value: step.queue.join(", ") || "-" },
      { label: "Order", value: step.order.join(" -> ") || "-" }
    );
  }

  if (config.type === "two-heaps") {
    baseItems.push(
      { label: "Inserted", value: String(step.value) },
      { label: "Median", value: String(step.median) },
      { label: "Left size", value: String(step.left.length) },
      { label: "Right size", value: String(step.right.length) }
    );
  }

  state.innerHTML = baseItems
    .map(
      (item) => `
        <div class="sim-state-item">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(item.value)}</span>
        </div>
      `
    )
    .join("");
};

const createSimulationRunner = (config, container) => {
  const track = container.querySelector("[data-sim-track]");
  const state = container.querySelector("[data-sim-state]");
  const playButton = container.querySelector('[data-sim-action="play"]');
  const stepButton = container.querySelector('[data-sim-action="step"]');
  const resetButton = container.querySelector('[data-sim-action="reset"]');

  if (!track || !state || !playButton || !stepButton || !resetButton) {
    return null;
  }

  const steps = buildSimulationSteps(config);
  if (!steps.length) {
    return null;
  }
  let stepIndex = 0;
  let timer = null;

  const updatePlayLabel = (isPlaying) => {
    playButton.textContent = isPlaying ? "Pause" : "Play";
  };

  const render = () => {
    const current = steps[stepIndex];
    renderSimulationTrack(track, config, current);
    renderSimulationState(state, config, current, stepIndex, steps.length);
  };

  const stop = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    updatePlayLabel(false);
  };

  const next = () => {
    if (stepIndex >= steps.length - 1) {
      stop();
      return;
    }
    stepIndex += 1;
    render();
    if (stepIndex >= steps.length - 1) {
      stop();
    }
  };

  const start = () => {
    if (timer) {
      stop();
      return;
    }
    updatePlayLabel(true);
    timer = setInterval(next, 900);
  };

  const reset = () => {
    stop();
    stepIndex = 0;
    render();
  };

  playButton.addEventListener("click", start);
  stepButton.addEventListener("click", () => {
    stop();
    next();
  });
  resetButton.addEventListener("click", reset);

  render();

  return { stop };
};

const setupSimulation = (item, detailPane) => {
  if (activeSimulation) {
    activeSimulation.stop();
    activeSimulation = null;
  }

  if (!item || !detailPane) {
    return;
  }

  const config = simulationConfigs[item.slug];
  if (!config) {
    return;
  }

  const container = detailPane.querySelector(`[data-sim="${item.slug}"]`);
  if (!container) {
    return;
  }

  activeSimulation = createSimulationRunner(config, container);
};

const buildSearchText = (item) => {
  const parts = [item.title, item.summary, item.description];
  if (item.sections) {
    item.sections.forEach((section) => {
      parts.push(section.title);
      parts.push(...section.items);
    });
  }
  if (item.signals) {
    parts.push(...item.signals);
  }
  if (item.invariants) {
    parts.push(...item.invariants);
  }
  if (item.pitfalls) {
    parts.push(...item.pitfalls);
  }
  if (item.exampleProblems) {
    parts.push(...item.exampleProblems);
  }
  if (item.complexity) {
    parts.push(item.complexity.time, item.complexity.space);
  }
  return parts.join(" ").toLowerCase();
};

const renderCards = (items, renderer, selector, limit) => {
  const grid = document.querySelector(selector);
  if (!grid) {
    return [];
  }
  const slice = limit ? items.slice(0, limit) : items;
  grid.innerHTML = slice.map(renderer).join("");
  return Array.from(grid.querySelectorAll(".card"));
};

const setupSearch = (items, cards, onFilter) => {
  const searchInput = document.querySelector("#search-input");
  const resultsCount = document.querySelector("#results-count");
  if (!searchInput || cards.length === 0) {
    return;
  }

  const itemBySlug = new Map(items.map((item) => [item.slug, item]));

  cards.forEach((card) => {
    const slug = card.dataset.slug;
    if (!slug) {
      return;
    }
    const item = itemBySlug.get(slug);
    if (!item) {
      return;
    }
    card.dataset.search = buildSearchText(item);
  });

  const updateFilter = () => {
    const term = searchInput.value.trim().toLowerCase();
    let visible = 0;

    cards.forEach((card) => {
      const haystack = card.dataset.search || "";
      const match = !term || haystack.includes(term);
      card.classList.toggle("is-hidden", !match);
      if (match) {
        visible += 1;
      }
    });

    if (resultsCount) {
      if (term) {
        resultsCount.textContent = `${visible} matches`;
      } else {
        resultsCount.textContent = `${cards.length} entries`;
      }
    }

    if (onFilter) {
      onFilter({ term, visible, cards });
    }
  };

  searchInput.addEventListener("input", updateFilter);
  updateFilter();
};

const setupDetailList = ({
  items,
  listSelector,
  detailSelector,
  listRenderer,
  detailRenderer,
  emptyState,
  onSelect,
}) => {
  const detailPane = document.querySelector(detailSelector);
  const cards = renderCards(items, listRenderer, listSelector);
  const itemBySlug = new Map(items.map((item) => [item.slug, item]));

  const showEmptyState = () => {
    if (!detailPane || !emptyState) {
      return;
    }
    detailPane.innerHTML = `
      <div class="side-panel-empty">
        <h3>${emptyState.title}</h3>
        <p>${emptyState.body}</p>
      </div>
    `;
    if (onSelect) {
      onSelect({ item: null, detailPane, card: null });
    }
  };

  const selectItem = (card) => {
    const slug = card?.dataset.slug;
    const item = slug ? itemBySlug.get(slug) : null;
    if (!item || !detailPane) {
      return;
    }

    cards.forEach((entry) => {
      entry.classList.toggle("is-selected", entry === card);
      entry.setAttribute("aria-pressed", entry === card ? "true" : "false");
    });

    detailPane.innerHTML = detailRenderer(item);
    if (onSelect) {
      onSelect({ item, detailPane, card });
    }
  };

  const selectFirstVisible = () => {
    const active = cards.find(
      (card) => card.classList.contains("is-selected") && !card.classList.contains("is-hidden")
    );
    if (active) {
      return;
    }
    const first = cards.find((card) => !card.classList.contains("is-hidden"));
    if (first) {
      selectItem(first);
      return;
    }
    showEmptyState();
  };

  cards.forEach((card) => {
    card.addEventListener("click", () => selectItem(card));
  });

  setupSearch(items, cards, selectFirstVisible);
  if (cards.length > 0) {
    selectItem(cards[0]);
  } else {
    showEmptyState();
  }
};

const page = document.body.dataset.page || "index";

if (page === "index") {
  const structuresCount = document.querySelector("#structures-count");
  const algorithmsCount = document.querySelector("#algorithms-count");
  const patternsCount = document.querySelector("#patterns-count");

  if (structuresCount) {
    structuresCount.textContent = String(dataStructures.length);
  }

  if (algorithmsCount) {
    algorithmsCount.textContent = String(algorithms.length);
  }

  if (patternsCount) {
    patternsCount.textContent = String(patterns.length);
  }

  renderCards(dataStructures, renderFoundationCard, "#structures-preview", 4);
  renderCards(algorithms, renderFoundationCard, "#algorithms-preview", 4);
  renderCards(patterns, renderPatternPreviewCard, "#patterns-preview", 4);
}

if (page === "data-structures") {
  setupDetailList({
    items: dataStructures,
    listSelector: "#structure-list",
    detailSelector: "#structure-detail",
    listRenderer: renderFoundationListItem,
    detailRenderer: renderFoundationDetail,
    emptyState: {
      title: "No data structures found",
      body: "Try a different search term to see matching data structures.",
    },
  });
}

if (page === "algorithms") {
  setupDetailList({
    items: algorithms,
    listSelector: "#algorithm-list",
    detailSelector: "#algorithm-detail",
    listRenderer: renderFoundationListItem,
    detailRenderer: renderFoundationDetail,
    emptyState: {
      title: "No algorithms found",
      body: "Try a different search term to see matching algorithms.",
    },
    onSelect: ({ item, detailPane }) => {
      setupSimulation(item, detailPane);
    },
  });
}

if (page === "patterns") {
  setupDetailList({
    items: patterns,
    listSelector: "#pattern-list",
    detailSelector: "#pattern-detail",
    listRenderer: renderPatternListItem,
    detailRenderer: renderPatternDetail,
    emptyState: {
      title: "No patterns found",
      body: "Try a different search term to see matching patterns.",
    },
    onSelect: ({ item, detailPane }) => {
      setupSimulation(item, detailPane);
    },
  });
}
