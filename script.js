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

const adjacencyMatrixCode = `def add_edge(matrix, u, v):
    matrix[u][v] = 1
    matrix[v][u] = 1

def has_edge(matrix, u, v):
    return matrix[u][v] == 1`;

const edgeListCode = `def add_edge(edges, u, v):
    edges.append((u, v))

def has_edge(edges, u, v):
    return (u, v) in edges or (v, u) in edges`;

const doublyLinkedListCode = `class Node:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

def insert_after(node, value):
    new_node = Node(value, node, node.next)
    if node.next:
        node.next.prev = new_node
    node.next = new_node
    return new_node`;

const trieCode = `class TrieNode:
    def __init__(self):
        self.children = {}
        self.terminal = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.terminal = True

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.terminal`;

const avlTreeCode = `class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

def height(node):
    return node.height if node else 0

def rotate_left(z):
    y = z.right
    t2 = y.left
    y.left = z
    z.right = t2
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def rotate_right(z):
    y = z.left
    t3 = y.right
    y.right = z
    z.left = t3
    z.height = 1 + max(height(z.left), height(z.right))
    y.height = 1 + max(height(y.left), height(y.right))
    return y

def insert(root, val):
    if not root:
        return Node(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    root.height = 1 + max(height(root.left), height(root.right))
    balance = height(root.left) - height(root.right)
    if balance > 1 and val < root.left.val:
        return rotate_right(root)
    if balance < -1 and val > root.right.val:
        return rotate_left(root)
    if balance > 1 and val > root.left.val:
        root.left = rotate_left(root.left)
        return rotate_right(root)
    if balance < -1 and val < root.right.val:
        root.right = rotate_right(root.right)
        return rotate_left(root)
    return root`;

const redBlackTreeCode = `RED = True
BLACK = False

class Node:
    def __init__(self, val, color=RED):
        self.val = val
        self.color = color
        self.left = None
        self.right = None

def rotate_left(root):
    x = root.right
    root.right = x.left
    x.left = root
    return x

def rotate_right(root):
    x = root.left
    root.left = x.right
    x.right = root
    return x

def insert(root, val):
    if not root:
        return Node(val, color=BLACK)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    # Fix red-black invariants with rotations + recoloring.
    return root`;

const bTreeCode = `class BTreeNode:
    def __init__(self, keys=None, children=None, leaf=True):
        self.keys = keys or []
        self.children = children or []
        self.leaf = leaf

def search(node, key):
    i = 0
    while i < len(node.keys) and key > node.keys[i]:
        i += 1
    if i < len(node.keys) and node.keys[i] == key:
        return True
    if node.leaf:
        return False
    return search(node.children[i], key)`;

const segmentTreeCode = `def build(nums):
    n = len(nums)
    tree = [0] * (2 * n)
    for i in range(n):
        tree[n + i] = nums[i]
    for i in range(n - 1, 0, -1):
        tree[i] = tree[i * 2] + tree[i * 2 + 1]
    return tree

def update(tree, index, value):
    n = len(tree) // 2
    i = index + n
    tree[i] = value
    while i > 1:
        i //= 2
        tree[i] = tree[i * 2] + tree[i * 2 + 1]

def query(tree, left, right):
    n = len(tree) // 2
    left += n
    right += n
    total = 0
    while left <= right:
        if left % 2 == 1:
            total += tree[left]
            left += 1
        if right % 2 == 0:
            total += tree[right]
            right -= 1
        left //= 2
        right //= 2
    return total`;

const fenwickTreeCode = `class Fenwick:
    def __init__(self, n):
        self.tree = [0] * (n + 1)

    def add(self, index, delta):
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, index):
        i = index + 1
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total`;

const skipListCode = `class SkipNode:
    def __init__(self, val, next_nodes):
        self.val = val
        self.next = next_nodes

class SkipList:
    def __init__(self):
        self.head = SkipNode(None, [None])

    def search(self, target):
        node = self.head
        for level in reversed(range(len(node.next))):
            while node.next[level] and node.next[level].val < target:
                node = node.next[level]
        node = node.next[0]
        return node is not None and node.val == target`;

const bloomFilterCode = `class BloomFilter:
    def __init__(self, size, hashes):
        self.bits = [0] * size
        self.hashes = hashes

    def add(self, value):
        for fn in self.hashes:
            self.bits[fn(value) % len(self.bits)] = 1

    def contains(self, value):
        return all(self.bits[fn(value) % len(self.bits)] for fn in self.hashes)`;

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

const dynamicProgrammingCode = `def fib_memo(n, memo=None):
    if memo is None:
        memo = {0: 0, 1: 1}
    if n in memo:
        return memo[n]
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def make_change(amount, coins):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for total in range(coin, amount + 1):
            dp[total] = min(dp[total], dp[total - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1`;

const fibonacciCode = `def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b`;

const coinChangeCode = `def coin_change(amount, coins):
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for total in range(coin, amount + 1):
            dp[total] = min(dp[total], dp[total - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1`;

const squareSubmatrixCode = `def largest_square(matrix):
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    best = 0

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                if r == 0 or c == 0:
                    dp[r][c] = 1
                else:
                    dp[r][c] = 1 + min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1])
                best = max(best, dp[r][c])

    return best`;

const targetSumCode = `def target_sum(nums, target):
    total = sum(nums)
    if (total + target) % 2 != 0:
        return 0
    subset = (total + target) // 2
    dp = [0] * (subset + 1)
    dp[0] = 1
    for num in nums:
        for s in range(subset, num - 1, -1):
            dp[s] += dp[s - num]
    return dp[subset]`;

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

const dijkstraCode = `import heapq

def dijkstra(graph, start):
    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist[node]:
            continue
        for nxt, weight in graph[node]:
            new_dist = cur_dist + weight
            if new_dist < dist[nxt]:
                dist[nxt] = new_dist
                heapq.heappush(heap, (new_dist, nxt))

    return dist`;

const bellmanFordCode = `def bellman_ford(nodes, edges, start):
    dist = {n: float("inf") for n in nodes}
    dist[start] = 0

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] == float("inf"):
                continue
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break

    return dist`;

const floydWarshallCode = `def floyd_warshall(nodes, edges):
    dist = {u: {v: float("inf") for v in nodes} for u in nodes}
    for n in nodes:
        dist[n][n] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w

    for k in nodes:
        for i in nodes:
            for j in nodes:
                alt = dist[i][k] + dist[k][j]
                if alt < dist[i][j]:
                    dist[i][j] = alt

    return dist`;

const aStarCode = `import heapq

def a_star(start, goal, neighbors, heuristic):
    frontier = [(heuristic(start), 0, start, None)]
    best = {start: 0}
    came_from = {}

    while frontier:
        _, cost, node, parent = heapq.heappop(frontier)
        if node in came_from:
            continue
        came_from[node] = parent
        if node == goal:
            break
        for nxt, weight in neighbors(node):
            new_cost = cost + weight
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                priority = new_cost + heuristic(nxt)
                heapq.heappush(frontier, (priority, new_cost, nxt, node))

    return came_from`;

const unionFindCode = `class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True`;

const kruskalCode = `class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def kruskal(n, edges):
    edges.sort(key=lambda x: x[2])
    dsu = DSU(n)
    total = 0
    mst = []

    for u, v, w in edges:
        if dsu.union(u, v):
            total += w
            mst.append((u, v, w))

    return total, mst`;

const primCode = `import heapq

def prim(graph, start):
    seen = {start}
    heap = []
    for v, w in graph[start]:
        heapq.heappush(heap, (w, start, v))

    total = 0
    mst = []

    while heap:
        w, u, v = heapq.heappop(heap)
        if v in seen:
            continue
        seen.add(v)
        total += w
        mst.append((u, v, w))
        for nxt, weight in graph[v]:
            if nxt not in seen:
                heapq.heappush(heap, (weight, v, nxt))

    return total, mst`;

const kosarajuCode = `def kosaraju(nodes, edges):
    graph = {n: [] for n in nodes}
    rev = {n: [] for n in nodes}
    for u, v in edges:
        graph[u].append(v)
        rev[v].append(u)

    seen = set()
    order = []

    def dfs(u):
        seen.add(u)
        for v in graph[u]:
            if v not in seen:
                dfs(v)
        order.append(u)

    for n in nodes:
        if n not in seen:
            dfs(n)

    comps = []
    seen.clear()

    def dfs_rev(u, comp):
        seen.add(u)
        comp.append(u)
        for v in rev[u]:
            if v not in seen:
                dfs_rev(v, comp)

    for n in reversed(order):
        if n not in seen:
            comp = []
            dfs_rev(n, comp)
            comps.append(comp)

    return comps`;

const bridgesArticulationCode = `def bridges_and_articulation(graph):
    timer = 0
    disc = {}
    low = {}
    parent = {}
    bridges = []
    points = set()

    def dfs(u):
        nonlocal timer
        timer += 1
        disc[u] = low[u] = timer
        child_count = 0

        for v in graph[u]:
            if v not in disc:
                parent[v] = u
                child_count += 1
                dfs(v)
                low[u] = min(low[u], low[v])

                if low[v] > disc[u]:
                    bridges.append((u, v))
                if parent.get(u) is None and child_count > 1:
                    points.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    points.add(u)
            elif parent.get(u) != v:
                low[u] = min(low[u], disc[v])

    for u in graph:
        if u not in disc:
            parent[u] = None
            dfs(u)

    return bridges, points`;

const eulerianPathCode = `def eulerian_path(graph, start):
    stack = [start]
    path = []
    local = {u: list(vs) for u, vs in graph.items()}

    while stack:
        node = stack[-1]
        if local[node]:
            stack.append(local[node].pop())
        else:
            path.append(stack.pop())

    return path[::-1]`;

const edmondsKarpCode = `from collections import deque

def edmonds_karp(n, edges, s, t):
    graph = [[] for _ in range(n)]
    cap = {}

    for u, v, c in edges:
        graph[u].append(v)
        graph[v].append(u)
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap[(v, u)] = cap.get((v, u), 0)

    flow = 0

    while True:
        parent = [-1] * n
        parent[s] = s
        q = deque([s])

        while q and parent[t] == -1:
            u = q.popleft()
            for v in graph[u]:
                if parent[v] == -1 and cap[(u, v)] > 0:
                    parent[v] = u
                    q.append(v)

        if parent[t] == -1:
            break

        bottleneck = float("inf")
        v = t
        while v != s:
            u = parent[v]
            bottleneck = min(bottleneck, cap[(u, v)])
            v = u

        v = t
        while v != s:
            u = parent[v]
            cap[(u, v)] -= bottleneck
            cap[(v, u)] += bottleneck
            v = u

        flow += bottleneck

    return flow`;

const bipartiteMatchingCode = `def max_bipartite_matching(left_nodes, graph):
    match_right = {}

    def dfs(u, seen):
        for v in graph.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_right or dfs(match_right[v], seen):
                match_right[v] = u
                return True
        return False

    matched = 0
    for u in left_nodes:
        if dfs(u, set()):
            matched += 1

    return matched`;

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
    operations: {
      read: { label: "O(1)", rank: 1, note: "index" },
      write: { label: "O(n)", rank: 4, note: "insert/delete" },
      search: { label: "O(n)", rank: 4, note: "unsorted" },
    },
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
    operations: {
      read: { label: "O(n)", rank: 4, note: "by index" },
      write: { label: "O(1)", rank: 1, note: "after locate" },
      search: { label: "O(n)", rank: 4, note: "scan" },
    },
    pythonCode: linkedListCode,
    exampleProblems: ["Reverse Linked List", "Merge Two Lists", "Detect Cycle"],
  },
  {
    slug: "doubly-linked-list",
    title: "Doubly Linked Lists",
    summary: "Linked nodes with prev/next pointers.",
    kind: "data-structure",
    description:
      "Doubly linked lists allow traversal in both directions and make deletions O(1) once you have the node.",
    sections: [
      {
        title: "Core operations",
        items: ["insert/delete O(1) with node", "append O(1) with tail", "bidirectional traversal"],
      },
      {
        title: "When to use",
        items: ["LRU cache lists", "frequent middle deletes", "back/forward navigation"],
      },
      {
        title: "Pitfalls",
        items: ["extra pointer per node", "pointer updates are error-prone", "no random access"],
      },
    ],
    complexity: { time: "Search O(n), insert/delete O(1) with node", space: "O(n)" },
    operations: {
      read: { label: "O(n)", rank: 4, note: "by index" },
      write: { label: "O(1)", rank: 1, note: "with node" },
      search: { label: "O(n)", rank: 4, note: "scan" },
    },
    pythonCode: doublyLinkedListCode,
    exampleProblems: ["LRU Cache", "Browser History", "Design Linked List"],
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
    operations: {
      read: { label: "O(1)", rank: 1, note: "peek" },
      write: { label: "O(1)", rank: 1, note: "push/pop" },
      search: { label: "O(n)", rank: 4, note: "scan" },
    },
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
    operations: {
      read: { label: "O(1)", rank: 1, note: "front" },
      write: { label: "O(1)", rank: 1, note: "enqueue/dequeue" },
      search: { label: "O(n)", rank: 4, note: "scan" },
    },
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
    operations: {
      read: { label: "O(1)", rank: 1, note: "avg" },
      write: { label: "O(1)", rank: 1, note: "avg" },
      search: { label: "O(1)", rank: 1, note: "avg" },
    },
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
    operations: {
      read: { label: "O(log n)", rank: 2, note: "balanced" },
      write: { label: "O(log n)", rank: 2, note: "balanced" },
      search: { label: "O(log n)", rank: 2, note: "balanced" },
    },
    pythonCode: treeCode,
    exampleProblems: ["Validate BST", "Kth Smallest in BST", "Lowest Common Ancestor"],
  },
  {
    slug: "avl-tree",
    title: "AVL Trees",
    summary: "Strictly balanced binary search trees.",
    kind: "data-structure",
    description:
      "AVL trees keep heights tightly balanced, guaranteeing log-time operations with more rotations on updates.",
    sections: [
      {
        title: "Core operations",
        items: ["search/insert/delete O(log n)", "rotate to rebalance", "track heights"],
      },
      {
        title: "When to use",
        items: ["read-heavy ordered sets", "range queries", "consistent log-time lookups"],
      },
      {
        title: "Pitfalls",
        items: ["more rotations on writes", "extra height bookkeeping"],
      },
    ],
    complexity: { time: "Search/insert/delete O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "balanced" },
      write: { label: "O(log n)", rank: 2, note: "rotations" },
      search: { label: "O(log n)", rank: 2, note: "balanced" },
    },
    pythonCode: avlTreeCode,
    exampleProblems: ["My Calendar I", "Kth Smallest in BST", "Range Module"],
  },
  {
    slug: "red-black-tree",
    title: "Red-Black Trees",
    summary: "Balanced BSTs with relaxed height rules.",
    kind: "data-structure",
    description:
      "Red-black trees keep balance using color rules, offering log-time operations with fewer rotations than AVL.",
    sections: [
      {
        title: "Core operations",
        items: ["search/insert/delete O(log n)", "recolor + rotate on insert", "black-height invariants"],
      },
      {
        title: "When to use",
        items: ["ordered maps", "tree-based sets", "standard library TreeMap/TreeSet"],
      },
      {
        title: "Pitfalls",
        items: ["complex rebalancing rules", "harder to implement correctly"],
      },
    ],
    complexity: { time: "Search/insert/delete O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "balanced" },
      write: { label: "O(log n)", rank: 2, note: "recolor" },
      search: { label: "O(log n)", rank: 2, note: "balanced" },
    },
    pythonCode: redBlackTreeCode,
    exampleProblems: ["My Calendar II", "Range Module", "Ordered Map"],
  },
  {
    slug: "b-tree",
    title: "B-Trees",
    summary: "Multiway balanced trees for storage engines.",
    kind: "data-structure",
    description:
      "B-trees store many keys per node to keep the tree shallow, making them ideal for disk and database indexes.",
    sections: [
      {
        title: "Core operations",
        items: ["search/insert/delete O(log n)", "split/merge nodes", "high fanout"],
      },
      {
        title: "When to use",
        items: ["database indexes", "filesystems", "large ordered data sets"],
      },
      {
        title: "Pitfalls",
        items: ["complex rebalancing", "tuning node size for hardware"],
      },
    ],
    complexity: { time: "Search/insert/delete O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "high fanout" },
      write: { label: "O(log n)", rank: 2, note: "splits" },
      search: { label: "O(log n)", rank: 2, note: "high fanout" },
    },
    pythonCode: bTreeCode,
    exampleProblems: ["Database Index Scan", "Filesystem Lookup", "Large Range Queries"],
  },
  {
    slug: "trie",
    title: "Tries (Prefix Trees)",
    summary: "Prefix indexing over characters.",
    kind: "data-structure",
    description:
      "Tries store characters along a path so prefix searches take time proportional to word length, not the number of words.",
    sections: [
      {
        title: "Core operations",
        items: ["insert/search O(k)", "prefix queries", "store words by character"],
      },
      {
        title: "When to use",
        items: ["autocomplete", "dictionary lookups", "prefix filtering"],
      },
      {
        title: "Pitfalls",
        items: ["high memory usage", "large alphabets add overhead"],
      },
    ],
    complexity: { time: "Insert/search O(k)", space: "O(total characters)" },
    operations: {
      read: { label: "O(k)", rank: 3, note: "k = key length" },
      write: { label: "O(k)", rank: 3, note: "k = key length" },
      search: { label: "O(k)", rank: 3, note: "k = key length" },
    },
    pythonCode: trieCode,
    exampleProblems: ["Implement Trie", "Word Search II", "Replace Words"],
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
    operations: {
      read: { label: "O(1)", rank: 1, note: "peek" },
      write: { label: "O(log n)", rank: 2, note: "insert" },
      search: { label: "O(n)", rank: 4, note: "scan" },
    },
    pythonCode: heapCode,
    exampleProblems: ["Kth Largest Element", "Merge K Sorted Lists", "Task Scheduler"],
  },
  {
    slug: "segment-tree",
    title: "Segment Trees",
    summary: "Range queries with log-time updates.",
    kind: "data-structure",
    description:
      "Segment trees store aggregate values over ranges, enabling fast range sum/min queries with point updates.",
    sections: [
      {
        title: "Core operations",
        items: ["range query O(log n)", "point update O(log n)", "build O(n)"],
      },
      {
        title: "When to use",
        items: ["dynamic range sums", "range minimum/maximum", "offline query batches"],
      },
      {
        title: "Pitfalls",
        items: ["extra memory overhead", "careful index boundaries"],
      },
    ],
    complexity: { time: "Query/update O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "range query" },
      write: { label: "O(log n)", rank: 2, note: "point update" },
      search: { label: "O(log n)", rank: 2, note: "range query" },
    },
    pythonCode: segmentTreeCode,
    exampleProblems: ["Range Sum Query", "Range Minimum Query", "Dynamic RMQ"],
  },
  {
    slug: "fenwick-tree",
    title: "Fenwick Trees (BIT)",
    summary: "Compact structure for prefix sums.",
    kind: "data-structure",
    description:
      "Fenwick trees store cumulative frequencies and support fast prefix sums and point updates with low constants.",
    sections: [
      {
        title: "Core operations",
        items: ["prefix sum O(log n)", "point update O(log n)", "space-efficient array layout"],
      },
      {
        title: "When to use",
        items: ["prefix sums", "inversion counts", "lightweight range sums"],
      },
      {
        title: "Pitfalls",
        items: ["harder to support range updates", "indexing is 1-based internally"],
      },
    ],
    complexity: { time: "Update/query O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "prefix sum" },
      write: { label: "O(log n)", rank: 2, note: "point update" },
      search: { label: "O(log n)", rank: 2, note: "prefix search" },
    },
    pythonCode: fenwickTreeCode,
    exampleProblems: ["Range Sum Query (mutable)", "Count of Smaller Numbers", "Inversion Count"],
  },
  {
    slug: "skip-list",
    title: "Skip Lists",
    summary: "Layered lists with probabilistic balance.",
    kind: "data-structure",
    description:
      "Skip lists add express lanes to linked lists, giving average log-time search and updates without rotations.",
    sections: [
      {
        title: "Core operations",
        items: ["search/insert/delete O(log n) avg", "multiple forward pointers", "randomized levels"],
      },
      {
        title: "When to use",
        items: ["ordered sets", "simpler balanced alternatives", "concurrent maps"],
      },
      {
        title: "Pitfalls",
        items: ["probabilistic performance", "extra pointers per node"],
      },
    ],
    complexity: { time: "Average O(log n)", space: "O(n)" },
    operations: {
      read: { label: "O(log n)", rank: 2, note: "avg" },
      write: { label: "O(log n)", rank: 2, note: "avg" },
      search: { label: "O(log n)", rank: 2, note: "avg" },
    },
    pythonCode: skipListCode,
    exampleProblems: ["Design Skiplist", "Ordered Set", "Range Queries"],
  },
  {
    slug: "disjoint-set",
    title: "Disjoint Set (Union-Find)",
    summary: "Track components under unions.",
    kind: "data-structure",
    description:
      "Disjoint sets support near-constant time connectivity checks using path compression and union by rank.",
    sections: [
      {
        title: "Core operations",
        items: ["find with path compression", "union by rank/size", "check connectivity"],
      },
      {
        title: "When to use",
        items: ["dynamic connectivity", "Kruskal MST", "grouping merges"],
      },
      {
        title: "Pitfalls",
        items: ["not designed for deletions", "needs careful initialization"],
      },
    ],
    complexity: { time: "Amortized O(alpha(n))", space: "O(n)" },
    operations: {
      read: { label: "O(alpha(n))", rank: 1, note: "amortized" },
      write: { label: "O(alpha(n))", rank: 1, note: "amortized" },
      search: { label: "O(alpha(n))", rank: 1, note: "connectivity" },
    },
    pythonCode: unionFindCode,
    exampleProblems: ["Number of Provinces", "Accounts Merge", "Kruskal MST"],
  },
  {
    slug: "bloom-filter",
    title: "Bloom Filters",
    summary: "Probabilistic set membership.",
    kind: "data-structure",
    description:
      "Bloom filters trade false positives for compact memory. Membership checks are fast but can say yes incorrectly.",
    sections: [
      {
        title: "Core operations",
        items: ["insert O(k)", "membership test O(k)", "no false negatives"],
      },
      {
        title: "When to use",
        items: ["cache filters", "duplicate detection", "large-scale existence checks"],
      },
      {
        title: "Pitfalls",
        items: ["false positives", "deletions require counting filters"],
      },
    ],
    complexity: { time: "Insert/check O(k)", space: "O(m) bits" },
    operations: {
      read: { label: "O(k)", rank: 3, note: "k hashes" },
      write: { label: "O(k)", rank: 3, note: "k hashes" },
      search: { label: "O(k)", rank: 3, note: "false positive" },
    },
    pythonCode: bloomFilterCode,
    exampleProblems: ["Cache Filtering", "Duplicate URL Check", "Set Membership"],
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
    operations: {
      read: { label: "O(1)", rank: 1, note: "neighbors" },
      write: { label: "O(1)", rank: 1, note: "add edge" },
      search: { label: "O(V+E)", rank: 5, note: "traverse" },
    },
    pythonCode: graphCode,
    exampleProblems: ["Course Schedule", "Clone Graph", "Connected Components"],
  },
  {
    slug: "adjacency-list",
    title: "Adjacency Lists",
    summary: "Store neighbors per node in lists.",
    kind: "data-structure",
    description:
      "Adjacency lists keep a list of neighbors for each vertex, making them ideal for sparse graphs.",
    sections: [
      {
        title: "Core operations",
        items: ["iterate neighbors fast", "add edge O(1)", "edge lookup O(deg v)"],
      },
      {
        title: "When to use",
        items: ["sparse graphs", "most traversals", "memory efficiency"],
      },
      {
        title: "Pitfalls",
        items: ["edge lookup can be linear in degree", "duplicate edges if not checked"],
      },
    ],
    complexity: { time: "Edge add O(1), lookup O(deg v)", space: "O(V + E)" },
    operations: {
      read: { label: "O(deg v)", rank: 3, note: "neighbors" },
      write: { label: "O(1)", rank: 1, note: "add edge" },
      search: { label: "O(deg v)", rank: 3, note: "edge lookup" },
    },
    pythonCode: graphCode,
    exampleProblems: ["Sparse Network", "Graph Traversal", "Topological Sort"],
  },
  {
    slug: "adjacency-matrix",
    title: "Adjacency Matrices",
    summary: "Matrix of edge connections.",
    kind: "data-structure",
    description:
      "Adjacency matrices use a V x V grid so edge checks are constant-time, at the cost of O(V^2) space.",
    sections: [
      {
        title: "Core operations",
        items: ["edge lookup O(1)", "toggle edge O(1)", "iterate neighbors O(V)"],
      },
      {
        title: "When to use",
        items: ["dense graphs", "fast edge checks", "small fixed-size graphs"],
      },
      {
        title: "Pitfalls",
        items: ["high memory cost", "slow neighbor iteration for sparse graphs"],
      },
    ],
    complexity: { time: "Edge lookup O(1), iterate O(V)", space: "O(V^2)" },
    operations: {
      read: { label: "O(1)", rank: 1, note: "edge check" },
      write: { label: "O(1)", rank: 1, note: "toggle" },
      search: { label: "O(1)", rank: 1, note: "edge lookup" },
    },
    pythonCode: adjacencyMatrixCode,
    exampleProblems: ["Dense Network", "Transitive Closure", "Graph Matrix Ops"],
  },
  {
    slug: "edge-list",
    title: "Edge Lists",
    summary: "Store each edge as a pair or triple.",
    kind: "data-structure",
    description:
      "Edge lists keep a flat list of edges, making them simple to build and easy to sort by weight.",
    sections: [
      {
        title: "Core operations",
        items: ["append edge O(1)", "edge lookup O(E)", "sort edges by weight"],
      },
      {
        title: "When to use",
        items: ["MST algorithms", "edge-centric processing", "streaming edges"],
      },
      {
        title: "Pitfalls",
        items: ["slow edge lookup", "neighbor iteration requires scans"],
      },
    ],
    complexity: { time: "Edge lookup O(E), append O(1)", space: "O(E)" },
    operations: {
      read: { label: "O(E)", rank: 4, note: "scan" },
      write: { label: "O(1)", rank: 1, note: "append" },
      search: { label: "O(E)", rank: 4, note: "edge lookup" },
    },
    pythonCode: edgeListCode,
    exampleProblems: ["Kruskal MST", "Edge Sorting", "Streamed Graphs"],
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
  {
    slug: "topological-sort",
    title: "Topological Sort",
    summary: "Order a DAG by dependencies.",
    kind: "algorithm",
    description:
      "Topological sort orders nodes so every directed edge goes from earlier to later. It is built from in-degrees or DFS finish times.",
    sections: [
      {
        title: "Key steps",
        items: ["compute in-degrees", "enqueue zero in-degree nodes", "peel nodes in order"],
      },
      {
        title: "When to use",
        items: ["dependency ordering", "DAG scheduling", "cycle detection in DAGs"],
      },
      {
        title: "Pitfalls",
        items: ["graph must be a DAG", "missing zero in-degree nodes in disconnected DAGs"],
      },
    ],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: topoSortCode,
    exampleProblems: ["Course Schedule II", "Alien Dictionary", "Build Order"],
  },
  {
    slug: "dijkstra",
    title: "Dijkstra's Algorithm",
    summary: "Shortest paths with non-negative weights.",
    kind: "algorithm",
    description:
      "Dijkstra repeatedly picks the closest unsettled node and relaxes its outgoing edges using a min-heap.",
    sections: [
      { title: "Key steps", items: ["init distances", "pop closest node", "relax edges + update heap"] },
      { title: "When to use", items: ["non-negative weights", "single-source shortest paths"] },
      { title: "Pitfalls", items: ["negative weights break correctness", "skip stale heap entries"] },
    ],
    complexity: { time: "O((V + E) log V)", space: "O(V + E)" },
    pythonCode: dijkstraCode,
    exampleProblems: ["Network Delay Time", "Path With Minimum Effort", "Swim in Rising Water"],
  },
  {
    slug: "bellman-ford",
    title: "Bellman-Ford",
    summary: "Shortest paths with negative edges.",
    kind: "algorithm",
    description:
      "Bellman-Ford relaxes all edges V-1 times and can detect negative cycles with an extra pass.",
    sections: [
      { title: "Key steps", items: ["initialize distances", "relax all edges V-1 times", "check for cycles"] },
      { title: "When to use", items: ["negative weights", "cycle detection", "small graphs"] },
      { title: "Pitfalls", items: ["slow for large graphs", "unreachable nodes stay inf"] },
    ],
    complexity: { time: "O(V * E)", space: "O(V)" },
    pythonCode: bellmanFordCode,
    exampleProblems: ["Cheapest Flights Within K Stops", "Detect Negative Cycle", "Currency Arbitrage"],
  },
  {
    slug: "floyd-warshall",
    title: "Floyd-Warshall",
    summary: "All-pairs shortest paths.",
    kind: "algorithm",
    description:
      "Dynamic programming over intermediate nodes updates a distance matrix for every pair of vertices.",
    sections: [
      { title: "Key steps", items: ["init distance matrix", "try each intermediate k", "update all pairs"] },
      { title: "When to use", items: ["many shortest-path queries", "dense graphs"] },
      { title: "Pitfalls", items: ["O(V^3) time is steep", "O(V^2) memory"] },
    ],
    complexity: { time: "O(V^3)", space: "O(V^2)" },
    pythonCode: floydWarshallCode,
    exampleProblems: ["Find the City With the Smallest Number of Neighbors", "All-Pairs Shortest Path"],
  },
  {
    slug: "a-star",
    title: "A* Search",
    summary: "Heuristic-guided shortest path.",
    kind: "algorithm",
    description:
      "A* uses cost-so-far plus a heuristic to prioritize nodes that appear closest to the goal.",
    sections: [
      { title: "Key steps", items: ["push start with heuristic", "pop lowest f = g + h", "relax neighbors"] },
      { title: "When to use", items: ["grid or map pathfinding", "admissible heuristics available"] },
      { title: "Pitfalls", items: ["inadmissible heuristic breaks optimality", "track visited states carefully"] },
    ],
    complexity: { time: "O(E log V) typical", space: "O(V)" },
    pythonCode: aStarCode,
    exampleProblems: ["Shortest Path in Grid With Obstacles", "Navigation on Maps"],
  },
  {
    slug: "union-find",
    title: "Union-Find (Disjoint Set Union)",
    summary: "Track connected components under unions.",
    kind: "algorithm",
    description:
      "Union-find supports near-constant time find/union with path compression and union by rank.",
    sections: [
      { title: "Key steps", items: ["find root with compression", "union by rank/size", "check connectivity"] },
      { title: "When to use", items: ["connectivity queries", "cycle detection", "Kruskal's MST"] },
      { title: "Pitfalls", items: ["forgetting compression", "not normalizing indices"] },
    ],
    complexity: { time: "Amortized O(alpha(n))", space: "O(n)" },
    pythonCode: unionFindCode,
    exampleProblems: ["Number of Provinces", "Redundant Connection", "Accounts Merge"],
  },
  {
    slug: "kruskal",
    title: "Kruskal's MST",
    summary: "Build an MST by sorting edges.",
    kind: "algorithm",
    description:
      "Kruskal adds edges in ascending weight order, skipping those that form cycles via union-find.",
    sections: [
      { title: "Key steps", items: ["sort edges by weight", "union endpoints", "skip cycles"] },
      { title: "When to use", items: ["sparse graphs", "edge list available"] },
      { title: "Pitfalls", items: ["disconnected graph yields a forest", "need union-find"] },
    ],
    complexity: { time: "O(E log E)", space: "O(V)" },
    pythonCode: kruskalCode,
    exampleProblems: ["Minimum Spanning Tree", "Connecting Cities With Minimum Cost", "Min Cost to Connect Points"],
  },
  {
    slug: "prim",
    title: "Prim's MST",
    summary: "Grow an MST from a start node.",
    kind: "algorithm",
    description:
      "Prim expands the tree by always taking the cheapest edge that connects a new node to the MST.",
    sections: [
      { title: "Key steps", items: ["start from any node", "push outgoing edges", "take cheapest edge"] },
      { title: "When to use", items: ["dense graphs", "adjacency list with weights"] },
      { title: "Pitfalls", items: ["forgetting to skip visited nodes", "need a min-heap"] },
    ],
    complexity: { time: "O(E log V)", space: "O(V + E)" },
    pythonCode: primCode,
    exampleProblems: ["Minimum Spanning Tree", "Optimize Water Distribution", "Min Cost to Connect Points"],
  },
  {
    slug: "strongly-connected-components",
    title: "Strongly Connected Components (Kosaraju)",
    summary: "Group mutually reachable nodes in a directed graph.",
    kind: "algorithm",
    description:
      "Kosaraju orders nodes by finish time, then runs DFS on the reversed graph to extract components.",
    sections: [
      { title: "Key steps", items: ["DFS for finish order", "reverse edges", "DFS in reverse order"] },
      { title: "When to use", items: ["directed graph condensation", "2-SAT", "cycle grouping"] },
      { title: "Pitfalls", items: ["forgetting to reverse edges", "reusing seen set"] },
    ],
    complexity: { time: "O(V + E)", space: "O(V + E)" },
    pythonCode: kosarajuCode,
    exampleProblems: ["Strongly Connected Components", "2-SAT", "Find Cycles in Directed Graph"],
  },
  {
    slug: "bridges-articulation",
    title: "Bridges & Articulation Points",
    summary: "Find critical edges and vertices.",
    kind: "algorithm",
    description:
      "DFS with discovery and low-link values identifies edges or vertices whose removal disconnects the graph.",
    sections: [
      { title: "Key steps", items: ["DFS with timestamps", "track low-link values", "check bridge/vertex rules"] },
      { title: "When to use", items: ["network reliability", "critical connections", "graph vulnerability"] },
      { title: "Pitfalls", items: ["root articulation special case", "undirected graph only"] },
    ],
    complexity: { time: "O(V + E)", space: "O(V)" },
    pythonCode: bridgesArticulationCode,
    exampleProblems: ["Critical Connections in a Network", "Articulation Points", "Bridge Edges"],
  },
  {
    slug: "eulerian-path",
    title: "Eulerian Path/Circuit",
    summary: "Traverse every edge exactly once.",
    kind: "algorithm",
    description:
      "Hierholzer's algorithm constructs an Eulerian path or circuit after verifying degree conditions.",
    sections: [
      { title: "Key steps", items: ["check degree conditions", "walk edges with stack", "splice cycles"] },
      { title: "When to use", items: ["route planning", "trail reconstruction", "edge-visit constraints"] },
      { title: "Pitfalls", items: ["directed vs undirected rules", "graph must be connected on edges"] },
    ],
    complexity: { time: "O(V + E)", space: "O(V + E)" },
    pythonCode: eulerianPathCode,
    exampleProblems: ["Reconstruct Itinerary", "Valid Arrangement of Pairs", "Eulerian Path"],
  },
  {
    slug: "max-flow",
    title: "Max Flow (Edmonds-Karp)",
    summary: "Find the maximum flow in a network.",
    kind: "algorithm",
    description:
      "Edmonds-Karp repeatedly finds shortest augmenting paths with BFS in the residual graph.",
    sections: [
      { title: "Key steps", items: ["build residual graph", "BFS for augmenting path", "update capacities"] },
      { title: "When to use", items: ["capacity constraints", "min cut", "flow networks"] },
      { title: "Pitfalls", items: ["slow on dense graphs", "need residual edges both ways"] },
    ],
    complexity: { time: "O(V * E^2)", space: "O(V + E)" },
    pythonCode: edmondsKarpCode,
    exampleProblems: ["Maximum Flow", "Min Cut", "Project Selection"],
  },
  {
    slug: "bipartite-matching",
    title: "Bipartite Matching (Kuhn)",
    summary: "Match nodes across two partitions.",
    kind: "algorithm",
    description:
      "Find augmenting paths from each left node to increase the size of the matching.",
    sections: [
      { title: "Key steps", items: ["try each left node", "DFS to find augmenting path", "flip matches"] },
      { title: "When to use", items: ["assignment problems", "pairing constraints", "scheduling"] },
      { title: "Pitfalls", items: ["reset visited per DFS", "ensure graph is bipartite"] },
    ],
    complexity: { time: "O(V * E)", space: "O(V + E)" },
    pythonCode: bipartiteMatchingCode,
    exampleProblems: ["Maximum Bipartite Matching", "Assign Tasks to Workers", "Minimum Vertex Cover"],
  },
  {
    slug: "dynamic-programming",
    title: "Dynamic Programming (FAST Method)",
    summary: "Store repeated work; build solutions with the FAST method.",
    kind: "algorithm",
    description:
      "Dynamic programming stores repeated computations to avoid recomputation, trading space for time. The FAST method is a repeatable way to move from a brute-force recursive solution to an optimal DP solution.",
    sections: [
      {
        title: "FAST: First solution",
        items: [
          "write the brute-force recursive solution",
          "ignore efficiency; aim for correctness",
          "keep recursive calls self-contained (no globals)",
          "avoid tail recursion; combine results after calls",
          "pass only necessary parameters",
        ],
      },
      {
        title: "FAST: Analyze",
        items: [
          "compute time and space complexity",
          "confirm optimal substructure",
          "check for overlapping subproblems (try a medium input)",
        ],
      },
      {
        title: "FAST: Find subproblems",
        items: [
          "define the meaning of each subproblem",
          "memoize overlapping results",
          "top-down solutions clarify the state",
        ],
      },
      {
        title: "FAST: Turn the solution around",
        items: [
          "convert to bottom-up iteration",
          "build from base cases",
          "compute successive subproblems to the target",
        ],
      },
      {
        title: "Key terms",
        items: [
          "recursion fundamentals first",
          "top-down = recursion + memoization",
          "bottom-up = iterative tabulation",
          "overlapping subproblems + optimal substructure",
        ],
      },
    ],
    complexity: { time: "Depends on states and transitions", space: "O(states)" },
    pythonCode: dynamicProgrammingCode,
    exampleProblems: [
      "Fibonacci Numbers",
      "Making Change",
      "Square Submatrix",
      "0-1 Knapsack",
      "Target Sum",
    ],
  },
  {
    slug: "fibonacci",
    title: "Fibonacci Numbers",
    summary: "Build a sequence from two previous values.",
    kind: "algorithm",
    description:
      "Each term is the sum of the two before it. Dynamic programming turns the recursive definition into an iterative linear pass.",
    sections: [
      { title: "Key steps", items: ["use base cases for 0 and 1", "iterate from 2..n", "carry two variables"] },
      { title: "When to use", items: ["warm-up DP", "recurrence practice", "similar linear recurrences"] },
      { title: "Pitfalls", items: ["off-by-one on n", "overflow for large n"] },
    ],
    complexity: { time: "O(n)", space: "O(1)" },
    pythonCode: fibonacciCode,
    exampleProblems: ["Fibonacci Number", "Climbing Stairs", "House Robber"],
  },
  {
    slug: "coin-change",
    title: "Coin Change (Min Coins)",
    summary: "Find the fewest coins to reach a target amount.",
    kind: "algorithm",
    description:
      "Use a 1D DP array where dp[a] is the minimum coins to make amount a, relaxing each coin across the amounts.",
    sections: [
      { title: "Key steps", items: ["initialize dp with inf, dp[0] = 0", "iterate coins", "relax dp[amount]"] },
      { title: "When to use", items: ["unbounded knapsack", "minimizing counts", "canonical coin sets"] },
      { title: "Pitfalls", items: ["forgetting impossible case", "wrong loop order for unbounded use"] },
    ],
    complexity: { time: "O(amount * coins)", space: "O(amount)" },
    pythonCode: coinChangeCode,
    exampleProblems: ["Coin Change", "Minimum Coins to Make Change", "Perfect Squares"],
  },
  {
    slug: "square-submatrix",
    title: "Square Submatrix",
    summary: "Find the largest all-1 square in a matrix.",
    kind: "algorithm",
    description:
      "DP tracks the largest square ending at each cell using the minimum of top, left, and diagonal neighbors.",
    sections: [
      { title: "Key steps", items: ["dp cell = 1 + min(top, left, diag)", "track global max", "handle borders"] },
      { title: "When to use", items: ["binary matrix", "largest square or area", "image/grid problems"] },
      { title: "Pitfalls", items: ["not handling first row/col", "treating strings as ints"] },
    ],
    complexity: { time: "O(R * C)", space: "O(R * C)" },
    pythonCode: squareSubmatrixCode,
    exampleProblems: ["Maximal Square", "Largest Square of 1s"],
  },
  {
    slug: "target-sum",
    title: "Target Sum",
    summary: "Count ways to assign +/- to reach a target.",
    kind: "algorithm",
    description:
      "Iteratively build a map of possible sums and counts by adding and subtracting each number.",
    sections: [
      { title: "Key steps", items: ["start with {0:1}", "update sums for +x and -x", "read dp[target]"] },
      { title: "When to use", items: ["sign assignment", "subset sum variants", "counting combinations"] },
      { title: "Pitfalls", items: ["forgetting multiplicities", "large sum range without pruning"] },
    ],
    complexity: { time: "O(n * S)", space: "O(S)" },
    pythonCode: targetSumCode,
    exampleProblems: ["Target Sum", "Assign Signs", "Subset Sum Count"],
  },
  {
    slug: "zero-one-knapsack",
    title: "0/1 Knapsack",
    summary: "Pick items once to maximize value under capacity.",
    kind: "algorithm",
    description:
      "Process each item once, updating capacities backward so each item is used at most once.",
    sections: [
      { title: "Key steps", items: ["dp by capacity", "loop items", "iterate capacity backward"] },
      { title: "When to use", items: ["choose or skip each item", "subset selection", "budgeted optimization"] },
      { title: "Pitfalls", items: ["iterating capacity forward", "mixing weights and values"] },
    ],
    complexity: { time: "O(n * C)", space: "O(C)" },
    pythonCode: knapsackCode,
    exampleProblems: ["0/1 Knapsack", "Partition Equal Subset Sum", "Last Stone Weight II"],
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

const baseTraversalLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 50, y: 10 },
    { id: "B", x: 20, y: 35 },
    { id: "C", x: 80, y: 35 },
    { id: "D", x: 10, y: 60 },
    { id: "E", x: 40, y: 60 },
    { id: "F", x: 90, y: 60 },
    { id: "G", x: 50, y: 85 },
  ],
  edges: [
    { from: "A", to: "B" },
    { from: "A", to: "C" },
    { from: "B", to: "D" },
    { from: "B", to: "E" },
    { from: "C", to: "F" },
    { from: "D", to: "G" },
    { from: "E", to: "G" },
    { from: "F", to: "G" },
  ],
};

const topoLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 15, y: 20 },
    { id: "B", x: 50, y: 10 },
    { id: "C", x: 85, y: 20 },
    { id: "D", x: 25, y: 60 },
    { id: "E", x: 60, y: 60 },
    { id: "F", x: 85, y: 80 },
  ],
  edges: [
    { from: "A", to: "D" },
    { from: "A", to: "E" },
    { from: "B", to: "E" },
    { from: "C", to: "F" },
    { from: "D", to: "F" },
    { from: "E", to: "F" },
  ],
};

const weightedGraphLayout = {
  directed: false,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 15, y: 20 },
    { id: "B", x: 50, y: 10 },
    { id: "C", x: 85, y: 25 },
    { id: "D", x: 30, y: 70 },
    { id: "E", x: 70, y: 75 },
  ],
  edges: [
    { from: "A", to: "B", weight: 2 },
    { from: "A", to: "D", weight: 6 },
    { from: "B", to: "C", weight: 3 },
    { from: "B", to: "D", weight: 8 },
    { from: "B", to: "E", weight: 5 },
    { from: "C", to: "E", weight: 7 },
    { from: "D", to: "E", weight: 2 },
  ],
};

const bellmanFordLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "S", x: 10, y: 50 },
    { id: "A", x: 35, y: 20 },
    { id: "B", x: 35, y: 80 },
    { id: "C", x: 65, y: 30 },
    { id: "D", x: 85, y: 60 },
  ],
  edges: [
    { from: "S", to: "A", weight: 4 },
    { from: "S", to: "B", weight: 5 },
    { from: "A", to: "C", weight: -3 },
    { from: "B", to: "C", weight: 6 },
    { from: "A", to: "D", weight: 5 },
    { from: "B", to: "D", weight: 2 },
    { from: "C", to: "D", weight: 2 },
  ],
};

const floydWarshallLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 20, y: 20 },
    { id: "B", x: 80, y: 20 },
    { id: "C", x: 20, y: 80 },
    { id: "D", x: 80, y: 80 },
  ],
  edges: [
    { from: "A", to: "B", weight: 3 },
    { from: "A", to: "C", weight: 7 },
    { from: "B", to: "D", weight: 2 },
    { from: "C", to: "D", weight: 1 },
    { from: "B", to: "C", weight: 2 },
  ],
};

const unionFindLayout = {
  directed: false,
  nodeRadius: 7,
  nodes: [
    { id: "1", x: 10, y: 50 },
    { id: "2", x: 25, y: 50 },
    { id: "3", x: 40, y: 50 },
    { id: "4", x: 55, y: 50 },
    { id: "5", x: 70, y: 50 },
    { id: "6", x: 85, y: 50 },
  ],
  edges: [
    { from: "1", to: "2" },
    { from: "2", to: "3" },
    { from: "4", to: "5" },
    { from: "3", to: "5" },
    { from: "6", to: "2" },
  ],
};

const sccLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 20, y: 30 },
    { id: "B", x: 50, y: 15 },
    { id: "C", x: 80, y: 30 },
    { id: "D", x: 35, y: 75 },
    { id: "E", x: 70, y: 80 },
  ],
  edges: [
    { from: "A", to: "B" },
    { from: "B", to: "C" },
    { from: "C", to: "A" },
    { from: "B", to: "D" },
    { from: "D", to: "E" },
    { from: "E", to: "D" },
  ],
};

const bridgesLayout = {
  directed: false,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 15, y: 40 },
    { id: "B", x: 35, y: 20 },
    { id: "C", x: 55, y: 40 },
    { id: "D", x: 75, y: 20 },
    { id: "E", x: 85, y: 55 },
    { id: "F", x: 35, y: 75 },
  ],
  edges: [
    { from: "A", to: "B" },
    { from: "B", to: "C" },
    { from: "C", to: "D" },
    { from: "D", to: "E" },
    { from: "C", to: "E" },
    { from: "C", to: "F" },
  ],
};

const eulerianLayout = {
  directed: false,
  nodeRadius: 7,
  nodes: [
    { id: "A", x: 25, y: 35 },
    { id: "B", x: 60, y: 20 },
    { id: "C", x: 75, y: 60 },
    { id: "D", x: 30, y: 75 },
  ],
  edges: [
    { from: "A", to: "B" },
    { from: "B", to: "C" },
    { from: "C", to: "A" },
    { from: "C", to: "D" },
  ],
};

const maxFlowLayout = {
  directed: true,
  nodeRadius: 7,
  nodes: [
    { id: "S", x: 10, y: 50 },
    { id: "A", x: 40, y: 20 },
    { id: "B", x: 40, y: 80 },
    { id: "T", x: 90, y: 50 },
  ],
  edges: [
    { from: "S", to: "A", weight: 3 },
    { from: "S", to: "B", weight: 2 },
    { from: "A", to: "B", weight: 1 },
    { from: "A", to: "T", weight: 2 },
    { from: "B", to: "T", weight: 3 },
  ],
};

const bipartiteLayout = {
  directed: false,
  nodeRadius: 7,
  nodes: [
    { id: "L1", x: 20, y: 20 },
    { id: "L2", x: 20, y: 50 },
    { id: "L3", x: 20, y: 80 },
    { id: "R1", x: 80, y: 20 },
    { id: "R2", x: 80, y: 50 },
    { id: "R3", x: 80, y: 80 },
  ],
  edges: [
    { from: "L1", to: "R1" },
    { from: "L1", to: "R2" },
    { from: "L2", to: "R1" },
    { from: "L2", to: "R3" },
    { from: "L3", to: "R2" },
    { from: "L3", to: "R3" },
  ],
};

const bfsSimulation = {
  title: "BFS traversal",
  description: "Visit nodes level by level using a queue.",
  goal: "Visit all reachable nodes from the start in level order.",
  inputs: ["Directed graph with adjacency list", "Start node A"],
  type: "graph-traversal",
  mode: "bfs",
  visual: "graph",
  graph: baseTraversalLayout,
  ...baseTraversalGraph,
};

const dfsSimulation = {
  title: "DFS traversal",
  description: "Go deep before backtracking with a stack.",
  goal: "Visit all reachable nodes from the start in depth-first order.",
  inputs: ["Directed graph with adjacency list", "Start node A"],
  type: "graph-traversal",
  mode: "dfs",
  visual: "graph",
  graph: baseTraversalLayout,
  ...baseTraversalGraph,
};

const simulationConfigs = {
  "sliding-window": {
    title: "Max sum window",
    description: "Find the max sum of any 3 consecutive elements.",
    goal: "Return the maximum sum of any window of size 3.",
    inputs: ["Array: [2, 1, 5, 1, 3, 2, 9, 7]", "Window size: 3"],
    type: "sliding-window",
    array: [2, 1, 5, 1, 3, 2, 9, 7],
    windowSize: 3,
  },
  "two-pointers": {
    title: "Two sum (sorted)",
    description: "Move pointers inward to hit the target sum.",
    goal: "Find indices whose values sum to the target.",
    inputs: ["Sorted array: [1, 2, 3, 4, 6, 8, 11]", "Target sum: 10"],
    type: "two-pointers",
    array: [1, 2, 3, 4, 6, 8, 11],
    target: 10,
  },
  "modified-binary-search": {
    title: "Binary search",
    description: "Halve the search range until the target is found.",
    goal: "Locate the target value in a sorted array.",
    inputs: ["Sorted array: [1..9]", "Target value: 6"],
    type: "binary-search",
    array: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    target: 6,
  },
  "cyclic-sort": {
    title: "Cyclic swap",
    description: "Swap each value into its correct index.",
    goal: "Place each value at index value-1 using swaps.",
    inputs: ["Array: [3, 1, 5, 4, 2]"],
    type: "cyclic-sort",
    array: [3, 1, 5, 4, 2],
  },
  "merge-sort": {
    title: "Merge sort",
    description: "Split, then merge sorted halves.",
    goal: "Sort the array using divide and conquer merges.",
    inputs: ["Array: [7, 2, 5, 3, 1, 6, 4]"],
    type: "merge-sort",
    array: [7, 2, 5, 3, 1, 6, 4],
  },
  "quick-sort": {
    title: "Quick sort",
    description: "Partition around a pivot and recurse.",
    goal: "Sort the array in-place using pivot partitions.",
    inputs: ["Array: [7, 2, 5, 3, 1, 6, 4]"],
    type: "quick-sort",
    array: [7, 2, 5, 3, 1, 6, 4],
  },
  "bfs-pattern": bfsSimulation,
  "dfs-pattern": dfsSimulation,
  "topological-sort": {
    title: "Topological order",
    description: "Peel off zero in-degree nodes to form a valid ordering.",
    goal: "Return a valid ordering of tasks in a DAG.",
    inputs: ["Nodes: A..F", "Directed edges list", "Graph is a DAG"],
    type: "topological-sort",
    visual: "graph",
    graph: topoLayout,
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
    goal: "Track the median after each insertion.",
    inputs: ["Stream: [5, 2, 10, 3, 8, 1]"],
    type: "two-heaps",
    stream: [5, 2, 10, 3, 8, 1],
  },
  "linear-search": {
    title: "Linear scan",
    description: "Check each value until the target is found.",
    goal: "Return the index of the target if present.",
    inputs: ["Array: [4, 1, 7, 3, 9, 2]", "Target: 9"],
    type: "linear-search",
    array: [4, 1, 7, 3, 9, 2],
    target: 9,
  },
  "binary-search": {
    title: "Binary search",
    description: "Halve the search range until the target is found.",
    goal: "Return the index of the target in a sorted array.",
    inputs: ["Sorted array: [1..9]", "Target: 6"],
    type: "binary-search",
    array: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    target: 6,
  },
  dijkstra: {
    title: "Shortest paths",
    description: "Relax edges using a min-priority frontier.",
    goal: "Compute the shortest distance from A to every node.",
    inputs: ["Weighted graph (non-negative)", "Start node A"],
    type: "dijkstra",
    visual: "graph",
    graph: weightedGraphLayout,
    start: "A",
  },
  "bellman-ford": {
    title: "Relax all edges",
    description: "Iteratively relax every edge to allow negative weights.",
    goal: "Compute shortest distances from S with possible negative edges.",
    inputs: ["Directed weighted graph", "Start node S"],
    type: "bellman-ford",
    visual: "graph",
    graph: bellmanFordLayout,
    start: "S",
  },
  "floyd-warshall": {
    title: "All-pairs distances",
    description: "Update the matrix with each intermediate node.",
    goal: "Compute shortest paths between every pair of nodes.",
    inputs: ["Directed weighted graph", "Distance matrix initialized"],
    type: "floyd-warshall",
    visual: "graph+matrix",
    graph: floydWarshallLayout,
  },
  "a-star": {
    title: "Heuristic search",
    description: "Pick the lowest f = g + h to guide the search.",
    goal: "Find a low-cost path from A to E using a heuristic.",
    inputs: ["Weighted graph", "Start A, Goal E", "Admissible heuristic"],
    type: "a-star",
    visual: "graph",
    graph: weightedGraphLayout,
    start: "A",
    goal: "E",
  },
  "union-find": {
    title: "Union operations",
    description: "Merge sets while tracking components.",
    goal: "Track connected components after each union.",
    inputs: ["Nodes 1..6", "Union operations list"],
    type: "union-find",
    visual: "graph",
    graph: unionFindLayout,
    operations: [
      ["1", "2"],
      ["2", "3"],
      ["4", "5"],
      ["3", "5"],
      ["6", "2"],
    ],
  },
  kruskal: {
    title: "MST via sorted edges",
    description: "Add the cheapest edge that does not form a cycle.",
    goal: "Build a minimum spanning tree by edge sorting.",
    inputs: ["Undirected weighted graph"],
    type: "kruskal",
    visual: "graph",
    graph: weightedGraphLayout,
  },
  prim: {
    title: "MST via growing frontier",
    description: "Expand the tree with the cheapest outgoing edge.",
    goal: "Build a minimum spanning tree from a start node.",
    inputs: ["Undirected weighted graph", "Start node A"],
    type: "prim",
    visual: "graph",
    graph: weightedGraphLayout,
    start: "A",
  },
  "strongly-connected-components": {
    title: "SCCs (Kosaraju)",
    description: "Run DFS order then reverse-graph DFS to group nodes.",
    goal: "Group nodes that are mutually reachable.",
    inputs: ["Directed graph"],
    type: "scc",
    visual: "graph",
    graph: sccLayout,
  },
  "bridges-articulation": {
    title: "Critical edges and nodes",
    description: "Track discovery/low links to find bridges.",
    goal: "Identify edges whose removal disconnects the graph.",
    inputs: ["Undirected graph"],
    type: "bridges-articulation",
    visual: "graph",
    graph: bridgesLayout,
  },
  "eulerian-path": {
    title: "Eulerian trail",
    description: "Walk edges once using a stack-based tour.",
    goal: "Visit every edge exactly once.",
    inputs: ["Undirected graph with Eulerian trail", "Start node D"],
    type: "eulerian-path",
    visual: "graph",
    graph: eulerianLayout,
    start: "D",
  },
  "max-flow": {
    title: "Augmenting paths",
    description: "Push flow until no path remains.",
    goal: "Compute the maximum flow from S to T.",
    inputs: ["Directed flow network with capacities", "Source S, Sink T"],
    type: "max-flow",
    visual: "graph",
    graph: maxFlowLayout,
    source: "S",
    sink: "T",
  },
  "bipartite-matching": {
    title: "Match left to right",
    description: "Find augmenting paths to grow matches.",
    goal: "Maximize the number of matched pairs.",
    inputs: ["Bipartite graph", "Left nodes L1..L3"],
    type: "bipartite-matching",
    visual: "graph",
    graph: bipartiteLayout,
    leftNodes: ["L1", "L2", "L3"],
  },
  fibonacci: {
    title: "DP array fill",
    description: "Build the Fibonacci table from base cases.",
    goal: "Compute the nth Fibonacci number via DP.",
    inputs: ["n = 10", "Base cases: fib(0)=0, fib(1)=1"],
    type: "dp-fibonacci",
    visual: "dp-array",
    n: 10,
  },
  "coin-change": {
    title: "Min coins DP",
    description: "Update dp[amount] using each coin.",
    goal: "Compute the minimum coins needed for the amount.",
    inputs: ["Amount: 12", "Coins: [1, 6, 10] (greedy fails here)"],
    inputArray: [1, 6, 10],
    type: "dp-coin-change",
    visual: "dp-array",
    amount: 12,
    coins: [1, 6, 10],
  },
  "square-submatrix": {
    title: "Maximal square DP",
    description: "Grow squares from top/left/diag neighbors.",
    goal: "Find the largest all-1 square submatrix.",
    inputs: ["Binary matrix 3x4 (True/False)"],
    type: "dp-square-submatrix",
    visual: "dp-matrix",
    inputMatrix: [
      ["False", "True", "False", "False"],
      ["True", "True", "True", "True"],
      ["False", "True", "True", "False"],
    ],
    matrix: [
      [0, 1, 0, 0],
      [1, 1, 1, 1],
      [0, 1, 1, 0],
    ],
  },
  "target-sum": {
    title: "Subset sum DP",
    description: "Count ways to reach the transformed target.",
    goal: "Count sign assignments that reach the target.",
    inputs: ["Numbers: [1, 1, 1, 1, 1]", "Target: 3"],
    inputArray: [1, 1, 1, 1, 1],
    type: "dp-target-sum",
    visual: "dp-array",
    nums: [1, 1, 1, 1, 1],
    target: 3,
  },
  "zero-one-knapsack": {
    title: "Capacity DP",
    description: "Update best value for each capacity.",
    goal: "Maximize value without exceeding capacity.",
    inputs: ["Weights: [2, 2, 3]", "Values: [6, 10, 12]", "Capacity: 5"],
    inputArrays: [
      { label: "Weights", values: [2, 2, 3] },
      { label: "Values", values: [6, 10, 12] },
    ],
    type: "dp-knapsack",
    visual: "dp-array",
    weights: [2, 2, 3],
    values: [6, 10, 12],
    capacity: 5,
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

const formatSimValue = (value) => {
  if (value === Infinity) {
    return "inf";
  }
  if (value === -Infinity) {
    return "-inf";
  }
  return String(value);
};

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

const clampOpRank = (rank) => {
  const value = Number(rank);
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.max(1, Math.min(5, value));
};

const renderOperationRow = (label, metric) => {
  if (!metric) {
    return "";
  }
  const noteHtml = metric.note ? `<span class="op-note">${escapeHtml(metric.note)}</span>` : "";
  const rank = clampOpRank(metric.rank);
  return `
    <div class="op-row" style="--rank:${rank};">
      <div class="op-label">
        <span class="op-name">${escapeHtml(label)}</span>
        ${noteHtml}
      </div>
      <div class="op-bar">
        <span class="op-fill"></span>
      </div>
      <div class="op-value">${escapeHtml(metric.label)}</div>
    </div>
  `;
};

const renderOperationsPanel = (operations) => {
  if (!operations) {
    return "";
  }
  const rows = [
    renderOperationRow("Read", operations.read),
    renderOperationRow("Write", operations.write),
    renderOperationRow("Search", operations.search),
  ]
    .filter(Boolean)
    .join("");
  if (!rows) {
    return "";
  }
  return `
    <div class="op-chart">
      <div class="op-chart-header">
        <h5 class="op-chart-title">Operation timing</h5>
        <p class="op-chart-note">Longer bars mean slower time complexity.</p>
      </div>
      <div class="op-chart-rows">${rows}</div>
    </div>
  `;
};

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

  const goalHtml = config.goal
    ? `
        <div>
          <strong>Goal</strong>
          <p>${escapeHtml(config.goal)}</p>
        </div>
      `
    : "";
  const inputsHtml = config.inputs?.length
    ? `
        <div>
          <strong>Inputs</strong>
          <ul class="sim-problem-list">${renderList(config.inputs)}</ul>
        </div>
      `
    : "";
  const problemHtml = goalHtml || inputsHtml ? `<div class="sim-problem">${goalHtml}${inputsHtml}</div>` : "";

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
      ${problemHtml}
      <div class="simulator-track" data-sim-track></div>
      <div class="simulator-state" data-sim-state></div>
    </section>
  `;
};

const edgeKey = (from, to, directed) => {
  if (directed) {
    return `${from}->${to}`;
  }
  return [from, to].sort().join("--");
};

const renderGraphCanvas = (graph, step) => {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const directed = Boolean(graph.directed);
  const selectedEdges = new Set(step.selectedEdges || []);
  const mutedEdges = new Set(step.mutedEdges || []);
  const activeKey = step.activeEdge ? edgeKey(step.activeEdge.from, step.activeEdge.to, directed) : null;
  const nodeRadius = graph.nodeRadius ?? 7;
  const offset = nodeRadius + 0.5;

  const edgesHtml = graph.edges
    .map((edge) => {
      const from = nodesById.get(edge.from);
      const to = nodesById.get(edge.to);
      if (!from || !to) {
        return "";
      }
      const key = edgeKey(edge.from, edge.to, directed);
      const classes = ["sim-edge"];
      if (key === activeKey) {
        classes.push("is-active");
      }
      if (selectedEdges.has(key)) {
        classes.push("is-selected");
      }
      if (mutedEdges.has(key)) {
        classes.push("is-muted");
      }
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy) || 1;
      const ox = (dx / length) * offset;
      const oy = (dy / length) * offset;
      const x1 = from.x + ox;
      const y1 = from.y + oy;
      const x2 = to.x - ox;
      const y2 = to.y - oy;
      const midX = (x1 + x2) / 2;
      const midY = (y1 + y2) / 2;
      const weightLabel = edge.weight !== undefined ? `<text class="sim-edge-weight" x="${midX}" y="${midY}">${escapeHtml(edge.weight)}</text>` : "";
      const marker = directed ? 'marker-end="url(#sim-arrow)"' : "";

      return `
        <line class="${classes.join(" ")}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${marker} />
        ${weightLabel}
      `;
    })
    .join("");

  const nodesHtml = graph.nodes
    .map((node) => {
      const classes = ["sim-node"];
      if (step.visited?.includes(node.id)) {
        classes.push("is-visited");
      }
      if (step.frontier?.includes(node.id)) {
        classes.push("is-frontier");
      }
      if (step.selectedNodes?.includes(node.id)) {
        classes.push("is-selected");
      }
      if (step.dependencyNodes?.includes(node.id)) {
        classes.push("is-dependency");
      }
      if (step.activeNode === node.id) {
        classes.push("is-active");
      }

      const label = step.nodeLabels?.[node.id];
      const labelHtml = label !== undefined ? `<span class="sim-degree">${escapeHtml(label)}</span>` : "";

      return `
        <div class="sim-graph-node" style="left:${node.x}%; top:${node.y}%;">
          <div class="${classes.join(" ")}">${escapeHtml(node.label ?? node.id)}</div>
          ${labelHtml}
        </div>
      `;
    })
    .join("");

  return `
    <div class="sim-graph-canvas">
      <svg class="sim-graph-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker id="sim-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto" markerUnits="strokeWidth">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path>
          </marker>
        </defs>
        ${edgesHtml}
      </svg>
      ${nodesHtml}
    </div>
  `;
};

const renderDpArray = (step) => {
  const values = step.dpArray || [];
  const itemsHtml = values
    .map((value, index) => {
      const classes = ["sim-cell"];
      if (step.baseIndices?.includes(index)) {
        classes.push("is-base");
      }
      if (step.dependencies?.includes(index)) {
        classes.push("is-dependency");
      }
      if (step.activeIndex === index) {
        classes.push("is-active");
      }

      return `
        <div class="sim-item">
          <div class="${classes.join(" ")}">${escapeHtml(formatSimValue(value))}</div>
          <span class="sim-index">${index}</span>
        </div>
      `;
    })
    .join("");

  return `<div class="sim-array">${itemsHtml}</div>`;
};

const renderDpMatrix = (step) => {
  const matrix = step.dpMatrix || [];
  const rowsHtml = matrix
    .map((row, rowIndex) => {
      const cellsHtml = row
        .map((value, colIndex) => {
          const classes = ["sim-cell"];
          if (step.baseCells?.some((cell) => cell.row === rowIndex && cell.col === colIndex)) {
            classes.push("is-base");
          }
          if (step.dependencies?.some((cell) => cell.row === rowIndex && cell.col === colIndex)) {
            classes.push("is-dependency");
          }
          if (step.activeCell?.row === rowIndex && step.activeCell?.col === colIndex) {
            classes.push("is-active");
          }
          return `<div class="${classes.join(" ")}">${escapeHtml(formatSimValue(value))}</div>`;
        })
        .join("");

      return `<div class="sim-matrix-row">${cellsHtml}</div>`;
    })
    .join("");

  return `<div class="sim-matrix">${rowsHtml}</div>`;
};

const renderInputMatrix = (matrix) => renderDpMatrix({ dpMatrix: matrix });

const renderStaticArray = (values, label) => {
  const itemsHtml = values
    .map((value, index) => {
      return `
        <div class="sim-item">
          <div class="sim-cell">${escapeHtml(formatSimValue(value))}</div>
          <span class="sim-index">${index}</span>
        </div>
      `;
    })
    .join("");

  return `
    <div>
      <p class="eyebrow">${escapeHtml(label)}</p>
      <div class="sim-array">${itemsHtml}</div>
    </div>
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
  const operationsHtml = item.kind === "data-structure" ? renderOperationsPanel(item.operations) : "";

  return `
    <div class="side-panel-header">
      <p class="eyebrow">${item.kind === "algorithm" ? "Algorithm" : "Data structure"}</p>
      <h3>${escapeHtml(item.title)}</h3>
      <p class="side-panel-summary">${escapeHtml(item.summary)}</p>
    </div>
    ${simulationHtml}
    <div class="side-panel-body">
      <p>${escapeHtml(item.description)}</p>
      ${operationsHtml}
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
      activeNode: current,
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
      activeNode: current,
      queue: [...queue],
      order: [...order],
      inDegree: { ...inDegree },
      visited: [...order],
      nodeLabels: Object.fromEntries(
        Object.entries(inDegree).map(([node, degree]) => [node, `deg ${degree}`])
      ),
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

const buildAdjacency = (graph) => {
  const adjacency = {};
  graph.nodes.forEach((node) => {
    adjacency[node.id] = [];
  });
  graph.edges.forEach((edge) => {
    adjacency[edge.from].push({ to: edge.to, weight: edge.weight ?? 1 });
    if (!graph.directed) {
      adjacency[edge.to].push({ to: edge.from, weight: edge.weight ?? 1 });
    }
  });
  return adjacency;
};

const buildDistanceLabels = (distances) => {
  const labels = {};
  Object.entries(distances).forEach(([node, value]) => {
    labels[node] = formatSimValue(value);
  });
  return labels;
};

const buildDijkstraSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const adjacency = buildAdjacency(graph);
  const distances = {};
  const visited = new Set();
  const frontier = [];

  nodes.forEach((node) => {
    distances[node] = Infinity;
  });
  distances[config.start] = 0;
  frontier.push({ node: config.start, dist: 0 });

  while (frontier.length) {
    frontier.sort((a, b) => a.dist - b.dist);
    const { node } = frontier.shift();
    if (visited.has(node)) {
      continue;
    }
    visited.add(node);

    steps.push({
      activeNode: node,
      visited: Array.from(visited),
      frontier: frontier.map((item) => item.node),
      distances: { ...distances },
      nodeLabels: buildDistanceLabels(distances),
    });

    adjacency[node].forEach((edge) => {
      const next = edge.to;
      const newDist = distances[node] + edge.weight;
      const updated = newDist < distances[next];
      if (updated) {
        distances[next] = newDist;
        frontier.push({ node: next, dist: newDist });
      }
      steps.push({
        activeNode: node,
        activeEdge: { from: node, to: next },
        visited: Array.from(visited),
        frontier: frontier.map((item) => item.node),
        distances: { ...distances },
        nodeLabels: buildDistanceLabels(distances),
        updated,
      });
    });
  }

  return steps;
};

const buildBellmanFordSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const edges = graph.edges;
  const distances = {};

  nodes.forEach((node) => {
    distances[node] = Infinity;
  });
  distances[config.start] = 0;

  for (let i = 0; i < nodes.length - 1; i += 1) {
    edges.forEach((edge) => {
      const from = edge.from;
      const to = edge.to;
      const weight = edge.weight ?? 0;
      let updated = false;

      if (distances[from] !== Infinity && distances[from] + weight < distances[to]) {
        distances[to] = distances[from] + weight;
        updated = true;
      }

      steps.push({
        iteration: i + 1,
        activeEdge: { from, to },
        updated,
        distances: { ...distances },
        nodeLabels: buildDistanceLabels(distances),
      });
    });
  }

  return steps;
};

const buildFloydWarshallSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const indexByNode = new Map(nodes.map((node, index) => [node, index]));
  const size = nodes.length;
  const dist = Array.from({ length: size }, () => Array(size).fill(Infinity));

  for (let i = 0; i < size; i += 1) {
    dist[i][i] = 0;
  }
  graph.edges.forEach((edge) => {
    const i = indexByNode.get(edge.from);
    const j = indexByNode.get(edge.to);
    if (edge.weight < dist[i][j]) {
      dist[i][j] = edge.weight;
    }
  });

  for (let k = 0; k < size; k += 1) {
    for (let i = 0; i < size; i += 1) {
      for (let j = 0; j < size; j += 1) {
        const alt = dist[i][k] + dist[k][j];
        if (alt < dist[i][j]) {
          dist[i][j] = alt;
        }
        steps.push({
          activeNode: nodes[k],
          dependencyNodes: [nodes[i], nodes[j]],
          dpMatrix: dist.map((row) => row.slice()),
          activeCell: { row: i, col: j },
          dependencies: [
            { row: i, col: k },
            { row: k, col: j },
          ],
          k,
          i,
          j,
        });
      }
    }
  }

  return steps;
};

const buildAStarSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const adjacency = buildAdjacency(graph);
  const goal = config.goal;
  const gScore = {};
  const frontier = [];
  const visited = new Set();

  const heuristic = (nodeId) => {
    const node = nodeById.get(nodeId);
    const target = nodeById.get(goal);
    if (!node || !target) {
      return 0;
    }
    const dx = node.x - target.x;
    const dy = node.y - target.y;
    return Math.round(Math.sqrt(dx * dx + dy * dy));
  };

  nodes.forEach((node) => {
    gScore[node] = Infinity;
  });
  gScore[config.start] = 0;
  frontier.push({ node: config.start, f: heuristic(config.start), g: 0 });

  while (frontier.length) {
    frontier.sort((a, b) => a.f - b.f);
    const current = frontier.shift();
    if (!current) {
      break;
    }
    const node = current.node;
    if (visited.has(node)) {
      continue;
    }
    visited.add(node);
    steps.push({
      activeNode: node,
      visited: Array.from(visited),
      frontier: frontier.map((item) => item.node),
      nodeLabels: buildDistanceLabels(gScore),
      goal,
    });
    if (node === goal) {
      break;
    }

    adjacency[node].forEach((edge) => {
      const next = edge.to;
      const tentative = gScore[node] + edge.weight;
      if (tentative < gScore[next]) {
        gScore[next] = tentative;
        frontier.push({ node: next, g: tentative, f: tentative + heuristic(next) });
      }
      steps.push({
        activeNode: node,
        activeEdge: { from: node, to: next },
        visited: Array.from(visited),
        frontier: frontier.map((item) => item.node),
        nodeLabels: buildDistanceLabels(gScore),
        goal,
      });
    });
  }

  return steps;
};

const buildUnionFindSteps = (config) => {
  const steps = [];
  const nodes = config.graph.nodes.map((node) => node.id);
  const parent = {};
  const rank = {};
  const selectedEdges = new Set();

  nodes.forEach((node) => {
    parent[node] = node;
    rank[node] = 0;
  });

  const find = (x) => {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  };

  const union = (a, b) => {
    let ra = find(a);
    let rb = find(b);
    if (ra === rb) {
      return false;
    }
    if (rank[ra] < rank[rb]) {
      [ra, rb] = [rb, ra];
    }
    parent[rb] = ra;
    if (rank[ra] === rank[rb]) {
      rank[ra] += 1;
    }
    return true;
  };

  config.operations.forEach(([a, b]) => {
    const merged = union(a, b);
    const key = edgeKey(a, b, false);
    if (merged) {
      selectedEdges.add(key);
    }
    steps.push({
      activeEdge: { from: a, to: b },
      selectedEdges: Array.from(selectedEdges),
      nodeLabels: { ...parent },
      merged,
    });
  });

  return steps;
};

const buildKruskalSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const parent = {};
  const rank = {};
  const selectedEdges = new Set();
  let totalWeight = 0;

  nodes.forEach((node) => {
    parent[node] = node;
    rank[node] = 0;
  });

  const find = (x) => {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  };

  const union = (a, b) => {
    let ra = find(a);
    let rb = find(b);
    if (ra === rb) {
      return false;
    }
    if (rank[ra] < rank[rb]) {
      [ra, rb] = [rb, ra];
    }
    parent[rb] = ra;
    if (rank[ra] === rank[rb]) {
      rank[ra] += 1;
    }
    return true;
  };

  const edges = [...graph.edges].sort((a, b) => (a.weight ?? 0) - (b.weight ?? 0));
  edges.forEach((edge) => {
    const key = edgeKey(edge.from, edge.to, false);
    const added = union(edge.from, edge.to);
    if (added) {
      selectedEdges.add(key);
      totalWeight += edge.weight ?? 0;
    }
    steps.push({
      activeEdge: { from: edge.from, to: edge.to },
      selectedEdges: Array.from(selectedEdges),
      action: added ? "add" : "skip",
      totalWeight,
    });
  });

  return steps;
};

const buildPrimSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const adjacency = buildAdjacency(graph);
  const visited = new Set([config.start]);
  const selectedEdges = new Set();
  let totalWeight = 0;
  const edgePool = [];

  adjacency[config.start].forEach((edge) => {
    edgePool.push({ from: config.start, to: edge.to, weight: edge.weight });
  });

  while (edgePool.length) {
    edgePool.sort((a, b) => a.weight - b.weight);
    const edge = edgePool.shift();
    if (!edge) {
      break;
    }
    if (visited.has(edge.to)) {
      continue;
    }
    visited.add(edge.to);
    totalWeight += edge.weight;
    selectedEdges.add(edgeKey(edge.from, edge.to, false));
    steps.push({
      activeEdge: { from: edge.from, to: edge.to },
      visited: Array.from(visited),
      selectedEdges: Array.from(selectedEdges),
      totalWeight,
    });
    adjacency[edge.to].forEach((next) => {
      if (!visited.has(next.to)) {
        edgePool.push({ from: edge.to, to: next.to, weight: next.weight });
      }
    });
  }

  return steps;
};

const buildSccSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const adjacency = buildAdjacency(graph);
  const reverseAdj = {};
  nodes.forEach((node) => {
    reverseAdj[node] = [];
  });
  graph.edges.forEach((edge) => {
    reverseAdj[edge.to].push(edge.from);
  });

  const seen = new Set();
  const order = [];

  const dfs = (node) => {
    seen.add(node);
    adjacency[node].forEach((edge) => {
      if (!seen.has(edge.to)) {
        dfs(edge.to);
      }
    });
    order.push(node);
    steps.push({
      phase: "order",
      activeNode: node,
      visited: Array.from(seen),
      order: [...order],
    });
  };

  nodes.forEach((node) => {
    if (!seen.has(node)) {
      dfs(node);
    }
  });

  const components = {};
  seen.clear();
  let componentId = 0;

  const dfsRev = (node) => {
    seen.add(node);
    components[node] = componentId;
    reverseAdj[node].forEach((next) => {
      if (!seen.has(next)) {
        dfsRev(next);
      }
    });
  };

  [...order].reverse().forEach((node) => {
    if (!seen.has(node)) {
      componentId += 1;
      dfsRev(node);
      steps.push({
        phase: "component",
        activeNode: node,
        nodeLabels: { ...components },
      });
    }
  });

  return steps;
};

const buildBridgesSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const adjacency = buildAdjacency(graph);
  const disc = {};
  const low = {};
  const parent = {};
  const bridges = [];
  let time = 0;

  const labelMap = () => {
    const labels = {};
    Object.keys(disc).forEach((node) => {
      labels[node] = `${disc[node]}/${low[node]}`;
    });
    return labels;
  };

  const dfs = (node) => {
    time += 1;
    disc[node] = time;
    low[node] = time;
    steps.push({
      activeNode: node,
      nodeLabels: labelMap(),
      selectedEdges: bridges.map((edge) => edgeKey(edge.from, edge.to, false)),
    });

    adjacency[node].forEach((edge) => {
      const next = edge.to;
      if (!(next in disc)) {
        parent[next] = node;
        steps.push({
          activeEdge: { from: node, to: next },
          nodeLabels: labelMap(),
          selectedEdges: bridges.map((bridge) => edgeKey(bridge.from, bridge.to, false)),
        });
        dfs(next);
        low[node] = Math.min(low[node], low[next]);
        if (low[next] > disc[node]) {
          bridges.push({ from: node, to: next });
        }
        steps.push({
          activeNode: node,
          nodeLabels: labelMap(),
          selectedEdges: bridges.map((bridge) => edgeKey(bridge.from, bridge.to, false)),
        });
      } else if (parent[node] !== next) {
        low[node] = Math.min(low[node], disc[next]);
      }
    });
  };

  graph.nodes.forEach((node) => {
    if (!(node.id in disc)) {
      parent[node.id] = null;
      dfs(node.id);
    }
  });

  return steps;
};

const buildEulerianSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const adjacency = {};
  graph.nodes.forEach((node) => {
    adjacency[node.id] = [];
  });
  graph.edges.forEach((edge) => {
    adjacency[edge.from].push(edge.to);
    adjacency[edge.to].push(edge.from);
  });

  const stack = [config.start];
  const path = [];
  const usedEdges = new Set();

  const useEdge = (from, to) => {
    const key = edgeKey(from, to, false);
    usedEdges.add(key);
  };

  while (stack.length) {
    const node = stack[stack.length - 1];
    if (adjacency[node].length) {
      const next = adjacency[node].pop();
      const idx = adjacency[next].indexOf(node);
      if (idx >= 0) {
        adjacency[next].splice(idx, 1);
      }
      useEdge(node, next);
      steps.push({
        activeNode: node,
        activeEdge: { from: node, to: next },
        selectedEdges: Array.from(usedEdges),
        stack: [...stack],
        path: [...path],
      });
      stack.push(next);
    } else {
      path.push(stack.pop());
      steps.push({
        activeNode: node,
        selectedEdges: Array.from(usedEdges),
        stack: [...stack],
        path: [...path],
      });
    }
  }

  return steps;
};

const buildMaxFlowSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const nodes = graph.nodes.map((node) => node.id);
  const capacity = {};
  const adjacency = {};
  nodes.forEach((node) => {
    adjacency[node] = [];
  });
  graph.edges.forEach((edge) => {
    capacity[`${edge.from}->${edge.to}`] = edge.weight ?? 0;
    capacity[`${edge.to}->${edge.from}`] = 0;
    adjacency[edge.from].push(edge.to);
    adjacency[edge.to].push(edge.from);
  });

  const source = config.source;
  const sink = config.sink;
  let flow = 0;

  while (true) {
    const parent = {};
    const queue = [source];
    parent[source] = source;

    while (queue.length && !(sink in parent)) {
      const node = queue.shift();
      adjacency[node].forEach((next) => {
        const cap = capacity[`${node}->${next}`] ?? 0;
        if (!(next in parent) && cap > 0) {
          parent[next] = node;
          queue.push(next);
        }
      });
    }

    if (!(sink in parent)) {
      break;
    }

    let bottleneck = Infinity;
    let node = sink;
    const pathEdges = [];
    while (node !== source) {
      const prev = parent[node];
      bottleneck = Math.min(bottleneck, capacity[`${prev}->${node}`]);
      pathEdges.push({ from: prev, to: node });
      node = prev;
    }

    node = sink;
    while (node !== source) {
      const prev = parent[node];
      capacity[`${prev}->${node}`] -= bottleneck;
      capacity[`${node}->${prev}`] += bottleneck;
      node = prev;
    }

    flow += bottleneck;
    steps.push({
      activeEdge: pathEdges[pathEdges.length - 1],
      selectedEdges: pathEdges.map((edge) => edgeKey(edge.from, edge.to, true)),
      flow,
      bottleneck,
    });
  }

  return steps;
};

const buildBipartiteMatchingSteps = (config) => {
  const steps = [];
  const graph = config.graph;
  const matchRight = {};
  const matchedEdges = new Set();

  const neighbors = {};
  graph.nodes.forEach((node) => {
    neighbors[node.id] = [];
  });
  graph.edges.forEach((edge) => {
    neighbors[edge.from].push(edge.to);
    neighbors[edge.to].push(edge.from);
  });

  const dfs = (u, seen) => {
    for (const v of neighbors[u]) {
      if (seen.has(v)) {
        continue;
      }
      seen.add(v);
      if (!(v in matchRight) || dfs(matchRight[v], seen)) {
        matchRight[v] = u;
        return true;
      }
    }
    return false;
  };

  config.leftNodes.forEach((u) => {
    const matched = dfs(u, new Set());
    matchedEdges.clear();
    Object.entries(matchRight).forEach(([v, left]) => {
      matchedEdges.add(edgeKey(left, v, false));
    });
    steps.push({
      activeNode: u,
      selectedEdges: Array.from(matchedEdges),
      matched,
    });
  });

  return steps;
};

const buildFibonacciDpSteps = (config) => {
  const steps = [];
  const n = config.n;
  const dp = Array(n + 1).fill(0);
  const baseIndices = [0, 1];
  dp[1] = 1;
  steps.push({
    dpArray: dp.slice(),
    activeIndex: 0,
    baseIndices,
  });
  steps.push({
    dpArray: dp.slice(),
    activeIndex: 1,
    baseIndices,
  });

  for (let i = 2; i <= n; i += 1) {
    dp[i] = dp[i - 1] + dp[i - 2];
    steps.push({
      dpArray: dp.slice(),
      activeIndex: i,
      dependencies: [i - 1, i - 2],
      baseIndices,
    });
  }

  return steps;
};

const buildCoinChangeDpSteps = (config) => {
  const steps = [];
  const amount = config.amount;
  const dp = Array(amount + 1).fill(Infinity);
  dp[0] = 0;
  steps.push({ dpArray: dp.slice(), activeIndex: 0, baseIndices: [0], coin: "-" });

  config.coins.forEach((coin) => {
    for (let total = coin; total <= amount; total += 1) {
      const candidate = dp[total - coin] + 1;
      if (candidate < dp[total]) {
        dp[total] = candidate;
      }
      steps.push({
        dpArray: dp.slice(),
        activeIndex: total,
        dependencies: [total - coin],
        baseIndices: [0],
        coin,
      });
    }
  });

  return steps;
};

const buildSquareSubmatrixDpSteps = (config) => {
  const steps = [];
  const matrix = config.matrix;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const dp = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const deps = [];
      const baseCells = [];
      if (matrix[r][c] === 1) {
        if (r === 0 || c === 0) {
          dp[r][c] = 1;
          baseCells.push({ row: r, col: c });
        } else {
          deps.push({ row: r - 1, col: c });
          deps.push({ row: r, col: c - 1 });
          deps.push({ row: r - 1, col: c - 1 });
          dp[r][c] = 1 + Math.min(dp[r - 1][c], dp[r][c - 1], dp[r - 1][c - 1]);
        }
      }
      steps.push({
        dpMatrix: dp.map((row) => row.slice()),
        activeCell: { row: r, col: c },
        dependencies: deps,
        baseCells,
        inputValue: matrix[r][c],
      });
    }
  }

  return steps;
};

const buildTargetSumDpSteps = (config) => {
  const steps = [];
  const nums = config.nums;
  const total = nums.reduce((acc, x) => acc + x, 0);
  const target = config.target;
  const subset = (total + target) / 2;
  if (!Number.isInteger(subset)) {
    steps.push({ dpArray: [0], activeIndex: 0 });
    return steps;
  }
  const dp = Array(subset + 1).fill(0);
  dp[0] = 1;
  steps.push({ dpArray: dp.slice(), activeIndex: 0, baseIndices: [0], num: "-" });

  nums.forEach((num) => {
    for (let sum = subset; sum >= num; sum -= 1) {
      dp[sum] += dp[sum - num];
      steps.push({
        dpArray: dp.slice(),
        activeIndex: sum,
        dependencies: [sum - num],
        baseIndices: [0],
        num,
        subset,
      });
    }
  });

  return steps;
};

const buildKnapsackDpSteps = (config) => {
  const steps = [];
  const capacity = config.capacity;
  const dp = Array(capacity + 1).fill(0);

  config.weights.forEach((weight, index) => {
    const value = config.values[index];
    for (let cap = capacity; cap >= weight; cap -= 1) {
      dp[cap] = Math.max(dp[cap], dp[cap - weight] + value);
      steps.push({
        dpArray: dp.slice(),
        activeIndex: cap,
        dependencies: [cap - weight],
        item: index + 1,
        weight,
        value,
      });
    }
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
    case "dijkstra":
      return buildDijkstraSteps(config);
    case "bellman-ford":
      return buildBellmanFordSteps(config);
    case "floyd-warshall":
      return buildFloydWarshallSteps(config);
    case "a-star":
      return buildAStarSteps(config);
    case "union-find":
      return buildUnionFindSteps(config);
    case "kruskal":
      return buildKruskalSteps(config);
    case "prim":
      return buildPrimSteps(config);
    case "scc":
      return buildSccSteps(config);
    case "bridges-articulation":
      return buildBridgesSteps(config);
    case "eulerian-path":
      return buildEulerianSteps(config);
    case "max-flow":
      return buildMaxFlowSteps(config);
    case "bipartite-matching":
      return buildBipartiteMatchingSteps(config);
    case "dp-fibonacci":
      return buildFibonacciDpSteps(config);
    case "dp-coin-change":
      return buildCoinChangeDpSteps(config);
    case "dp-square-submatrix":
      return buildSquareSubmatrixDpSteps(config);
    case "dp-target-sum":
      return buildTargetSumDpSteps(config);
    case "dp-knapsack":
      return buildKnapsackDpSteps(config);
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
  if (config.visual === "graph") {
    track.innerHTML = renderGraphCanvas(config.graph, step);
    return;
  }

  if (config.visual === "graph+matrix") {
    const graphHtml = renderGraphCanvas(config.graph, step);
    const matrixHtml = renderDpMatrix(step);
    track.innerHTML = `<div class="sim-split">${graphHtml}${matrixHtml}</div>`;
    return;
  }

  if (config.visual === "dp-array") {
    if (config.inputArray || config.inputArrays) {
      const arrays = config.inputArrays || [{ label: "Input", values: config.inputArray }];
      const inputsHtml = arrays.map((arr) => renderStaticArray(arr.values, arr.label)).join("");
      const dpHtml = `
        <div>
          <p class="eyebrow">DP table</p>
          ${renderDpArray(step)}
        </div>
      `;
      track.innerHTML = `<div class="sim-stack">${inputsHtml}${dpHtml}</div>`;
      return;
    }
    track.innerHTML = renderDpArray(step);
    return;
  }

  if (config.visual === "dp-matrix") {
    if (config.inputMatrix) {
      const inputHtml = renderInputMatrix(config.inputMatrix);
      const dpHtml = renderDpMatrix(step);
      track.innerHTML = `
        <div class="sim-split">
          <div>
            <p class="eyebrow">Input</p>
            ${inputHtml}
          </div>
          <div>
            <p class="eyebrow">DP table</p>
            ${dpHtml}
          </div>
        </div>
      `;
      return;
    }
    track.innerHTML = renderDpMatrix(step);
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

  if (config.type === "dijkstra") {
    const distances = Object.entries(step.distances || {})
      .map(([node, value]) => `${node}:${formatSimValue(value)}`)
      .join(", ");
    baseItems.push(
      { label: "Current", value: step.activeNode || "-" },
      { label: "Frontier", value: (step.frontier || []).join(", ") || "-" },
      { label: "Distances", value: distances || "-" }
    );
  }

  if (config.type === "bellman-ford") {
    const distances = Object.entries(step.distances || {})
      .map(([node, value]) => `${node}:${formatSimValue(value)}`)
      .join(", ");
    const edgeLabel = step.activeEdge ? `${step.activeEdge.from}->${step.activeEdge.to}` : "-";
    baseItems.push(
      { label: "Iteration", value: String(step.iteration) },
      { label: "Edge", value: edgeLabel },
      { label: "Updated", value: step.updated ? "yes" : "no" },
      { label: "Distances", value: distances || "-" }
    );
  }

  if (config.type === "floyd-warshall") {
    baseItems.push(
      { label: "k", value: String(step.k) },
      { label: "i", value: String(step.i) },
      { label: "j", value: String(step.j) }
    );
  }

  if (config.type === "a-star") {
    const distances = Object.entries(step.nodeLabels || {})
      .map(([node, value]) => `${node}:${value}`)
      .join(", ");
    baseItems.push(
      { label: "Current", value: step.activeNode || "-" },
      { label: "Goal", value: step.goal || "-" },
      { label: "Frontier", value: (step.frontier || []).join(", ") || "-" },
      { label: "gScore", value: distances || "-" }
    );
  }

  if (config.type === "union-find") {
    const parents = Object.entries(step.nodeLabels || {})
      .map(([node, value]) => `${node}->${value}`)
      .join(", ");
    const edgeLabel = step.activeEdge ? `${step.activeEdge.from}-${step.activeEdge.to}` : "-";
    baseItems.push(
      { label: "Union", value: edgeLabel },
      { label: "Merged", value: step.merged ? "yes" : "no" },
      { label: "Parents", value: parents || "-" }
    );
  }

  if (config.type === "kruskal") {
    const edgeLabel = step.activeEdge ? `${step.activeEdge.from}-${step.activeEdge.to}` : "-";
    baseItems.push(
      { label: "Edge", value: edgeLabel },
      { label: "Action", value: step.action || "-" },
      { label: "Total weight", value: String(step.totalWeight ?? 0) }
    );
  }

  if (config.type === "prim") {
    const edgeLabel = step.activeEdge ? `${step.activeEdge.from}-${step.activeEdge.to}` : "-";
    baseItems.push(
      { label: "Edge", value: edgeLabel },
      { label: "Visited", value: (step.visited || []).join(", ") || "-" },
      { label: "Total weight", value: String(step.totalWeight ?? 0) }
    );
  }

  if (config.type === "scc") {
    const components = Object.entries(step.nodeLabels || {})
      .map(([node, value]) => `${node}:${value}`)
      .join(", ");
    baseItems.push(
      { label: "Phase", value: step.phase || "-" },
      { label: "Order", value: (step.order || []).join(" -> ") || "-" },
      { label: "Components", value: components || "-" }
    );
  }

  if (config.type === "bridges-articulation") {
    const bridges = (step.selectedEdges || []).join(", ") || "-";
    baseItems.push(
      { label: "Bridges", value: bridges },
      { label: "Labels", value: Object.values(step.nodeLabels || {}).join(", ") || "-" }
    );
  }

  if (config.type === "eulerian-path") {
    baseItems.push(
      { label: "Stack", value: (step.stack || []).join(" -> ") || "-" },
      { label: "Path", value: (step.path || []).join(" -> ") || "-" }
    );
  }

  if (config.type === "max-flow") {
    baseItems.push(
      { label: "Flow", value: String(step.flow ?? 0) },
      { label: "Bottleneck", value: String(step.bottleneck ?? "-") },
      { label: "Path edges", value: (step.selectedEdges || []).join(", ") || "-" }
    );
  }

  if (config.type === "bipartite-matching") {
    baseItems.push(
      { label: "Active left", value: step.activeNode || "-" },
      { label: "Matched", value: step.matched ? "yes" : "no" },
      { label: "Matches", value: (step.selectedEdges || []).join(", ") || "-" }
    );
  }

  if (config.type === "dp-fibonacci") {
    baseItems.push(
      { label: "Index", value: String(step.activeIndex) },
      { label: "Value", value: formatSimValue(step.dpArray?.[step.activeIndex]) }
    );
  }

  if (config.type === "dp-coin-change") {
    baseItems.push(
      { label: "Coin", value: String(step.coin) },
      { label: "Amount", value: String(step.activeIndex) },
      { label: "Min coins", value: formatSimValue(step.dpArray?.[step.activeIndex]) }
    );
  }

  if (config.type === "dp-square-submatrix") {
    const cellValue = step.dpMatrix?.[step.activeCell?.row]?.[step.activeCell?.col];
    baseItems.push(
      { label: "Cell", value: `${step.activeCell?.row},${step.activeCell?.col}` },
      { label: "Value", value: String(step.inputValue) },
      { label: "Square size", value: formatSimValue(cellValue) }
    );
  }

  if (config.type === "dp-target-sum") {
    baseItems.push(
      { label: "Num", value: String(step.num) },
      { label: "Sum", value: String(step.activeIndex) },
      { label: "Ways", value: formatSimValue(step.dpArray?.[step.activeIndex]) }
    );
  }

  if (config.type === "dp-knapsack") {
    baseItems.push(
      { label: "Item", value: String(step.item) },
      { label: "Capacity", value: String(step.activeIndex) },
      { label: "Best value", value: formatSimValue(step.dpArray?.[step.activeIndex]) }
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
  if (item.operations) {
    Object.values(item.operations).forEach((metric) => {
      if (!metric) {
        return;
      }
      if (metric.label) {
        parts.push(metric.label);
      }
      if (metric.note) {
        parts.push(metric.note);
      }
    });
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
