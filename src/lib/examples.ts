export const exampleArrays = {
  medium: [4, 2, 1, 7, 5, 3, 6],
  window: [2, 1, 5, 1, 3, 2],
  sorted: [-4, -1, 0, 2, 5, 7, 9],
};

export const exampleIntervals = [
  { start: 1, end: 3 },
  { start: 2, end: 6 },
  { start: 8, end: 10 },
  { start: 9, end: 12 },
];

export const exampleGraph = {
  nodes: ['A', 'B', 'C', 'D', 'E', 'F'],
  edges: [
    ['A', 'B'],
    ['A', 'C'],
    ['B', 'D'],
    ['C', 'D'],
    ['C', 'E'],
    ['E', 'F'],
  ] as [string, string][],
};

export const exampleDag = {
  nodes: ['Intro', 'Arrays', 'Graphs', 'DP', 'Heap', 'Greedy'],
  edges: [
    ['Intro', 'Arrays'],
    ['Intro', 'Graphs'],
    ['Arrays', 'DP'],
    ['Graphs', 'Heap'],
    ['Heap', 'Greedy'],
  ] as [string, string][],
};

export const exampleLists = [
  [1, 4, 7],
  [2, 6, 8],
  [3, 5, 9],
];
