export type Pointer = { name: string; index: number; color?: string };

export type VizState = {
  array?: number[];
  window?: { l: number; r: number };
  pointers?: Pointer[];
  intervals?: { start: number; end: number }[];
  graph?: { nodes: string[]; edges: [string, string][] };
  heapLeft?: number[];
  heapRight?: number[];
  matrix?: number[][];
  notes?: string;
};

export type Step = {
  title: string;
  state: VizState;
  codeLineHighlights?: number[];
};

export type PatternExample = {
  title: string;
  url?: string;
};

export type PatternPreset<TInput> = {
  label: string;
  input: TInput;
};

export type PatternDefinition<TInput = any> = {
  slug: string;
  title: string;
  summary: string;
  description: string;
  signals: string[];
  invariants: string[];
  pitfalls: string[];
  complexity: { time: string; space: string };
  pythonCode: string;
  exampleProblems: PatternExample[];
  presets: PatternPreset<TInput>[];
  randomInput: (seed: number) => TInput;
  makeSteps: (input: TInput) => Step[];
};
