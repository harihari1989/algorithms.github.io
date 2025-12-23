import { useEffect, useMemo, useState } from 'react';
import type { Step } from './types';

type StepperOptions = {
  initialIndex?: number;
  initialSpeedMs?: number;
};

export type StepperControls = {
  isPlaying: boolean;
  speedMs: number;
  next: () => void;
  prev: () => void;
  reset: () => void;
  seek: (index: number) => void;
  togglePlay: () => void;
  setSpeedMs: (value: number) => void;
};

export function useStepper(steps: Step[], options?: StepperOptions) {
  const [index, setIndex] = useState(options?.initialIndex ?? 0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(options?.initialSpeedMs ?? 900);

  const total = steps.length;

  useEffect(() => {
    setIndex(0);
    setIsPlaying(false);
  }, [steps]);

  useEffect(() => {
    if (!isPlaying || total <= 1) {
      return;
    }
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= total - 1) {
          return current;
        }
        return current + 1;
      });
    }, speedMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, speedMs, total]);

  useEffect(() => {
    if (isPlaying && index >= total - 1) {
      setIsPlaying(false);
    }
  }, [index, isPlaying, total]);

  const maxIndex = Math.max(total - 1, 0);

  const controls = useMemo<StepperControls>(
    () => ({
      isPlaying,
      speedMs,
      next: () => setIndex((current) => Math.min(current + 1, maxIndex)),
      prev: () => setIndex((current) => Math.max(current - 1, 0)),
      reset: () => {
        setIndex(0);
        setIsPlaying(false);
      },
      seek: (nextIndex) => setIndex(Math.max(0, Math.min(nextIndex, maxIndex))),
      togglePlay: () => setIsPlaying((value) => !value),
      setSpeedMs,
    }),
    [isPlaying, speedMs, maxIndex]
  );

  return {
    index,
    step: steps[index],
    controls,
  };
}
