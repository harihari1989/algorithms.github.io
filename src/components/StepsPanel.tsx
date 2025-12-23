import type { Step } from '../lib/types';

type StepsPanelProps = {
  steps: Step[];
  activeIndex: number;
  onSelect: (index: number) => void;
};

export default function StepsPanel({ steps, activeIndex, onSelect }: StepsPanelProps) {
  return (
    <div className="steps-panel">
      {steps.map((step, index) => (
        <button
          key={`${step.title}-${index}`}
          type="button"
          className={`step-button${index === activeIndex ? ' active' : ''}`}
          onClick={() => onSelect(index)}
        >
          {index + 1}. {step.title}
        </button>
      ))}
    </div>
  );
}
