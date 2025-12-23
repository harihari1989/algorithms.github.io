import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useStepper } from '../lib/runner';
import type { PatternDefinition } from '../lib/types';
import CodePanel from './CodePanel';
import Controls from './Controls';
import StepsPanel from './StepsPanel';
import VizCanvas from './VizCanvas';

type PatternLayoutProps = {
  pattern: PatternDefinition;
};

export default function PatternLayout({ pattern }: PatternLayoutProps) {
  const [presetIndex, setPresetIndex] = useState(0);
  const [randomSeed, setRandomSeed] = useState(1);
  const useRandom = presetIndex === pattern.presets.length;

  const input = useMemo(() => {
    return useRandom ? pattern.randomInput(randomSeed) : pattern.presets[presetIndex].input;
  }, [pattern, presetIndex, randomSeed, useRandom]);

  const steps = useMemo(() => pattern.makeSteps(input), [pattern, input]);
  const stepper = useStepper(steps);

  const activeStep = stepper.step ?? steps[0];
  const stepLabel =
    steps.length > 0 ? `Step ${stepper.index + 1} of ${steps.length}: ${activeStep?.title ?? ''}` : 'No steps yet.';

  return (
    <div className="fade-in">
      <Link className="nav-pill" to="/">
        {'<- Back to Pattern Picker'}
      </Link>
      <h2 className="section-title">{pattern.title}</h2>
      <p>{pattern.description}</p>

      <div className="meta-grid">
        <div className="panel">
          <h3 className="section-title">How to recognize it</h3>
          <ul>
            {pattern.signals.map((signal) => (
              <li key={signal}>{signal}</li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <h3 className="section-title">Core idea / invariants</h3>
          <ul>
            {pattern.invariants.map((invariant) => (
              <li key={invariant}>{invariant}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="pattern-layout" style={{ marginTop: '24px' }}>
        <div className="panel">
          <h3 className="section-title">Interactive walkthrough</h3>
          <div className="input-panel">
            <label>
              Preset
              <select
                value={presetIndex}
                onChange={(event) => setPresetIndex(Number(event.target.value))}
              >
                {pattern.presets.map((preset, index) => (
                  <option key={preset.label} value={index}>
                    {preset.label}
                  </option>
                ))}
                <option value={pattern.presets.length}>Random (seeded)</option>
              </select>
            </label>
            {useRandom && (
              <button
                className="button secondary"
                type="button"
                onClick={() => setRandomSeed((seed) => seed + 1)}
              >
                New seed
              </button>
            )}
            <pre>{JSON.stringify(input, null, 2)}</pre>
          </div>
          <VizCanvas state={activeStep?.state ?? {}} />
          <Controls controls={stepper.controls} />
          <p>{stepLabel}</p>
        </div>

        <div className="panel">
          <h3 className="section-title">Steps</h3>
          <StepsPanel steps={steps} activeIndex={stepper.index} onSelect={stepper.controls.seek} />
          <h3 className="section-title">Python template</h3>
          <CodePanel code={pattern.pythonCode} step={activeStep} />
        </div>
      </div>

      <div className="meta-grid" style={{ marginTop: '24px' }}>
        <div className="panel">
          <h3 className="section-title">Complexity</h3>
          <p>
            Time: {pattern.complexity.time} | Space: {pattern.complexity.space}
          </p>
          <h3 className="section-title">Common pitfalls</h3>
          <ul>
            {pattern.pitfalls.map((pitfall) => (
              <li key={pitfall}>{pitfall}</li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <h3 className="section-title">Example problems</h3>
          <ul>
            {pattern.exampleProblems.map((problem) => (
              <li key={problem.title}>
                {problem.url ? (
                  <a href={problem.url} target="_blank" rel="noreferrer">
                    {problem.title}
                  </a>
                ) : (
                  problem.title
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
