import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useStepper } from '../lib/runner';
import type { LessonDefinition } from '../lib/types';
import CodePanel from './CodePanel';
import Controls from './Controls';
import StepsPanel from './StepsPanel';
import VizCanvas from './VizCanvas';

type LessonLayoutProps = {
  lesson: LessonDefinition;
};

export default function LessonLayout({ lesson }: LessonLayoutProps) {
  const [presetIndex, setPresetIndex] = useState(0);
  const [randomSeed, setRandomSeed] = useState(1);
  const useRandom = presetIndex === lesson.presets.length;

  const input = useMemo(() => {
    return useRandom ? lesson.randomInput(randomSeed) : lesson.presets[presetIndex].input;
  }, [lesson, presetIndex, randomSeed, useRandom]);

  const steps = useMemo(() => lesson.makeSteps(input), [lesson, input]);
  const stepper = useStepper(steps);

  const activeStep = stepper.step ?? steps[0];
  const stepLabel =
    steps.length > 0 ? `Step ${stepper.index + 1} of ${steps.length}: ${activeStep?.title ?? ''}` : 'No steps yet.';

  const kindLabel = lesson.kind === 'data-structure' ? 'Data structure' : 'Algorithm';

  return (
    <div className="fade-in">
      <Link className="nav-pill" to="/">
        {'<- Back to Explorer'}
      </Link>
      <div className="lesson-title">
        <span className={`kind-pill ${lesson.kind}`}>{kindLabel}</span>
        <h2 className="section-title">{lesson.title}</h2>
      </div>
      <p className="lede">{lesson.description}</p>

      <div className="meta-grid" style={{ marginTop: '24px' }}>
        {lesson.sections.map((section) => (
          <div className="panel" key={section.title}>
            <h3 className="section-title">{section.title}</h3>
            <ul>
              {section.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="lesson-layout" style={{ marginTop: '24px' }}>
        <div className="panel">
          <h3 className="section-title">Interactive walkthrough</h3>
          <div className="input-panel">
            <label>
              Preset
              <select value={presetIndex} onChange={(event) => setPresetIndex(Number(event.target.value))}>
                {lesson.presets.map((preset, index) => (
                  <option key={preset.label} value={index}>
                    {preset.label}
                  </option>
                ))}
                <option value={lesson.presets.length}>Random (seeded)</option>
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
          <h3 className="section-title">Python implementation</h3>
          <CodePanel code={lesson.pythonCode} step={activeStep} />
        </div>
      </div>

      <div className="meta-grid" style={{ marginTop: '24px' }}>
        <div className="panel">
          <h3 className="section-title">Complexity</h3>
          <p>
            Time: {lesson.complexity.time} | Space: {lesson.complexity.space}
          </p>
        </div>
        <div className="panel">
          <h3 className="section-title">Example problems</h3>
          <ul>
            {lesson.exampleProblems.map((problem) => (
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
