import type { StepperControls } from '../lib/runner';

type ControlsProps = {
  controls: StepperControls;
};

export default function Controls({ controls }: ControlsProps) {
  const speedLabel = `${(1000 / controls.speedMs).toFixed(1)}x`;

  return (
    <div className="controls">
      <button className="button" type="button" onClick={controls.togglePlay}>
        {controls.isPlaying ? 'Pause' : 'Play'}
      </button>
      <button className="button secondary" type="button" onClick={controls.prev}>
        Prev
      </button>
      <button className="button secondary" type="button" onClick={controls.next}>
        Next
      </button>
      <button className="button secondary" type="button" onClick={controls.reset}>
        Reset
      </button>
      <label>
        Speed: {speedLabel}
        <input
          type="range"
          min={250}
          max={1500}
          step={100}
          value={controls.speedMs}
          onChange={(event) => controls.setSpeedMs(Number(event.target.value))}
        />
      </label>
    </div>
  );
}
