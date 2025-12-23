import { Link } from 'react-router-dom';
import type { PatternDefinition } from '../lib/types';

type PatternCardProps = {
  pattern: PatternDefinition;
};

export default function PatternCard({ pattern }: PatternCardProps) {
  return (
    <Link className="pattern-card" to={`/patterns/${pattern.slug}`}>
      <div>
        <h3>{pattern.title}</h3>
        <p>{pattern.summary}</p>
      </div>
      <div className="chips">
        {pattern.signals.slice(0, 3).map((signal) => (
          <span key={signal} className="chip">
            {signal}
          </span>
        ))}
      </div>
    </Link>
  );
}
