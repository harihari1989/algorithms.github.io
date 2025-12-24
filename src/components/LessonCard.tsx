import { Link } from 'react-router-dom';
import type { LessonDefinition } from '../lib/types';

type LessonCardProps = {
  lesson: LessonDefinition;
};

export default function LessonCard({ lesson }: LessonCardProps) {
  const routeBase = lesson.kind === 'data-structure' ? '/data-structures' : '/algorithms';
  const kindLabel = lesson.kind === 'data-structure' ? 'Data structure' : 'Algorithm';
  const tags = lesson.sections[0]?.items.slice(0, 3) ?? [];

  return (
    <Link className="lesson-card" to={`${routeBase}/${lesson.slug}`}>
      <div>
        <span className={`kind-pill ${lesson.kind}`}>{kindLabel}</span>
        <h3>{lesson.title}</h3>
        <p>{lesson.summary}</p>
      </div>
      <div className="chips">
        {tags.map((item) => (
          <span key={item} className="chip">
            {item}
          </span>
        ))}
      </div>
    </Link>
  );
}
