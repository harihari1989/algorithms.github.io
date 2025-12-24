import { useMemo, useState } from 'react';
import LessonCard from '../components/LessonCard';
import PatternCard from '../components/PatternCard';
import { algorithms, dataStructures } from '../lib/foundations';
import { patterns } from '../lib/patterns';

export default function Home() {
  const [query, setQuery] = useState('');

  const term = query.trim().toLowerCase();
  const matches = (parts: string[]) => parts.join(' ').toLowerCase().includes(term);

  const filteredStructures = useMemo(() => {
    if (!term) {
      return dataStructures;
    }
    return dataStructures.filter((lesson) =>
      matches([
        lesson.title,
        lesson.summary,
        lesson.description,
        ...lesson.sections.map((section) => section.title),
        ...lesson.sections.flatMap((section) => section.items),
      ])
    );
  }, [term]);

  const filteredAlgorithms = useMemo(() => {
    if (!term) {
      return algorithms;
    }
    return algorithms.filter((lesson) =>
      matches([
        lesson.title,
        lesson.summary,
        lesson.description,
        ...lesson.sections.map((section) => section.title),
        ...lesson.sections.flatMap((section) => section.items),
      ])
    );
  }, [term]);

  const filtered = useMemo(() => {
    if (!term) {
      return patterns;
    }
    return patterns.filter((pattern) => {
      const haystack = [
        pattern.title,
        pattern.summary,
        pattern.description,
        ...pattern.signals,
        ...pattern.invariants,
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(term);
    });
  }, [term]);

  const totalLessons = dataStructures.length + algorithms.length;

  return (
    <div className="fade-in">
      <section className="hero">
        <div className="hero-intro">
          <p className="eyebrow">Foundations + patterns</p>
          <h2 className="hero-title">Algorithms you can see, not just memorize.</h2>
          <p className="hero-subtitle">
            Walk through the data structures and algorithms that power interviews and real systems.
            Each guide includes a visual stepper, Python implementation, and clarity on tradeoffs.
          </p>
          <input
            className="search-input"
            type="search"
            placeholder="Search data structures, algorithms, patterns..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <div className="hero-metrics">
            <div>
              <strong>{totalLessons}</strong>
              <span>Foundations</span>
            </div>
            <div>
              <strong>{patterns.length}</strong>
              <span>Patterns</span>
            </div>
            <div>
              <strong>Interactive</strong>
              <span>Visual steps</span>
            </div>
          </div>
        </div>
        <div className="hero-stack">
          <div className="hero-card">
            <h3>Data structures</h3>
            <p>Arrays, lists, stacks, queues, trees, heaps, and graphs with live state changes.</p>
          </div>
          <div className="hero-card">
            <h3>Algorithms</h3>
            <p>Searches, sorts, and traversals with clear invariants and runtime guides.</p>
          </div>
          <div className="hero-card">
            <h3>Interview patterns</h3>
            <p>16 problem patterns with signals, invariants, and step-by-step walkthroughs.</p>
          </div>
        </div>
      </section>

      <section id="data-structures" className="section-block">
        <div className="section-head">
          <div>
            <h2 className="section-title">Data structures</h2>
            <p>Build intuition for how data is stored, linked, and accessed.</p>
          </div>
        </div>
        <div className="card-grid">
          {filteredStructures.map((lesson) => (
            <LessonCard key={lesson.slug} lesson={lesson} />
          ))}
        </div>
        {filteredStructures.length === 0 && (
          <p className="empty-state">No data structures match that search.</p>
        )}
      </section>

      <section id="algorithms" className="section-block">
        <div className="section-head">
          <div>
            <h2 className="section-title">Algorithms</h2>
            <p>Understand the flow of decisions, pivots, and queues.</p>
          </div>
        </div>
        <div className="card-grid">
          {filteredAlgorithms.map((lesson) => (
            <LessonCard key={lesson.slug} lesson={lesson} />
          ))}
        </div>
        {filteredAlgorithms.length === 0 && (
          <p className="empty-state">No algorithms match that search.</p>
        )}
      </section>

      <section id="patterns" className="section-block">
        <div className="section-head">
          <div>
            <h2 className="section-title">Interview patterns</h2>
            <p>Reusable patterns that show up across real interview problems.</p>
          </div>
        </div>
        <div className="card-grid">
          {filtered.map((pattern) => (
            <PatternCard key={pattern.slug} pattern={pattern} />
          ))}
        </div>
        {filtered.length === 0 && <p className="empty-state">No patterns match that search.</p>}
        <p className="footer-note">
          Can&apos;t find what you need? Patterns still cover 90%+ of interview problems.
        </p>
      </section>
    </div>
  );
}
