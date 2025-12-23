import { useMemo, useState } from 'react';
import PatternCard from '../components/PatternCard';
import { patterns } from '../lib/patterns';

export default function Home() {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const term = query.trim().toLowerCase();
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
  }, [query]);

  return (
    <div className="fade-in">
      <section className="hero">
        <div className="hero-card">
          <h2 className="section-title">Pattern Picker</h2>
          <p>
            Jump into any of the 16 core interview patterns. Each guide pairs visual steps
            with Python templates and common pitfalls.
          </p>
          <input
            className="search-input"
            type="search"
            placeholder="Search patterns, signals, invariants..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <div className="hero-card">
          <h2 className="section-title">How to Use This Site</h2>
          <ul>
            <li>Pick a pattern and skim the signals + invariants.</li>
            <li>Run the stepper to see pointer movement and data structure changes.</li>
            <li>Copy the Python template and adapt it to the example problems.</li>
          </ul>
        </div>
      </section>
      <section>
        <div className="pattern-grid">
          {filtered.map((pattern) => (
            <PatternCard key={pattern.slug} pattern={pattern} />
          ))}
        </div>
        <p className="footer-note">
          Can&apos;t find what you need? The patterns cover 90%+ of interview problems.
        </p>
      </section>
    </div>
  );
}
