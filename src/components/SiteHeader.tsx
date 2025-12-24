import { Link } from 'react-router-dom';

export default function SiteHeader() {
  return (
    <header className="site-header">
      <div>
        <h1 className="site-title">Algorithm Field Guide</h1>
        <p className="site-subtitle">
          Visual walkthroughs of foundational data structures, algorithms, and interview patterns.
        </p>
      </div>
      <nav className="site-nav">
        <Link className="nav-pill" to="/#data-structures">
          Data structures
        </Link>
        <Link className="nav-pill" to="/#algorithms">
          Algorithms
        </Link>
        <Link className="nav-pill" to="/#patterns">
          Patterns
        </Link>
        <a className="nav-pill" href="https://github.com/" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </nav>
    </header>
  );
}
