import { Link } from 'react-router-dom';

export default function SiteHeader() {
  return (
    <header className="site-header">
      <div>
        <h1 className="site-title">Algorithm Pattern Atlas</h1>
        <p className="site-subtitle">
          Visual, step-by-step guides for the 16 core interview patterns.
        </p>
      </div>
      <nav className="site-nav">
        <Link className="nav-pill" to="/">
          Pattern Picker
        </Link>
        <a className="nav-pill" href="https://github.com/" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </nav>
    </header>
  );
}
