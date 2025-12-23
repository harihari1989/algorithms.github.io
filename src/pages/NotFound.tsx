import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="panel">
      <h2 className="section-title">Page not found</h2>
      <p>The pattern you&apos;re looking for isn&apos;t here yet.</p>
      <Link className="nav-pill" to="/">
        Back to Pattern Picker
      </Link>
    </div>
  );
}
