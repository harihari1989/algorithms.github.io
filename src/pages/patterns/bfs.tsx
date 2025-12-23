import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['bfs'];

export default function BfsPage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
