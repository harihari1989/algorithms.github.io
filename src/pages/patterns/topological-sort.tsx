import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['topological-sort'];

export default function TopologicalSortPage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
