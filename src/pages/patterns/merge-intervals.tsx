import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['merge-intervals'];

export default function MergeIntervalsPage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
