import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['two-heaps'];

export default function TwoHeapsPage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
