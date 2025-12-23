import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['modified-binary-search'];

export default function ModifiedBinarySearchPage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
