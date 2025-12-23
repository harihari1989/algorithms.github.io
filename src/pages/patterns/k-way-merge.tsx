import PatternLayout from '../../components/PatternLayout';
import { patternBySlug } from '../../lib/patterns';

const pattern = patternBySlug['k-way-merge'];

export default function KWayMergePage() {
  if (!pattern) {
    return null;
  }
  return <PatternLayout pattern={pattern} />;
}
