import { useParams } from 'react-router-dom';
import LessonLayout from '../../components/LessonLayout';
import { algorithms, dataStructures } from '../../lib/foundations';
import NotFound from '../NotFound';

type FoundationPageProps = {
  kind: 'data-structure' | 'algorithm';
};

export default function FoundationPage({ kind }: FoundationPageProps) {
  const { slug } = useParams();
  const lessons = kind === 'data-structure' ? dataStructures : algorithms;
  const lesson = lessons.find((item) => item.slug === slug);

  if (!lesson) {
    return <NotFound />;
  }

  return <LessonLayout lesson={lesson} />;
}
