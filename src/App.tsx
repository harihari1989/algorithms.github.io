import { BrowserRouter, Route, Routes } from 'react-router-dom';
import SiteHeader from './components/SiteHeader';
import Home from './pages/Home';
import NotFound from './pages/NotFound';
import BacktrackingPage from './pages/patterns/backtracking';
import BfsPage from './pages/patterns/bfs';
import CyclicSortPage from './pages/patterns/cyclic-sort';
import DfsPage from './pages/patterns/dfs';
import FastSlowPointersPage from './pages/patterns/fast-slow-pointers';
import InPlaceReversalPage from './pages/patterns/in-place-reversal';
import KWayMergePage from './pages/patterns/k-way-merge';
import KnapsackPage from './pages/patterns/knapsack';
import MergeIntervalsPage from './pages/patterns/merge-intervals';
import ModifiedBinarySearchPage from './pages/patterns/modified-binary-search';
import SlidingWindowPage from './pages/patterns/sliding-window';
import SubsetsPage from './pages/patterns/subsets';
import TopKElementsPage from './pages/patterns/top-k-elements';
import TopologicalSortPage from './pages/patterns/topological-sort';
import TwoHeapsPage from './pages/patterns/two-heaps';
import TwoPointersPage from './pages/patterns/two-pointers';
import FoundationPage from './pages/foundations/FoundationPage';

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <div className="app">
        <SiteHeader />
        <main className="main fade-in">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/patterns/sliding-window" element={<SlidingWindowPage />} />
            <Route path="/patterns/two-pointers" element={<TwoPointersPage />} />
            <Route path="/patterns/fast-slow-pointers" element={<FastSlowPointersPage />} />
            <Route path="/patterns/merge-intervals" element={<MergeIntervalsPage />} />
            <Route path="/patterns/cyclic-sort" element={<CyclicSortPage />} />
            <Route path="/patterns/in-place-reversal" element={<InPlaceReversalPage />} />
            <Route path="/patterns/bfs" element={<BfsPage />} />
            <Route path="/patterns/dfs" element={<DfsPage />} />
            <Route path="/patterns/two-heaps" element={<TwoHeapsPage />} />
            <Route path="/patterns/subsets" element={<SubsetsPage />} />
            <Route path="/patterns/modified-binary-search" element={<ModifiedBinarySearchPage />} />
            <Route path="/patterns/top-k-elements" element={<TopKElementsPage />} />
            <Route path="/patterns/k-way-merge" element={<KWayMergePage />} />
            <Route path="/patterns/knapsack" element={<KnapsackPage />} />
            <Route path="/patterns/topological-sort" element={<TopologicalSortPage />} />
            <Route path="/patterns/backtracking" element={<BacktrackingPage />} />
            <Route path="/data-structures/:slug" element={<FoundationPage kind="data-structure" />} />
            <Route path="/algorithms/:slug" element={<FoundationPage kind="algorithm" />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
