import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const repoBase = '/algorithms.github.io/';

export default defineConfig(({ mode }) => ({
  base: mode === 'production' ? repoBase : '/',
  plugins: [react()],
}));
