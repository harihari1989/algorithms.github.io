import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
    base: '/algorithms.github.io/',
    plugins: [react()],
});
