import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
var repoBase = '/algorithms.github.io/';
export default defineConfig(function (_a) {
    var mode = _a.mode;
    return ({
        base: mode === 'production' ? repoBase : '/',
        plugins: [react()],
    });
});
