import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/OpenSpliceAI/genome/',
  cacheDir: process.env.BROWSER_CACHE_DIR || 'node_modules/.vite',
  build: { outDir: 'dist/genome', emptyOutDir: true, target: 'es2022' },
  server: { port: Number(process.env.BROWSER_PORT || 4173), strictPort: true },
  test: { include: ['src/**/*.test.ts'], maxWorkers: 2 },
});
