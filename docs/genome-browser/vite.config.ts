import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/OpenSpliceAI/genome/',
  build: { outDir: 'dist/genome', emptyOutDir: true, target: 'es2022' },
  server: { port: 4173, strictPort: true },
  test: { include: ['src/**/*.test.ts'], maxWorkers: 2 },
});
