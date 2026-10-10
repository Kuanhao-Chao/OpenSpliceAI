import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', timeout: 90000, workers: 1, fullyParallel: false,
  use: { baseURL: 'http://127.0.0.1:4173/OpenSpliceAI/genome/', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }, { name: 'firefox', use: { ...devices['Desktop Firefox'] } }, { name: 'webkit', use: { ...devices['Desktop Safari'] } }],
  webServer: { command: 'npm run dev', url: 'http://127.0.0.1:4173/OpenSpliceAI/genome/', reuseExistingServer: !process.env.CI, timeout: 30000 },
});
