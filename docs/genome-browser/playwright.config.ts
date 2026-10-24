import { defineConfig, devices } from '@playwright/test';
const port = Number(process.env.BROWSER_PORT || 4173);
const baseURL = process.env.BROWSER_BASE_URL || `http://127.0.0.1:${port}/OpenSpliceAI/genome/`;
export default defineConfig({
  testDir: './e2e', timeout: 90000, workers: 1, fullyParallel: false,
  use: { baseURL, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }, { name: 'firefox', use: { ...devices['Desktop Firefox'] } }, { name: 'webkit', use: { ...devices['Desktop Safari'] } }],
  webServer: process.env.BROWSER_BASE_URL ? undefined : { command: 'npm run dev', url: baseURL, reuseExistingServer: !process.env.CI, timeout: 30000 },
});
