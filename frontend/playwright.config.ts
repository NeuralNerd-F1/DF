import { defineConfig, devices } from '@playwright/test';

// E2E runs against a dev server serving the app with the API mocked by MSW (F16 spec §8).
// Browser binaries: `npx playwright install chromium` before the first run.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
