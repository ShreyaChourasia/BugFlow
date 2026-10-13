import { defineConfig, devices } from "@playwright/test";

// Phase 10 hardening: Playwright e2e tests for the two main flows
// (pre-merge PR risk, and triage). These run against the real
// docker-compose stack, not a mocked backend — `docker compose up` must
// already be running, with `make seed` and at least one mined+trained
// repository (the README's own quickstart walkthrough), since that's
// exactly the state a real demo/viva runs these against.
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
