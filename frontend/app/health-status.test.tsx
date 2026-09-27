// US-50
import { render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import { HealthStatus } from "./health-status";

afterEach(() => {
  vi.unstubAllGlobals();
});

test("shows API reachable once the health endpoint resolves", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" }),
    }),
  );

  render(<HealthStatus />);

  expect(await screen.findByText("API is reachable")).toBeInTheDocument();
});
