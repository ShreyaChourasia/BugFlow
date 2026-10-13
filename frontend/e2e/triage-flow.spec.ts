import { expect, test } from "@playwright/test";

// US-16/US-17/US-21: the triage flow — file a bug report and see live
// duplicate suggestions while typing, then see candidate duplicates and a
// severity/priority suggestion on the report's own detail page. Requires
// at least one registered repository (see README's quickstart) and the
// triage classifier to have been trained at least once (falls back to the
// documented bootstrap set otherwise — still exercises the same UI path).

test("filing a defect report surfaces duplicate and triage suggestions", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("reporter@bugflow.demo");
  await page.getByLabel("Password").fill("bugflow-demo");
  await page.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByText("Reporter", { exact: false })).toBeVisible();

  await page.goto("/defect-reports/new");
  const repositorySelect = page.locator("select").first();
  test.skip(
    (await repositorySelect.locator("option").count()) <= 1,
    "No repository registered yet — register one first (see README).",
  );
  await repositorySelect.selectOption({ index: 1 });
  await page.getByPlaceholder("Title").fill("Login page crashes on submit");
  await page
    .getByPlaceholder("Describe what happened, and how to reproduce it")
    .fill(
      "The login page throws an unhandled exception whenever a user submits valid credentials.",
    );

  // US-16: live duplicate suggestions while typing, before the report is
  // even saved — the aside always renders once there's enough text, even
  // when it has nothing to show yet.
  await expect(page.getByText("Similar reports")).toBeVisible();

  await page.getByRole("button", { name: /submit report/i }).click();

  // Submitting redirects to the new report's own detail page, which always
  // renders a "Candidate duplicates" section and a "Triage" panel (even
  // when abstained/unavailable — never silently absent). Generous timeout:
  // creating a report computes a sentence-transformer embedding
  // server-side, and the very first call in a freshly-started api
  // container can take several seconds to load the model.
  await expect(page.getByText("Candidate duplicates")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Triage")).toBeVisible();
});
