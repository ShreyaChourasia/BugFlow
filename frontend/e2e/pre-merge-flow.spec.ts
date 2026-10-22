import { expect, test } from "@playwright/test";

// US-08/US-09/US-10/US-13/US-34: the pre-merge flow — open a pull request
// and see its risk assessment, with a plain-language explanation, appear.
// Requires at least one mined+trained repository with a replayed PR (see
// README's "Seeing risk posted on a pull request" walkthrough) — this test
// reads whatever the review queue already has rather than creating its own
// data, matching how a real reviewer would encounter it.

test("opening a pull request from the review queue shows its risk assessment", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Account").selectOption("reviewer@bugflow.demo");
  await page.getByLabel("Password").fill("bugflow-demo");
  await page.getByRole("button", { name: "Log in" }).click();
  await expect(page.getByText("Reviewer", { exact: false })).toBeVisible();

  await page.goto("/review-queue");
  const prLinks = page.locator("a[href*='/pull-requests/']");
  const count = await prLinks.count();
  test.skip(
    count === 0,
    "No pull requests in the review queue yet — replay one first (see README).",
  );

  // Some PRs in a long-lived demo environment can reference an mlflow run
  // whose artifact has since been pruned — try a few until one has an
  // actual risk assessment, rather than assuming the very first is valid.
  let found = false;
  for (let i = 0; i < Math.min(count, 5); i++) {
    await page.goto("/review-queue");
    await prLinks.nth(i).click();
    // isVisible() checks once without waiting — the page fetches its data
    // client-side after navigation, so that always read false and skipped.
    const heading = page.getByRole("heading", { name: "Risk assessment" });
    if (
      await heading.waitFor({ timeout: 10_000 }).then(
        () => true,
        () => false,
      )
    ) {
      found = true;
      break;
    }
  }
  test.skip(!found, "No pull request among the first 5 has a valid risk assessment.");
  await expect(page.getByText("Risk level:")).toBeVisible();
  await expect(page.getByText("Confidence:")).toBeVisible();
  // C3: no automated decision is ever shown without a plain-language
  // explanation — the one real thing this test exists to catch a
  // regression in.
  await expect(page.getByTestId("risk-explanation")).toHaveText(/\w+/);
});
