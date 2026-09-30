// C1/Phase 4 performance test: measures wall-clock time from webhook receipt
// to the check resolving (no longer "pending") — the metric the master
// prompt actually asks for, not just the webhook handler's own response
// time (which is deliberately fast and uninteresting on its own).
//
// Usage:
//   k6 run scripts/perf/webhook_p95.js \
//     -e BASE_URL=http://localhost:8000 \
//     -e WEBHOOK_SECRET=<same value as .env's GITHUB_WEBHOOK_SECRET> \
//     -e REPOSITORY_ID=<a repo id> \
//     -e REPO_HTML_URL=<that repo's registered url> \
//     -e HEAD_SHA=<a sha already mined for that repo, with a champion trained> \
//     -e API_EMAIL=admin@bugflow.demo -e API_PASSWORD=bugflow-demo \
//     --vus 5 --duration 30s
//
// Every VU replays a PR "opened" event for the SAME already-mined commit
// under a unique PR number each time — see docs/PROGRESS.md for why (and
// for the honestly-reported numbers from an actual run).

import http from "k6/http";
import crypto from "k6/crypto";
import { check, sleep } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const WEBHOOK_SECRET = __ENV.WEBHOOK_SECRET;
const REPOSITORY_ID = __ENV.REPOSITORY_ID;
const REPO_HTML_URL = __ENV.REPO_HTML_URL;
const HEAD_SHA = __ENV.HEAD_SHA;
const POLL_TIMEOUT_MS = 10000;

if (!WEBHOOK_SECRET || !REPOSITORY_ID || !REPO_HTML_URL || !HEAD_SHA) {
  throw new Error(
    "Set WEBHOOK_SECRET, REPOSITORY_ID, REPO_HTML_URL and HEAD_SHA env vars (see file header)."
  );
}

export const options = {
  vus: Number(__ENV.VUS || 5),
  duration: __ENV.DURATION || "30s",
  thresholds: {
    webhook_to_check_posted_ms: ["p(95)<5000"], // adjust once you have a real baseline
  },
};

const webhookToCheckPosted = new Trend("webhook_to_check_posted_ms", true);

function sign(body) {
  return "sha256=" + crypto.hmac("sha256", WEBHOOK_SECRET, body, "hex");
}

export function setup() {
  const loginRes = http.post(
    `${BASE_URL}/auth/login`,
    `username=${__ENV.API_EMAIL}&password=${__ENV.API_PASSWORD}`,
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
  );
  check(loginRes, { "login succeeded": (r) => r.status === 200 });
  return { token: JSON.parse(loginRes.body).access_token };
}

export default function (data) {
  const prNumber = Math.floor(Math.random() * 1e9);
  const payload = JSON.stringify({
    action: "opened",
    pull_request: {
      number: prNumber,
      title: "k6 perf test PR",
      head: { sha: HEAD_SHA },
      base: { sha: HEAD_SHA },
    },
    repository: { full_name: "perf/test", html_url: REPO_HTML_URL },
  });

  const start = Date.now();
  const webhookRes = http.post(`${BASE_URL}/webhooks/github`, payload, {
    headers: {
      "Content-Type": "application/json",
      "X-GitHub-Event": "pull_request",
      "X-Hub-Signature-256": sign(payload),
    },
  });
  check(webhookRes, { "webhook accepted (202)": (r) => r.status === 202 });

  let status = "pending";
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (status === "pending" && Date.now() < deadline) {
    sleep(0.1);
    const detail = http.get(
      `${BASE_URL}/repositories/${REPOSITORY_ID}/pull-requests/${prNumber}`,
      { headers: { Authorization: `Bearer ${data.token}` } }
    );
    if (detail.status === 200) {
      status = JSON.parse(detail.body).check_status;
    }
  }

  webhookToCheckPosted.add(Date.now() - start);
  check(null, { "check resolved within timeout (C1: never stuck partial)": () => status !== "pending" });
}
