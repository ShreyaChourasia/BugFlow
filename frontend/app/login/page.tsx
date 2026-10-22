"use client";

import { useState } from "react";

import { ApiError, login } from "@/lib/api";

// Seeded by scripts/seed_demo.py — picking one here saves typing the address.
const DEMO_ACCOUNTS = [
  ["ML Engineer", "ml-engineer@bugflow.demo"],
  ["QA Engineer", "qa@bugflow.demo"],
  ["Developer", "developer@bugflow.demo"],
  ["Code Reviewer", "reviewer@bugflow.demo"],
  ["Bug Triager", "triager@bugflow.demo"],
  ["Bug Reporter", "reporter@bugflow.demo"],
  ["Engineering Manager", "manager@bugflow.demo"],
  ["System Administrator", "admin@bugflow.demo"],
] as const;
const OTHER = "other";

export default function LoginPage() {
  const [account, setAccount] = useState<string>(DEMO_ACCOUNTS[0][1]);
  const [otherEmail, setOtherEmail] = useState("");
  const email = account === OTHER ? otherEmail : account;
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      // Hard navigation (not router.push): the nav bar and any page-level
      // auth checks only read the token on mount, so they need a full
      // remount to notice we just logged in.
      window.location.href = "/";
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-sm space-y-6">
      <h1 className="text-2xl font-semibold">Log in</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1">
          <label htmlFor="account" className="text-sm font-medium">
            Account
          </label>
          <select
            id="account"
            value={account}
            onChange={(e) => setAccount(e.target.value)}
            className="w-full rounded-md border border-border bg-background px-3 py-2"
          >
            {DEMO_ACCOUNTS.map(([role, address]) => (
              <option key={address} value={address}>
                {role} ({address})
              </option>
            ))}
            <option value={OTHER}>Other (type an email)…</option>
          </select>
        </div>
        {account === OTHER && (
          <div className="space-y-1">
            <label htmlFor="email" className="text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              autoFocus
              value={otherEmail}
              onChange={(e) => setOtherEmail(e.target.value)}
              className="w-full rounded-md border border-border px-3 py-2"
            />
          </div>
        )}
        <div className="space-y-1">
          <label htmlFor="password" className="text-sm font-medium">
            Password
          </label>
          <input
            id="password"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md border border-border px-3 py-2"
          />
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-md bg-primary px-3 py-2 text-primary-foreground disabled:opacity-50"
        >
          {submitting ? "Logging in…" : "Log in"}
        </button>
      </form>
      <p className="text-sm text-muted-foreground">
        Demo accounts are seeded by <code>make seed</code>; their password is{" "}
        <code>bugflow-demo</code>.
      </p>
    </div>
  );
}
