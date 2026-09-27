"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";
import { useCurrentUser } from "@/lib/use-current-user";

type Repository = {
  id: number;
  name: string;
  url: string;
  issue_tracker_url: string | null;
  default_branch: string;
  merge_blocking_enabled: boolean;
  risk_threshold: number;
};

export default function RepositoriesPage() {
  const user = useCurrentUser();
  const [repos, setRepos] = useState<Repository[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");

  async function loadRepos() {
    try {
      const data = await apiFetch<Repository[]>("/repositories");
      setRepos(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        clearTokens();
        window.location.href = "/login";
        return;
      }
      setError(err instanceof ApiError ? err.message : "Failed to load repositories");
    }
  }

  useEffect(() => {
    loadRepos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      await apiFetch("/repositories", {
        method: "POST",
        body: JSON.stringify({ name, url }),
      });
      setName("");
      setUrl("");
      await loadRepos();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create repository");
    }
  }

  async function handleToggleMergeBlocking(repo: Repository) {
    await apiFetch(`/repositories/${repo.id}`, {
      method: "PATCH",
      body: JSON.stringify({ merge_blocking_enabled: !repo.merge_blocking_enabled }),
    });
    await loadRepos();
  }

  const isAdmin = user.status === "authenticated" && user.user.role === "admin";

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-semibold">Repositories</h1>
      {error && <p className="text-sm text-red-600">{error}</p>}

      {repos === null && !error && <p className="text-muted-foreground">Loading…</p>}

      {repos !== null && repos.length === 0 && (
        <p className="text-muted-foreground">No repositories registered yet.</p>
      )}

      <ul className="space-y-2">
        {repos?.map((repo) => (
          <li key={repo.id} className="rounded-md border border-border p-3">
            <div className="flex items-center justify-between">
              <div>
                <Link href={`/repositories/${repo.id}`} className="font-medium underline">
                  {repo.name}
                </Link>
                <p className="text-sm text-muted-foreground">{repo.url}</p>
              </div>
              {isAdmin && (
                <button
                  onClick={() => handleToggleMergeBlocking(repo)}
                  className="text-sm underline"
                >
                  Merge blocking: {repo.merge_blocking_enabled ? "on" : "off"}
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>

      {isAdmin && (
        <form onSubmit={handleCreate} className="space-y-3 rounded-md border border-border p-4">
          <h2 className="font-medium">Register a repository</h2>
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full rounded-md border border-border px-3 py-2"
          />
          <input
            placeholder="https://github.com/org/repo"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            required
            className="w-full rounded-md border border-border px-3 py-2"
          />
          <button type="submit" className="rounded-md bg-primary px-3 py-2 text-primary-foreground">
            Register
          </button>
        </form>
      )}
    </div>
  );
}
