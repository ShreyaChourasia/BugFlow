"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type Repository = {
  id: number;
  name: string;
  url: string;
};

type MiningRun = {
  id: number;
  repository_id: number;
  status: string;
  checkpoint: { last_sha?: string; processed?: number } | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
};

const STATUS_COLOR: Record<string, string> = {
  completed: "text-green-600",
  running: "text-blue-600",
  pending: "text-muted-foreground",
  failed: "text-red-600",
};

export default function RepositoryDetailPage() {
  const params = useParams<{ id: string }>();
  const repositoryId = params.id;

  const [repo, setRepo] = useState<Repository | null>(null);
  const [runs, setRuns] = useState<MiningRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    try {
      const [repoData, runsData] = await Promise.all([
        apiFetch<Repository>(`/repositories/${repositoryId}`),
        apiFetch<MiningRun[]>(`/repositories/${repositoryId}/mining-runs`),
      ]);
      setRepo(repoData);
      setRuns(runsData);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        clearTokens();
        window.location.href = "/login";
        return;
      }
      setError(err instanceof ApiError ? err.message : "Failed to load repository");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repositoryId]);

  async function handleStartMining() {
    setError(null);
    try {
      await apiFetch(`/repositories/${repositoryId}/mining-runs`, { method: "POST" });
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to start mining");
    }
  }

  async function handleResume(runId: number) {
    setError(null);
    try {
      await apiFetch(`/repositories/${repositoryId}/mining-runs/${runId}/resume`, {
        method: "POST",
      });
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to resume mining run");
    }
  }

  if (!repo && !error) return <p className="text-muted-foreground">Loading…</p>;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">{repo?.name}</h1>
        <p className="text-sm text-muted-foreground">{repo?.url}</p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex items-center justify-between">
        <h2 className="font-medium">Mining runs</h2>
        <button
          onClick={handleStartMining}
          className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground"
        >
          Start mining
        </button>
      </div>

      {runs !== null && runs.length === 0 && (
        <p className="text-muted-foreground">No mining runs yet.</p>
      )}

      <ul className="space-y-2">
        {runs?.map((run) => {
          const processed = run.checkpoint?.processed ?? 0;
          return (
            <li key={run.id} className="rounded-md border border-border p-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className={`font-medium ${STATUS_COLOR[run.status] ?? ""}`}>
                  {run.status}
                </span>
                {run.status === "failed" && (
                  <button onClick={() => handleResume(run.id)} className="text-sm underline">
                    Resume
                  </button>
                )}
              </div>
              <p className="text-sm text-muted-foreground">{processed} commit(s) processed</p>
              {run.error && <p className="text-sm text-red-600">{run.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
