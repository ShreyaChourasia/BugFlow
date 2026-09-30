"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type PullRequest = {
  id: number;
  number: number;
  title: string;
  status: string;
  check_status: string;
  check_conclusion: string | null;
};

const CONCLUSION_COLOR: Record<string, string> = {
  success: "text-green-600",
  failure: "text-red-600",
  neutral: "text-muted-foreground",
};

export default function PullRequestListPage() {
  const params = useParams<{ id: string }>();
  const repositoryId = params.id;

  const [pullRequests, setPullRequests] = useState<PullRequest[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<PullRequest[]>(`/repositories/${repositoryId}/pull-requests`)
      .then(setPullRequests)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load pull requests");
      });
  }, [repositoryId]);

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-semibold">Pull requests</h1>
      {error && <p className="text-sm text-red-600">{error}</p>}

      {pullRequests !== null && pullRequests.length === 0 && (
        <p className="text-muted-foreground">
          No pull requests yet. Open one on a connected GitHub repo, or replay one offline with{" "}
          <code>scripts/replay_pr_events.py</code>.
        </p>
      )}

      <ul className="space-y-2">
        {pullRequests?.map((pr) => (
          <li key={pr.id} className="rounded-md border border-border p-3">
            <Link
              href={`/repositories/${repositoryId}/pull-requests/${pr.number}`}
              className="flex items-center justify-between"
            >
              <span className="font-medium underline">
                #{pr.number} {pr.title}
              </span>
              <span
                className={
                  pr.check_status === "pending"
                    ? "text-sm text-muted-foreground"
                    : `text-sm font-medium ${CONCLUSION_COLOR[pr.check_conclusion ?? ""] ?? ""}`
                }
              >
                {pr.check_status === "pending" ? "pending" : pr.check_conclusion}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
