"use client";

import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";
import { useCurrentUser } from "@/lib/use-current-user";

type DefectReport = {
  id: number;
  title: string;
  description: string;
  status: string;
  severity: string | null;
  priority: string | null;
  duplicate_of_id: number | null;
};

type DuplicateCandidate = {
  id: number;
  title: string;
  status: string;
  score: number;
  shared_phrases: string[];
};

type DuplicatesResponse = {
  status: "ready" | "rebuilding";
  results: DuplicateCandidate[];
};

// US-18: wraps each shared phrase in <mark> wherever it appears in `text`.
function highlightPhrases(text: string, phrases: string[]): React.ReactNode {
  if (phrases.length === 0) return text;
  const pattern = new RegExp(
    `(${phrases.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`,
    "gi",
  );
  return text
    .split(pattern)
    .map((part, i) =>
      phrases.some((p) => p.toLowerCase() === part.toLowerCase()) ? (
        <mark key={i}>{part}</mark>
      ) : (
        <span key={i}>{part}</span>
      ),
    );
}

export default function DefectReportDetailPage() {
  const params = useParams<{ id: string }>();
  const user = useCurrentUser();

  const [report, setReport] = useState<DefectReport | null>(null);
  const [duplicates, setDuplicates] = useState<DuplicatesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mergingId, setMergingId] = useState<number | null>(null);

  const load = useCallback(() => {
    apiFetch<DefectReport>(`/defect-reports/${params.id}`)
      .then(setReport)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load the report");
      });
    apiFetch<DuplicatesResponse>(`/defect-reports/${params.id}/duplicates`)
      .then(setDuplicates)
      .catch(() => {
        /* duplicates are supplementary; the report itself still renders without them */
      });
  }, [params.id]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleMerge(originalId: number) {
    setMergingId(originalId);
    try {
      await apiFetch(`/defect-reports/${params.id}/merge`, {
        method: "POST",
        body: JSON.stringify({ original_id: originalId }),
      });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to merge the report");
    } finally {
      setMergingId(null);
    }
  }

  const isTriager = user.status === "authenticated" && user.user.role === "triager";

  if (!report && !error) return <p className="text-muted-foreground">Loading…</p>;

  return (
    <div className="max-w-2xl space-y-6">
      {error && <p className="text-sm text-red-600">{error}</p>}

      {report && (
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold">{report.title}</h1>
          <p className="text-sm text-muted-foreground">
            Status: {report.status}
            {report.severity && ` · Severity: ${report.severity}`}
            {report.priority && ` · Priority: ${report.priority}`}
          </p>
          {report.duplicate_of_id && (
            <p className="text-sm text-amber-600">
              Marked as a duplicate of{" "}
              <a href={`/defect-reports/${report.duplicate_of_id}`} className="underline">
                #{report.duplicate_of_id}
              </a>
            </p>
          )}
          <p className="whitespace-pre-wrap text-sm">{report.description}</p>
        </div>
      )}

      <div className="space-y-2">
        <h2 className="font-medium">Candidate duplicates</h2>
        {duplicates?.status === "rebuilding" && (
          <p className="text-sm text-muted-foreground">
            The search index is rebuilding — candidates aren&apos;t available right now.
          </p>
        )}
        {duplicates?.status === "ready" && duplicates.results.length === 0 && (
          <p className="text-sm text-muted-foreground">No similar reports found.</p>
        )}
        <ul className="space-y-2">
          {duplicates?.results.map((candidate) => (
            <li key={candidate.id} className="rounded-md border border-border p-3">
              <div className="flex items-center justify-between gap-2">
                <a href={`/defect-reports/${candidate.id}`} className="font-medium underline">
                  {highlightPhrases(candidate.title, candidate.shared_phrases)}
                </a>
                <span className="text-xs text-muted-foreground">
                  {(candidate.score * 100).toFixed(0)}% similar
                </span>
              </div>
              <p className="text-xs text-muted-foreground">{candidate.status}</p>
              {isTriager && report && !report.duplicate_of_id && (
                <button
                  onClick={() => handleMerge(candidate.id)}
                  disabled={mergingId === candidate.id}
                  className="mt-2 text-xs text-red-600 underline disabled:opacity-50"
                >
                  {mergingId === candidate.id ? "Merging…" : "Merge this report into it"}
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
