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

type TriageSuggestion = {
  status: "available" | "abstained" | "decided" | "unavailable";
  severity: string | null;
  priority: string | null;
  severity_confidence: number | null;
  priority_confidence: number | null;
  severity_top_words: string[] | null;
  priority_top_words: string[] | null;
  is_automated: boolean;
};

const SEVERITY_OPTIONS = ["blocker", "critical", "major", "minor", "trivial"];
const PRIORITY_OPTIONS = ["P1", "P2", "P3", "P4", "P5"];

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
  const [triage, setTriage] = useState<TriageSuggestion | null>(null);
  const [draftSeverity, setDraftSeverity] = useState("");
  const [draftPriority, setDraftPriority] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [mergingId, setMergingId] = useState<number | null>(null);
  const [decidingTriage, setDecidingTriage] = useState(false);

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
    apiFetch<TriageSuggestion>(`/defect-reports/${params.id}/triage-suggestion`)
      .then((suggestion) => {
        setTriage(suggestion);
        setDraftSeverity(suggestion.severity ?? SEVERITY_OPTIONS[2]);
        setDraftPriority(suggestion.priority ?? PRIORITY_OPTIONS[2]);
      })
      .catch(() => {
        /* triage suggestion is supplementary; the report still renders without it */
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

  async function handleDecideTriage(severity: string, priority: string) {
    setDecidingTriage(true);
    try {
      await apiFetch(`/defect-reports/${params.id}/triage`, {
        method: "POST",
        body: JSON.stringify({ severity, priority }),
      });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to save the triage decision");
    } finally {
      setDecidingTriage(false);
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

      {triage?.status === "abstained" && (
        <div className="rounded-md border border-border p-4">
          <h2 className="mb-1 font-medium">Triage</h2>
          <p className="text-sm text-muted-foreground">
            This report doesn&apos;t have enough detail to suggest a severity or priority yet — add
            more description to get a suggestion.
          </p>
        </div>
      )}

      {triage?.status === "unavailable" && (
        <div className="rounded-md border border-border p-4">
          <h2 className="mb-1 font-medium">Triage</h2>
          <p className="text-sm text-muted-foreground">No triage model has been trained yet.</p>
        </div>
      )}

      {triage && (triage.status === "available" || triage.status === "decided") && (
        <div className="space-y-3 rounded-md border border-border p-4">
          <div className="flex items-center gap-2">
            <h2 className="font-medium">Triage</h2>
            {triage.status === "available" && (
              <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                Automated
              </span>
            )}
          </div>

          {triage.status === "available" && (
            <p className="text-sm text-muted-foreground">
              Suggested <span className="font-medium">{triage.severity}</span> severity
              {triage.severity_confidence !== null &&
                ` (${(triage.severity_confidence * 100).toFixed(0)}% confidence)`}
              {triage.severity_top_words &&
                triage.severity_top_words.length > 0 &&
                ` — driven by: ${triage.severity_top_words.join(", ")}`}
              . Suggested <span className="font-medium">{triage.priority}</span> priority
              {triage.priority_confidence !== null &&
                ` (${(triage.priority_confidence * 100).toFixed(0)}% confidence)`}
              {triage.priority_top_words &&
                triage.priority_top_words.length > 0 &&
                ` — driven by: ${triage.priority_top_words.join(", ")}`}
              .
            </p>
          )}
          {triage.status === "decided" && (
            <p className="text-sm text-muted-foreground">
              Decided: {triage.severity} severity, {triage.priority} priority.
            </p>
          )}

          {isTriager && triage.status === "available" && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => handleDecideTriage(triage.severity ?? "", triage.priority ?? "")}
                disabled={decidingTriage}
                className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
              >
                Accept
              </button>
              <select
                value={draftSeverity}
                onChange={(e) => setDraftSeverity(e.target.value)}
                className="rounded-md border border-border px-2 py-1.5 text-sm"
              >
                {SEVERITY_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <select
                value={draftPriority}
                onChange={(e) => setDraftPriority(e.target.value)}
                className="rounded-md border border-border px-2 py-1.5 text-sm"
              >
                {PRIORITY_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <button
                onClick={() => handleDecideTriage(draftSeverity, draftPriority)}
                disabled={decidingTriage}
                className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
              >
                Change &amp; save
              </button>
            </div>
          )}
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
