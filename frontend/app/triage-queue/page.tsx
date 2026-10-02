"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type DefectReport = {
  id: number;
  title: string;
  status: string;
  severity: string | null;
  priority: string | null;
  reported_at: string;
};

type TriageSuggestion = {
  status: "available" | "abstained" | "decided" | "unavailable";
  severity: string | null;
  priority: string | null;
};

export default function TriageQueuePage() {
  const [reports, setReports] = useState<DefectReport[] | null>(null);
  const [suggestions, setSuggestions] = useState<Record<number, TriageSuggestion>>({});
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [accepting, setAccepting] = useState(false);

  useEffect(() => {
    apiFetch<DefectReport[]>("/defect-reports?status=open&limit=100")
      .then(async (loaded) => {
        setReports(loaded);
        // US-21/US-23: fetch each undecided report's suggestion so "bulk
        // accept" has something to accept — bounded by the same limit=100
        // the list itself already applies, each call is a cheap in-process
        // prediction (no network-bound work), so doing them in parallel is
        // fine at this scale.
        const pending = loaded.filter((r) => r.severity === null);
        const results = await Promise.all(
          pending.map((r) =>
            apiFetch<TriageSuggestion>(`/defect-reports/${r.id}/triage-suggestion`)
              .then((suggestion) => [r.id, suggestion] as const)
              .catch(() => null),
          ),
        );
        setSuggestions(
          Object.fromEntries(results.filter((r): r is [number, TriageSuggestion] => r !== null)),
        );
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load the triage queue");
      });
  }, []);

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function acceptSelected() {
    setAccepting(true);
    try {
      await Promise.all(
        [...selected].map((id) => {
          const suggestion = suggestions[id];
          if (!suggestion || suggestion.status !== "available") return Promise.resolve();
          return apiFetch(`/defect-reports/${id}/triage`, {
            method: "POST",
            body: JSON.stringify({ severity: suggestion.severity, priority: suggestion.priority }),
          });
        }),
      );
      const refreshed = await apiFetch<DefectReport[]>("/defect-reports?status=open&limit=100");
      setReports(refreshed);
      setSelected(new Set());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to accept the selected reports");
    } finally {
      setAccepting(false);
    }
  }

  const selectableCount = [...selected].filter(
    (id) => suggestions[id]?.status === "available",
  ).length;

  return (
    <div className="max-w-2xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Triage queue</h1>
          <p className="text-sm text-muted-foreground">
            Open reports awaiting triage, newest first. Open a report to see candidate duplicates
            and merge it.
          </p>
        </div>
        <button
          onClick={acceptSelected}
          disabled={selectableCount === 0 || accepting}
          className="shrink-0 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-50"
        >
          {accepting ? "Accepting…" : `Accept selected (${selectableCount})`}
        </button>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {reports !== null && reports.length === 0 && (
        <p className="text-muted-foreground">Nothing to triage right now.</p>
      )}

      <ul className="space-y-2">
        {reports?.map((report) => {
          const suggestion = suggestions[report.id];
          return (
            <li
              key={report.id}
              className="flex items-center gap-3 rounded-md border border-border p-3"
            >
              {report.severity === null && (
                <input
                  type="checkbox"
                  checked={selected.has(report.id)}
                  onChange={() => toggle(report.id)}
                  disabled={suggestion?.status !== "available"}
                  aria-label={`Select report ${report.id} for bulk accept`}
                />
              )}
              <Link
                href={`/defect-reports/${report.id}`}
                className="flex flex-1 items-center justify-between"
              >
                <span className="font-medium underline">
                  #{report.id} {report.title}
                </span>
                <span className="text-sm text-muted-foreground">
                  {report.severity ??
                    (suggestion?.status === "available"
                      ? `${suggestion.severity} (suggested)`
                      : "unassessed")}
                  {report.priority ? ` · ${report.priority}` : ""}
                  {!report.priority && suggestion?.status === "available"
                    ? ` · ${suggestion.priority} (suggested)`
                    : ""}
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
