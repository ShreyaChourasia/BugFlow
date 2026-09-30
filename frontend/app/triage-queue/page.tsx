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

export default function TriageQueuePage() {
  const [reports, setReports] = useState<DefectReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<DefectReport[]>("/defect-reports?status=open&limit=100")
      .then(setReports)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load the triage queue");
      });
  }, []);

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Triage queue</h1>
        <p className="text-sm text-muted-foreground">
          Open reports awaiting triage, newest first. Open a report to see candidate duplicates and
          merge it.
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {reports !== null && reports.length === 0 && (
        <p className="text-muted-foreground">Nothing to triage right now.</p>
      )}

      <ul className="space-y-2">
        {reports?.map((report) => (
          <li key={report.id} className="rounded-md border border-border p-3">
            <Link
              href={`/defect-reports/${report.id}`}
              className="flex items-center justify-between"
            >
              <span className="font-medium underline">
                #{report.id} {report.title}
              </span>
              <span className="text-sm text-muted-foreground">
                {report.severity ?? "unassessed"}
                {report.priority ? ` · ${report.priority}` : ""}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
