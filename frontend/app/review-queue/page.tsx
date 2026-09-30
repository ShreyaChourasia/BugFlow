"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type ReviewQueueItem = {
  repository_id: number;
  repository_name: string;
  number: number;
  title: string;
  check_status: string;
  calibrated_probability: number | null;
  risk_level: string | null;
  lines_added: number | null;
  lines_deleted: number | null;
  files_changed: number | null;
};

const RISK_COLOR: Record<string, string> = {
  high: "text-red-600",
  medium: "text-amber-600",
  low: "text-green-600",
};

export default function ReviewQueuePage() {
  const [items, setItems] = useState<ReviewQueueItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<ReviewQueueItem[]>("/review-queue")
      .then(setItems)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load the review queue");
      });
  }, []);

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Review queue</h1>
        <p className="text-sm text-muted-foreground">
          Open pull requests across every repository, riskiest first.
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {items !== null && items.length === 0 && (
        <p className="text-muted-foreground">No open pull requests right now.</p>
      )}

      <ul className="space-y-2">
        {items?.map((item) => (
          <li
            key={`${item.repository_id}-${item.number}`}
            className="rounded-md border border-border p-3"
          >
            <Link
              href={`/repositories/${item.repository_id}/pull-requests/${item.number}`}
              className="flex items-center justify-between gap-4"
            >
              <div>
                <span className="font-medium underline">
                  {item.repository_name} #{item.number}
                </span>
                <p className="text-sm text-muted-foreground">{item.title}</p>
                {(item.lines_added !== null || item.files_changed !== null) && (
                  <p className="text-xs text-muted-foreground">
                    +{item.lines_added ?? 0} -{item.lines_deleted ?? 0} across{" "}
                    {item.files_changed ?? 0} file(s)
                  </p>
                )}
              </div>
              <span
                className={`text-sm font-medium ${RISK_COLOR[item.risk_level ?? ""] ?? "text-muted-foreground"}`}
              >
                {item.calibrated_probability !== null
                  ? `${(item.calibrated_probability * 100).toFixed(0)}%`
                  : "not scored"}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
