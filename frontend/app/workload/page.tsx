"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";
import { DEFAULT_VISIBLE_DEVELOPERS, visibleWorkload, type WorkloadEntry } from "@/lib/workload";

export default function WorkloadPage() {
  const [entries, setEntries] = useState<WorkloadEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const visible = entries ? visibleWorkload(entries, query) : [];

  useEffect(() => {
    apiFetch<WorkloadEntry[]>("/assignments/workload")
      .then(setEntries)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(
          err instanceof ApiError ? err.message : "Failed to load the workload distribution",
        );
      });
  }, []);

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Workload distribution</h1>
        <p className="text-sm text-muted-foreground">
          Current queue depth against capacity, per developer (US-28).
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {entries !== null && entries.length === 0 && (
        <p className="text-muted-foreground">No developers on file yet.</p>
      )}

      {entries && entries.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <input
              type="search"
              aria-label="Search developers"
              placeholder="Search developers"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="rounded-md border border-border px-3 py-1.5 text-sm"
            />
            <span className="text-sm text-muted-foreground">
              {query
                ? `${visible.length} matching`
                : `Top ${Math.min(DEFAULT_VISIBLE_DEVELOPERS, entries.length)} of ${entries.length} by load`}
            </span>
          </div>
          <div className="h-96 rounded-md border border-border p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={visible}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="developer_name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="current_queue_depth" name="Current load" fill="#2563eb" />
                <Bar dataKey="capacity" name="Capacity" fill="#d1d5db" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
