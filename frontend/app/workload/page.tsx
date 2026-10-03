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

type WorkloadEntry = {
  developer_id: number;
  developer_name: string;
  capacity: number;
  current_queue_depth: number;
};

export default function WorkloadPage() {
  const [entries, setEntries] = useState<WorkloadEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        <div className="h-96 rounded-md border border-border p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={entries}>
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
      )}
    </div>
  );
}
