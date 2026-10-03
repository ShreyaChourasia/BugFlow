"use client";

import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type MyAssignment = {
  id: number;
  defect_id: number;
  defect_title: string;
  source: string;
  assigned_at: string;
  reason: string;
};

export default function MyAssignmentsPage() {
  const [assignments, setAssignments] = useState<MyAssignment[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [objectingId, setObjectingId] = useState<number | null>(null);
  const [objectionText, setObjectionText] = useState<Record<number, string>>({});
  const [objected, setObjected] = useState<Set<number>>(new Set());

  function load() {
    apiFetch<MyAssignment[]>("/assignments/mine")
      .then(setAssignments)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load your assignments");
      });
  }

  useEffect(load, []);

  async function handleObject(assignmentId: number) {
    const reason = objectionText[assignmentId]?.trim();
    if (!reason) {
      setError("A reason is required to object to an assignment.");
      return;
    }
    setObjectingId(assignmentId);
    try {
      await apiFetch(`/assignments/${assignmentId}/objection`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      });
      setObjected((prev) => new Set(prev).add(assignmentId));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to record the objection");
    } finally {
      setObjectingId(null);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">My assignments</h1>
        <p className="text-sm text-muted-foreground">
          Defects assigned to you, newest first, with the reason behind each one.
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {assignments !== null && assignments.length === 0 && (
        <p className="text-muted-foreground">No assignments yet.</p>
      )}

      <ul className="space-y-2">
        {assignments?.map((assignment) => (
          <li key={assignment.id} className="rounded-md border border-border p-3">
            <a href={`/defect-reports/${assignment.defect_id}`} className="font-medium underline">
              #{assignment.defect_id} {assignment.defect_title}
            </a>
            <p className="text-xs text-muted-foreground">
              {assignment.source} · {new Date(assignment.assigned_at).toLocaleString()}
            </p>
            <p className="mt-1 text-sm">Why me: {assignment.reason}</p>

            {objected.has(assignment.id) ? (
              <p className="mt-2 text-xs text-green-600">Objection recorded.</p>
            ) : (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <input
                  type="text"
                  value={objectionText[assignment.id] ?? ""}
                  onChange={(e) =>
                    setObjectionText((prev) => ({ ...prev, [assignment.id]: e.target.value }))
                  }
                  placeholder="Why doesn't this fit?"
                  className="flex-1 rounded-md border border-border px-2 py-1 text-xs"
                />
                <button
                  onClick={() => handleObject(assignment.id)}
                  disabled={objectingId === assignment.id}
                  className="shrink-0 rounded-md border border-border px-2 py-1 text-xs text-red-600 disabled:opacity-50"
                >
                  {objectingId === assignment.id ? "Objecting…" : "Object"}
                </button>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
