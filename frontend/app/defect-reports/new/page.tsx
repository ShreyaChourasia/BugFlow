"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type Repository = {
  id: number;
  name: string;
};

type Suggestion = {
  id: number;
  title: string;
  status: string;
};

type SuggestionsResponse = {
  status: "ready" | "rebuilding";
  results: Suggestion[];
};

const DEBOUNCE_MS = 400;

export default function NewDefectReportPage() {
  const router = useRouter();

  const [repositories, setRepositories] = useState<Repository[] | null>(null);
  const [repositoryId, setRepositoryId] = useState<string>("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [suggestions, setSuggestions] = useState<SuggestionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    apiFetch<Repository[]>("/repositories")
      .then(setRepositories)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
        }
      });
  }, []);

  // US-16: debounced live suggestions as the reporter types — the request
  // fires a moment after typing stops, and never touches (or clears) the
  // form's own text state, so opening a suggestion in a new tab never loses
  // what's been typed so far.
  useEffect(() => {
    const text = `${title} ${description}`.trim();
    if (!text) {
      setSuggestions(null);
      return;
    }
    const timeout = setTimeout(() => {
      apiFetch<SuggestionsResponse>(`/defect-reports/suggestions?text=${encodeURIComponent(text)}`)
        .then(setSuggestions)
        .catch(() => {
          /* live suggestions are a nice-to-have; a failed lookup shouldn't block typing */
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timeout);
  }, [title, description]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const report = await apiFetch<{ id: number }>("/defect-reports", {
        method: "POST",
        body: JSON.stringify({
          repository_id: Number(repositoryId),
          title,
          description,
        }),
      });
      router.push(`/defect-reports/${report.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create the report");
      setSubmitting(false);
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <h1 className="text-2xl font-semibold">Report a bug</h1>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <div className="flex gap-6">
        <form onSubmit={handleSubmit} className="flex-1 space-y-3">
          <select
            value={repositoryId}
            onChange={(e) => setRepositoryId(e.target.value)}
            required
            className="w-full rounded-md border border-border px-3 py-2"
          >
            <option value="" disabled>
              Select a repository
            </option>
            {repositories?.map((repo) => (
              <option key={repo.id} value={repo.id}>
                {repo.name}
              </option>
            ))}
          </select>
          <input
            placeholder="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
            className="w-full rounded-md border border-border px-3 py-2"
          />
          <textarea
            placeholder="Describe what happened, and how to reproduce it"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            required
            rows={8}
            className="w-full rounded-md border border-border px-3 py-2"
          />
          <button
            type="submit"
            disabled={submitting}
            className="rounded-md bg-primary px-3 py-2 text-primary-foreground disabled:opacity-50"
          >
            {submitting ? "Submitting…" : "Submit report"}
          </button>
        </form>

        <aside className="w-72 shrink-0 space-y-2 rounded-md border border-border p-3">
          <h2 className="text-sm font-medium text-muted-foreground">Similar reports</h2>
          {suggestions?.status === "rebuilding" && (
            <p className="text-sm text-muted-foreground">
              The search index is rebuilding — try again shortly.
            </p>
          )}
          {suggestions?.status === "ready" && suggestions.results.length === 0 && (
            <p className="text-sm text-muted-foreground">
              Keep typing — suggestions appear once there&apos;s enough text.
            </p>
          )}
          <ul className="space-y-2">
            {suggestions?.results.map((suggestion) => (
              <li key={suggestion.id} className="text-sm">
                <a
                  href={`/defect-reports/${suggestion.id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="underline"
                >
                  {suggestion.title}
                </a>
                <span className="ml-1 text-xs text-muted-foreground">({suggestion.status})</span>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
