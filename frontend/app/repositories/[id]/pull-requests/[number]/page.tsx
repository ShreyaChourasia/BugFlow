"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type Factor = {
  feature: string;
  value: number;
  shap_value: number;
  impact: "increases" | "decreases";
};

type PullRequestRisk = {
  probability: number;
  calibrated_probability: number;
  confidence: number;
  risk_level: string;
  explanation_text: string;
  factors: Factor[];
};

type PullRequestDetail = {
  number: number;
  title: string;
  head_sha: string;
  base_sha: string;
  check_status: string;
  check_conclusion: string | null;
  check_summary: string | null;
  comment_body: string | null;
  risk: PullRequestRisk | null;
};

const CONCLUSION_COLOR: Record<string, string> = {
  success: "text-green-600",
  failure: "text-red-600",
  neutral: "text-muted-foreground",
};

export default function PullRequestDetailPage() {
  const params = useParams<{ id: string; number: string }>();
  const { id: repositoryId, number } = params;

  const [pr, setPr] = useState<PullRequestDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<PullRequestDetail>(`/repositories/${repositoryId}/pull-requests/${number}`)
      .then(setPr)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load pull request");
      });
  }, [repositoryId, number]);

  if (!pr && !error) return <p className="text-muted-foreground">Loading…</p>;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">
          #{pr?.number} {pr?.title}
        </h1>
        <p className="text-sm text-muted-foreground">
          {pr?.head_sha.slice(0, 7)} into {pr?.base_sha.slice(0, 7)}
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {pr && (
        <div className="rounded-md border border-border p-4 space-y-2">
          <div className="flex items-center gap-2">
            <span className="font-medium">Check:</span>
            {pr.check_status === "pending" ? (
              <span className="text-muted-foreground">pending</span>
            ) : (
              <span className={`font-medium ${CONCLUSION_COLOR[pr.check_conclusion ?? ""] ?? ""}`}>
                {pr.check_conclusion}
              </span>
            )}
          </div>
          {pr.check_summary && <p className="text-sm text-muted-foreground">{pr.check_summary}</p>}
        </div>
      )}

      {pr?.risk && (
        <div className="rounded-md border border-border p-4 space-y-3">
          <h2 className="font-medium">Risk assessment</h2>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-muted-foreground">Risk level: </span>
              <span className="font-medium">{pr.risk.risk_level}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Probability: </span>
              {(pr.risk.calibrated_probability * 100).toFixed(0)}%
            </div>
            <div>
              <span className="text-muted-foreground">Confidence: </span>
              {(pr.risk.confidence * 100).toFixed(0)}%
            </div>
          </div>
          <p className="text-sm">{pr.risk.explanation_text}</p>
          <ul className="text-sm space-y-1">
            {pr.risk.factors.map((factor) => (
              <li key={factor.feature} className="text-muted-foreground">
                <span className="font-mono">{factor.feature}</span> = {factor.value} (
                {factor.impact} risk)
              </li>
            ))}
          </ul>
        </div>
      )}

      {pr?.comment_body && (
        <div className="rounded-md border border-border p-4">
          <h2 className="mb-2 font-medium">Posted comment</h2>
          <pre className="whitespace-pre-wrap text-sm">{pr.comment_body}</pre>
        </div>
      )}
    </div>
  );
}
