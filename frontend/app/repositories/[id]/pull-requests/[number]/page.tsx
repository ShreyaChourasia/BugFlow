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

type LineRisk = {
  id: number;
  file_path: string;
  line_no: number;
  code: string;
  risk_score: number;
  rank: number;
  reason: string | null;
  marked_false_alarm: boolean;
};

type PullRequestRisk = {
  probability: number;
  calibrated_probability: number;
  confidence: number;
  risk_level: string;
  explanation_text: string;
  factors: Factor[];
  line_risk_status: "not_applicable" | "unavailable" | "available";
  lines: LineRisk[];
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

function DiffViewer({
  lines,
  onFalseAlarm,
}: {
  lines: LineRisk[];
  onFalseAlarm: (lineId: number) => void;
}) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  function toggle(lineId: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(lineId)) next.delete(lineId);
      else next.add(lineId);
      return next;
    });
  }

  const byFile = new Map<string, LineRisk[]>();
  for (const line of lines) {
    byFile.set(line.file_path, [...(byFile.get(line.file_path) ?? []), line]);
  }

  return (
    <div className="space-y-3">
      {[...byFile.entries()].map(([filePath, fileLines]) => (
        <div key={filePath} className="rounded-md border border-border">
          <div className="border-b border-border bg-muted/50 px-3 py-1 font-mono text-xs">
            {filePath}
          </div>
          <ul>
            {fileLines.map((line) => (
              <li key={line.id} className="border-b border-border last:border-b-0">
                <button
                  onClick={() => toggle(line.id)}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left font-mono text-sm hover:bg-muted/30"
                >
                  <span>
                    <span className="text-muted-foreground">
                      #{line.rank} L{line.line_no}
                    </span>{" "}
                    {line.code}
                  </span>
                  <span className="whitespace-nowrap text-xs text-muted-foreground">
                    {(line.risk_score * 100).toFixed(0)}%
                  </span>
                </button>
                {expanded.has(line.id) && (
                  <div className="space-y-2 bg-muted/20 px-3 py-2 text-sm">
                    {line.reason && (
                      <p className="text-muted-foreground">Flagged for: {line.reason}</p>
                    )}
                    {line.marked_false_alarm ? (
                      <p className="text-xs text-muted-foreground">Marked as a false alarm.</p>
                    ) : (
                      <button
                        onClick={() => onFalseAlarm(line.id)}
                        className="text-xs text-red-600 underline"
                      >
                        Mark as false alarm
                      </button>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

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

  async function handleFalseAlarm(lineId: number) {
    await apiFetch(`/line-risks/${lineId}/false-alarm`, { method: "POST" });
    setPr((prev) =>
      prev && prev.risk
        ? {
            ...prev,
            risk: {
              ...prev.risk,
              lines: prev.risk.lines.map((line) =>
                line.id === lineId ? { ...line, marked_false_alarm: true } : line,
              ),
            },
          }
        : prev,
    );
  }

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

      {pr?.risk?.line_risk_status === "not_applicable" && (
        <p className="text-sm text-muted-foreground">
          This change did not meet the threshold for line-level analysis.
        </p>
      )}

      {pr?.risk?.line_risk_status === "available" && pr.risk.lines.length > 0 && (
        <div className="space-y-2">
          <h2 className="font-medium">Riskiest lines</h2>
          <DiffViewer lines={pr.risk.lines} onFalseAlarm={handleFalseAlarm} />
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
