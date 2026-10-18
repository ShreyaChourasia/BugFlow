"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type RiskTrendPoint = { week_start: string; mean_calibrated_probability: number; count: number };
type RiskTrends = { points: RiskTrendPoint[] };

type AcceptanceEntry = {
  decision_type: string;
  total: number;
  accepted: number;
  acceptance_rate: number;
};
type AcceptanceRate = { entries: AcceptanceEntry[]; overall_rate: number | null };

type HotspotEntry = { component: string; defect_count: number };
type Hotspots = { entries: HotspotEntry[] };

type TopRiskFactorEntry = { feature: string; occurrence_count: number; mean_shap_value: number };
type TopRiskFactors = { entries: TopRiskFactorEntry[] };

type Fairness = {
  status: "available" | "unavailable";
  junior_mean_risk: number | null;
  senior_mean_risk: number | null;
  junior_count: number;
  senior_count: number;
  disparity: number | null;
  junior_fpr: number | null;
  senior_fpr: number | null;
  fpr_disparity: number | null;
  model_version: string | null;
};

const pct = (value: number | null) => (value === null ? "n/a" : `${(value * 100).toFixed(0)}%`);

function useAnalytics<T>(path: string): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<T>(path)
      .then(setData)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : `Failed to load ${path}`);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  return { data, error };
}

export default function AnalyticsPage() {
  const riskTrends = useAnalytics<RiskTrends>("/analytics/risk-trends");
  const acceptanceRate = useAnalytics<AcceptanceRate>("/analytics/acceptance-rate");
  const hotspots = useAnalytics<Hotspots>("/analytics/hotspots");
  const topRiskFactors = useAnalytics<TopRiskFactors>("/analytics/top-risk-factors");
  const fairness = useAnalytics<Fairness>("/analytics/fairness");

  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          Project-wide trends across risk scoring, triage, and defect reports (US-37 to US-40,
          NFR-US-11).
        </p>
      </div>

      <section className="space-y-2">
        <h2 className="font-medium">Risk trend (mean calibrated probability per week)</h2>
        {riskTrends.error && <p className="text-sm text-red-600">{riskTrends.error}</p>}
        {riskTrends.data && riskTrends.data.points.length === 0 && (
          <p className="text-sm text-muted-foreground">No risk predictions yet.</p>
        )}
        {riskTrends.data && riskTrends.data.points.length > 0 && (
          <div className="h-56 rounded-md border border-border p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={riskTrends.data.points}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="week_start" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                <Line
                  type="monotone"
                  dataKey="mean_calibrated_probability"
                  stroke="#2563eb"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="font-medium">Recommendation acceptance rate</h2>
        {acceptanceRate.error && <p className="text-sm text-red-600">{acceptanceRate.error}</p>}
        {acceptanceRate.data && acceptanceRate.data.entries.length === 0 && (
          <p className="text-sm text-muted-foreground">No feedback recorded yet.</p>
        )}
        {acceptanceRate.data && acceptanceRate.data.entries.length > 0 && (
          <ul className="space-y-1 text-sm">
            {acceptanceRate.data.entries.map((e) => (
              <li
                key={e.decision_type}
                className="flex justify-between rounded-md border border-border p-2"
              >
                <span>{e.decision_type}</span>
                <span className="text-muted-foreground">
                  {(e.acceptance_rate * 100).toFixed(0)}% ({e.accepted}/{e.total})
                </span>
              </li>
            ))}
            {acceptanceRate.data.overall_rate !== null && (
              <li className="pt-1 text-xs text-muted-foreground">
                Overall: {(acceptanceRate.data.overall_rate * 100).toFixed(0)}%
              </li>
            )}
          </ul>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="font-medium">Defect hotspots by component</h2>
        {hotspots.error && <p className="text-sm text-red-600">{hotspots.error}</p>}
        {hotspots.data && hotspots.data.entries.length === 0 && (
          <p className="text-sm text-muted-foreground">No defect reports yet.</p>
        )}
        {hotspots.data && hotspots.data.entries.length > 0 && (
          <div className="h-56 rounded-md border border-border p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hotspots.data.entries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="defect_count" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="font-medium">Top project-wide risk factors</h2>
        {topRiskFactors.error && <p className="text-sm text-red-600">{topRiskFactors.error}</p>}
        {topRiskFactors.data && topRiskFactors.data.entries.length === 0 && (
          <p className="text-sm text-muted-foreground">No scored commits yet.</p>
        )}
        {topRiskFactors.data && topRiskFactors.data.entries.length > 0 && (
          <ul className="space-y-1 text-sm">
            {topRiskFactors.data.entries.map((f) => (
              <li
                key={f.feature}
                className="flex justify-between rounded-md border border-border p-2"
              >
                <span className="font-mono">{f.feature}</span>
                <span className="text-muted-foreground">
                  flagged {f.occurrence_count}x (avg impact {f.mean_shap_value.toFixed(2)})
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="font-medium">
          Fairness (risk and false-positive rate by contributor tenure)
        </h2>
        {fairness.error && <p className="text-sm text-red-600">{fairness.error}</p>}
        {fairness.data?.status === "unavailable" && (
          <p className="text-sm text-muted-foreground">
            No commit-risk champion trained yet, or no tenure data available.
          </p>
        )}
        {fairness.data?.status === "available" && (
          <div className="rounded-md border border-border p-3 text-sm">
            <p>
              Newer contributors (&lt;6mo):{" "}
              {((fairness.data.junior_mean_risk ?? 0) * 100).toFixed(0)}% mean risk (
              {fairness.data.junior_count} commits), false-positive rate{" "}
              {pct(fairness.data.junior_fpr)}
            </p>
            <p>
              Established contributors (&ge;6mo):{" "}
              {((fairness.data.senior_mean_risk ?? 0) * 100).toFixed(0)}% mean risk (
              {fairness.data.senior_count} commits), false-positive rate{" "}
              {pct(fairness.data.senior_fpr)}
            </p>
            {fairness.data.disparity !== null && (
              <p className="mt-1 text-muted-foreground">
                Disparity: {(fairness.data.disparity * 100).toFixed(1)} percentage points
              </p>
            )}
            {fairness.data.fpr_disparity !== null && (
              <p className="text-muted-foreground">
                False-positive rate disparity: {(fairness.data.fpr_disparity * 100).toFixed(1)}{" "}
                percentage points
              </p>
            )}
            <p className="mt-1 text-xs text-muted-foreground">
              False-positive rate: share of clean commits the model flagged high-risk (&ge;70%).
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
