"use client";

import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type MLModel = {
  id: number;
  task: string;
  version: string;
  stage: string;
  metrics: Record<string, unknown> | null;
  data_version: string;
  seed: number;
  trained_at: string;
};

type ShadowComparison = {
  status: "available" | "unavailable";
  champion_version: string | null;
  challenger_version: string | null;
  n_compared: number;
  mean_absolute_difference: number | null;
  risk_level_agreement_rate: number | null;
};

type DriftAlert = {
  id: number;
  model_id: number;
  feature: string;
  psi: number;
  severity: string;
  created_at: string;
  acknowledged: boolean;
};

const RETRAIN_GROUPS = [
  "commit_risk",
  "line_risk",
  "triage",
  "resolver_suitability",
  "resolution_forecast",
];

function STAGE_BADGE(stage: string): string {
  if (stage === "champion") return "bg-green-100 text-green-700";
  if (stage === "challenger") return "bg-amber-100 text-amber-700";
  return "bg-muted text-muted-foreground";
}

export default function ModelsPage() {
  const [models, setModels] = useState<MLModel[] | null>(null);
  const [shadow, setShadow] = useState<ShadowComparison | null>(null);
  const [driftAlerts, setDriftAlerts] = useState<DriftAlert[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  function load() {
    apiFetch<MLModel[]>("/models")
      .then(setModels)
      .catch((err) => {
        if (err instanceof ApiError && err.status === 401) {
          clearTokens();
          window.location.href = "/login";
          return;
        }
        setError(err instanceof ApiError ? err.message : "Failed to load models");
      });
    apiFetch<ShadowComparison>("/models/shadow-comparison")
      .then(setShadow)
      .catch(() => {});
    apiFetch<DriftAlert[]>("/models/drift-alerts")
      .then(setDriftAlerts)
      .catch(() => {});
  }

  useEffect(load, []);

  async function handleRetrain(group: string) {
    setBusy(`retrain-${group}`);
    try {
      await apiFetch("/models/retrain", { method: "POST", body: JSON.stringify({ group }) });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to queue retraining");
    } finally {
      setBusy(null);
    }
  }

  async function handleRollback(modelId: number) {
    setBusy(`rollback-${modelId}`);
    try {
      await apiFetch(`/models/${modelId}/rollback`, { method: "POST" });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to roll back");
    } finally {
      setBusy(null);
    }
  }

  async function handleCheckDrift() {
    setBusy("drift-check");
    try {
      await apiFetch("/models/drift-check", { method: "POST" });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to run the drift check");
    } finally {
      setBusy(null);
    }
  }

  const byTask = new Map<string, MLModel[]>();
  for (const model of models ?? []) {
    byTask.set(model.task, [...(byTask.get(model.task) ?? []), model]);
  }

  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Models</h1>
        <p className="text-sm text-muted-foreground">
          Versions, metrics, drift alerts, retrain, shadow comparison, and rollback (US-41, US-43 to
          US-45).
        </p>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <section className="space-y-2">
        <h2 className="font-medium">Retrain</h2>
        <div className="flex flex-wrap gap-2">
          {RETRAIN_GROUPS.map((group) => (
            <button
              key={group}
              onClick={() => handleRetrain(group)}
              disabled={busy === `retrain-${group}`}
              className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
            >
              {busy === `retrain-${group}` ? "Queuing…" : `Retrain ${group}`}
            </button>
          ))}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="font-medium">Trained models</h2>
        {[...byTask.entries()].map(([task, taskModels]) => (
          <div key={task} className="space-y-1">
            <h3 className="text-sm font-medium text-muted-foreground">{task}</h3>
            <ul className="space-y-1">
              {taskModels.map((model) => (
                <li
                  key={model.id}
                  className="flex items-center justify-between gap-2 rounded-md border border-border p-2 text-sm"
                >
                  <div className="flex items-center gap-2">
                    <span className={`rounded px-2 py-0.5 text-xs ${STAGE_BADGE(model.stage)}`}>
                      {model.stage}
                    </span>
                    <span className="font-mono text-xs">{model.version.slice(0, 12)}</span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(model.trained_at).toLocaleString()}
                    </span>
                  </div>
                  {model.stage !== "champion" && (
                    <button
                      onClick={() => handleRollback(model.id)}
                      disabled={busy === `rollback-${model.id}`}
                      className="shrink-0 rounded-md border border-border px-2 py-1 text-xs disabled:opacity-50"
                    >
                      {busy === `rollback-${model.id}` ? "Rolling back…" : "Make champion"}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
        {models !== null && models.length === 0 && (
          <p className="text-sm text-muted-foreground">No models trained yet.</p>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="font-medium">Shadow comparison (commit risk)</h2>
        {shadow?.status === "unavailable" && (
          <p className="text-sm text-muted-foreground">
            No challenger is currently shadow-scoring alongside the champion.
          </p>
        )}
        {shadow?.status === "available" && (
          <div className="rounded-md border border-border p-3 text-sm">
            <p>
              Champion {shadow.champion_version?.slice(0, 12)} vs. challenger{" "}
              {shadow.challenger_version?.slice(0, 12)} — compared on {shadow.n_compared} commit(s)
            </p>
            <p className="text-muted-foreground">
              Mean absolute difference: {((shadow.mean_absolute_difference ?? 0) * 100).toFixed(1)}{" "}
              points · risk-level agreement:{" "}
              {((shadow.risk_level_agreement_rate ?? 0) * 100).toFixed(0)}%
            </p>
          </div>
        )}
      </section>

      <section className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="font-medium">Drift alerts (commit risk)</h2>
          <button
            onClick={handleCheckDrift}
            disabled={busy === "drift-check"}
            className="rounded-md border border-border px-3 py-1.5 text-sm disabled:opacity-50"
          >
            {busy === "drift-check" ? "Checking…" : "Run drift check now"}
          </button>
        </div>
        {driftAlerts !== null && driftAlerts.length === 0 && (
          <p className="text-sm text-muted-foreground">No drift detected.</p>
        )}
        {driftAlerts && driftAlerts.length > 0 && (
          <ul className="space-y-1 text-sm">
            {driftAlerts.map((alert) => (
              <li
                key={alert.id}
                className="flex justify-between rounded-md border border-border p-2"
              >
                <span className="font-mono">{alert.feature}</span>
                <span className={alert.severity === "severe" ? "text-red-600" : "text-amber-600"}>
                  {alert.severity} (PSI {alert.psi.toFixed(2)})
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
