"use client";

import { useEffect, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Status = { state: "loading" } | { state: "ok" } | { state: "error"; message: string };

export function HealthStatus() {
  const [status, setStatus] = useState<Status>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE_URL}/health`)
      .then((res) => {
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) {
          setStatus(
            data.status === "ok"
              ? { state: "ok" }
              : { state: "error", message: "unexpected response" },
          );
        }
      })
      .catch((err: Error) => {
        if (!cancelled) setStatus({ state: "error", message: err.message });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const color =
    status.state === "ok"
      ? "bg-green-500"
      : status.state === "error"
        ? "bg-red-500"
        : "bg-muted-foreground";

  const label =
    status.state === "ok"
      ? "API is reachable"
      : status.state === "error"
        ? `API unreachable: ${status.message}`
        : "Checking API…";

  return (
    <div className="flex items-center gap-2 rounded-md border border-border p-3">
      <span className={`h-2.5 w-2.5 rounded-full ${color}`} />
      <span>{label}</span>
    </div>
  );
}
