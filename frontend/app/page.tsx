import { HealthStatus } from "./health-status";

export default function HomePage() {
  return (
    <div className="max-w-xl space-y-4">
      <h1 className="text-2xl font-semibold">Welcome to BugFlow</h1>
      <p className="text-muted-foreground">
        Phase 0 skeleton. This page calls the FastAPI backend&apos;s <code>/health</code> endpoint
        to confirm the frontend and API can talk to each other.
      </p>
      <HealthStatus />
    </div>
  );
}
