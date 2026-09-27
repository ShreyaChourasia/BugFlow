"use client";

import Link from "next/link";

import { logout } from "@/lib/api";
import { useCurrentUser } from "@/lib/use-current-user";

const REPOSITORY_ROLES = new Set(["admin", "ml_engineer"]);

export function Nav() {
  const state = useCurrentUser();

  async function handleLogout() {
    await logout();
    // Hard navigation so every page/nav re-reads auth state from scratch.
    window.location.href = "/login";
  }

  if (state.status !== "authenticated") {
    return (
      <nav className="flex items-center gap-4 text-sm">
        <Link href="/login">Log in</Link>
      </nav>
    );
  }

  const { user } = state;

  return (
    <nav className="flex items-center gap-4 text-sm">
      <Link href="/">Home</Link>
      {REPOSITORY_ROLES.has(user.role) && <Link href="/repositories">Repositories</Link>}
      {user.role === "admin" && <Link href="/admin/users">Admin · Users</Link>}
      <span className="text-muted-foreground">
        {user.name} ({user.role})
      </span>
      <button onClick={handleLogout} className="text-muted-foreground underline">
        Log out
      </button>
    </nav>
  );
}
