"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ApiError, apiFetch, clearTokens } from "@/lib/api";

type User = {
  id: number;
  name: string;
  email: string;
  role: string;
  is_active: boolean;
};

const ROLES = [
  "developer",
  "reviewer",
  "triager",
  "reporter",
  "qa",
  "manager",
  "ml_engineer",
  "admin",
];

export default function AdminUsersPage() {
  const router = useRouter();
  const [users, setUsers] = useState<User[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState(ROLES[0]);

  async function loadUsers() {
    try {
      const data = await apiFetch<User[]>("/users");
      setUsers(data);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        clearTokens();
        window.location.href = "/login";
        return;
      }
      if (err instanceof ApiError && err.status === 403) {
        router.push("/");
        return;
      }
      setError(err instanceof ApiError ? err.message : "Failed to load users");
    }
  }

  useEffect(() => {
    loadUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleCreate(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      await apiFetch("/users", {
        method: "POST",
        body: JSON.stringify({ name, email, password, role }),
      });
      setName("");
      setEmail("");
      setPassword("");
      await loadUsers();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create user");
    }
  }

  async function handleToggleActive(user: User) {
    await apiFetch(`/users/${user.id}`, {
      method: "PATCH",
      body: JSON.stringify({ is_active: !user.is_active }),
    });
    await loadUsers();
  }

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-semibold">Users</h1>
      {error && <p className="text-sm text-red-600">{error}</p>}

      <ul className="space-y-2">
        {users?.map((user) => (
          <li
            key={user.id}
            className="flex items-center justify-between rounded-md border border-border p-3"
          >
            <div>
              <p className="font-medium">{user.name}</p>
              <p className="text-sm text-muted-foreground">
                {user.email} · {user.role}
              </p>
            </div>
            <button onClick={() => handleToggleActive(user)} className="text-sm underline">
              {user.is_active ? "Deactivate" : "Activate"}
            </button>
          </li>
        ))}
      </ul>

      <form onSubmit={handleCreate} className="space-y-3 rounded-md border border-border p-4">
        <h2 className="font-medium">Add a user</h2>
        <input
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          className="w-full rounded-md border border-border px-3 py-2"
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="w-full rounded-md border border-border px-3 py-2"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          className="w-full rounded-md border border-border px-3 py-2"
        />
        <select
          value={role}
          onChange={(e) => setRole(e.target.value)}
          className="w-full rounded-md border border-border px-3 py-2"
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
        <button type="submit" className="rounded-md bg-primary px-3 py-2 text-primary-foreground">
          Create user
        </button>
      </form>
    </div>
  );
}
