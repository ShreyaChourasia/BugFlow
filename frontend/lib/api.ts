const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const ACCESS_TOKEN_KEY = "bugflow_access_token";
const REFRESH_TOKEN_KEY = "bugflow_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getAccessToken();
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new ApiError(response.status, detail?.detail ?? `Request failed (${response.status})`);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<void> {
  const body = new URLSearchParams({ username: email, password });
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!response.ok) {
    throw new ApiError(response.status, "Incorrect email or password");
  }

  const tokens = await response.json();
  setTokens(tokens.access_token, tokens.refresh_token);
}

export async function logout(): Promise<void> {
  try {
    await apiFetch("/auth/logout", { method: "POST" });
  } finally {
    clearTokens();
  }
}

export type CurrentUser = {
  id: number;
  name: string;
  email: string;
  role: string;
  is_active: boolean;
};

export function fetchCurrentUser(): Promise<CurrentUser> {
  return apiFetch<CurrentUser>("/auth/me");
}
