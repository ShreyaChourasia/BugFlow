"use client";

import { useEffect, useState } from "react";

import { type CurrentUser, fetchCurrentUser, getAccessToken } from "./api";

type State =
  | { status: "loading" }
  | { status: "anonymous" }
  | { status: "authenticated"; user: CurrentUser };

export function useCurrentUser(): State {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    if (!getAccessToken()) {
      setState({ status: "anonymous" });
      return;
    }
    fetchCurrentUser()
      .then((user) => setState({ status: "authenticated", user }))
      .catch(() => setState({ status: "anonymous" }));
  }, []);

  return state;
}
