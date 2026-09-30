import time
from pathlib import Path

import httpx
from jose import jwt


def create_app_jwt(app_id: str, private_key_path: str) -> str:
    """Signs a short-lived JWT identifying the GitHub App itself (RS256, as
    GitHub requires) — used only to request an installation access token,
    never for anything else."""
    private_key = Path(private_key_path).read_text()
    now = int(time.time())
    claims = {
        "iat": now - 60,  # allow for clock drift
        "exp": now + (9 * 60),  # GitHub caps this at 10 minutes
        "iss": app_id,
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


def get_installation_token(
    client: httpx.Client,
    api_base_url: str,
    app_id: str,
    private_key_path: str,
    installation_id: str,
) -> str:
    """Exchanges the App's own JWT for a short-lived (1 hour) token scoped to
    one installation — this is what actually authenticates Checks/Issues API
    calls for a specific repository."""
    app_jwt = create_app_jwt(app_id, private_key_path)
    response = client.post(
        f"{api_base_url}/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
        },
    )
    response.raise_for_status()
    token: str = response.json()["token"]
    return token
