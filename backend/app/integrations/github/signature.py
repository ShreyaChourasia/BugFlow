import hashlib
import hmac


def verify_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """Verifies GitHub's `X-Hub-Signature-256` header (HMAC-SHA256 of the raw
    request body, keyed by the webhook secret). Constant-time comparison to
    avoid timing attacks."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)
