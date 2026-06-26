from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
QUERY = os.getenv("SMOKE_QUERY", "How to setup VPN?")
DEMO_USERS = [
    ("public@demo.com", "public123", 0),
    ("internal@demo.com", "internal123", 1),
    ("restricted@demo.com", "restricted123", 2),
]


def api(path: str, payload: dict[str, Any] | None = None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(f"{BASE_URL}{path}", data=data, headers=headers)

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"{path} failed with {exc.code}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"{path} failed: {exc}") from exc


def login(email: str, password: str) -> str:
    response = api("/auth/login", {"email": email, "password": password})
    token = response.get("access_token")
    if not token:
        raise SystemExit(f"login returned no token for {email}")
    return str(token)


def chat(token: str):
    return api(
        "/chat",
        {"query": QUERY, "top_k": 5, "mode": "citations_only"},
        token=token,
    )


def main() -> None:
    health = api("/health")
    if not health.get("ok"):
        raise SystemExit(f"health check failed: {health}")

    for email, password, max_level in DEMO_USERS:
        token = login(email, password)
        response = chat(token)
        results = response.get("results") or []

        blocked = [
            result
            for result in results
            if result.get("citation", {}).get("access_level", 0) > max_level
        ]
        if blocked:
            raise SystemExit(f"{email} received sources above level {max_level}")

        print(f"{email}: ok ({len(results)} sources)")


if __name__ == "__main__":
    main()
