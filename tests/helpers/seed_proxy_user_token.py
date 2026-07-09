#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _optional_env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _json_request(
    url: str,
    *,
    method: str = "POST",
    headers: Dict[str, str] | None = None,
    body: Dict[str, Any] | None = None,
    form: Dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    req_headers = dict(headers or {})
    data: bytes | None = None
    if form is not None:
        data = urlencode(form).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = Request(url=url, method=method.upper(), headers=req_headers, data=data)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {payload}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _to_webapp_base_url(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"RASA_PROXY_URL is not a valid absolute URL: {proxy_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _acquire_bearer_token(timeout: float) -> tuple[str, int | None]:
    direct_token = _optional_env("DIRECT_BEARER_TOKEN")
    if direct_token:
        return direct_token, None

    token_url = _required_env("KEYCLOAK_TOKEN_URL")
    client_id = _required_env("KEYCLOAK_CLIENT_ID")
    client_secret = _optional_env("KEYCLOAK_CLIENT_SECRET")
    username = _required_env("KEYCLOAK_USERNAME")
    password = _required_env("KEYCLOAK_PASSWORD")
    scope = _optional_env("KEYCLOAK_SCOPE", "openid profile email")

    form: Dict[str, str] = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
    }
    if client_secret:
        form["client_secret"] = client_secret
    if scope:
        form["scope"] = scope

    token_payload = _json_request(token_url, method="POST", form=form, timeout=timeout)
    access_token = str(token_payload.get("access_token") or "").strip()
    if not access_token:
        raise RuntimeError(
            "Token endpoint did not return access_token; "
            f"payload keys={sorted(token_payload.keys())}"
        )

    expires_in_raw = token_payload.get("expires_in")
    expires_in = int(expires_in_raw) if isinstance(expires_in_raw, (int, float)) else None
    access_token_expires_at = int(time.time() * 1000) + expires_in * 1000 if expires_in and expires_in > 0 else None
    return access_token, access_token_expires_at


def main() -> int:
    try:
        timeout = float(_optional_env("SEED_TIMEOUT_SECONDS", "30") or "30")
        sender_id = _optional_env("SEED_SENDER_ID", "external-e2e-user")
        if not sender_id:
            raise RuntimeError("SEED_SENDER_ID must not be empty")

        proxy_url = _required_env("RASA_PROXY_URL")
        action_server_token = _required_env("ACTION_SERVER_TOKEN")

        access_token, access_token_expires_at = _acquire_bearer_token(timeout)

        webapp_base = _to_webapp_base_url(proxy_url)
        seed_url = f"{webapp_base}/api/dev/seed-user-token"

        payload: Dict[str, Any] = {
            "senderId": sender_id,
            "accessToken": access_token,
        }
        if access_token_expires_at is not None:
            payload["accessTokenExpiresAt"] = access_token_expires_at

        response = _json_request(
            seed_url,
            method="POST",
            headers={"x-action-server-token": action_server_token},
            body=payload,
            timeout=timeout,
        )

        if response.get("ok") is not True:
            raise RuntimeError(f"Seed endpoint returned unexpected response: {response}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "seededSenderId": response.get("senderId"),
                    "seededVia": seed_url,
                }
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
