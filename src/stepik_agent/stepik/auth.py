import hmac
import json
import os
import socket
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

from stepik_agent.core.errors import OAuthError

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


@dataclass(frozen=True, slots=True)
class OAuthToken:
    access_token: str
    token_type: str
    obtained_at: str
    expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> "OAuthToken":
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise OAuthError("Stepik token response has no access_token")

        token_type = payload.get("token_type", "Bearer")
        if not isinstance(token_type, str) or not token_type:
            raise OAuthError("Stepik token response has an invalid token_type")

        expires_in_value = payload.get("expires_in")
        expires_in: int | None = None
        if expires_in_value is not None:
            try:
                expires_in = int(expires_in_value)
            except (TypeError, ValueError) as error:
                raise OAuthError("Stepik token response has an invalid expires_in") from error
            if expires_in <= 0:
                raise OAuthError("Stepik token response has a non-positive expires_in")

        refresh_token = payload.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise OAuthError("Stepik token response has an invalid refresh_token")

        scope = payload.get("scope")
        if scope is not None and not isinstance(scope, str):
            raise OAuthError("Stepik token response has an invalid scope")

        return cls(
            access_token=access_token,
            token_type=token_type,
            obtained_at=datetime.now(UTC).isoformat(),
            expires_in=expires_in,
            refresh_token=refresh_token,
            scope=scope,
        )


class TokenStore:
    """Store OAuth tokens atomically in an owner-readable file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def exists(self) -> bool:
        return self.path.is_file() and not self.path.is_symlink()

    def save(self, token: OAuthToken) -> None:
        if self.path.is_symlink():
            raise OAuthError("Stepik token path must not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(file_descriptor, 0o600)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as token_file:
                json.dump(asdict(token), token_file, ensure_ascii=False, indent=2)
                token_file.write("\n")
                token_file.flush()
                os.fsync(token_file.fileno())
            os.replace(temporary_path, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(self) -> OAuthToken:
        self._validate_private_file()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise OAuthError("Stepik token file does not exist") from error
        except (OSError, json.JSONDecodeError) as error:
            raise OAuthError("Stepik token file cannot be read") from error
        if not isinstance(payload, dict):
            raise OAuthError("Stepik token file must contain a JSON object")
        try:
            return OAuthToken(**payload)
        except TypeError as error:
            raise OAuthError("Stepik token file has an invalid schema") from error

    def _validate_private_file(self) -> None:
        if self.path.is_symlink():
            raise OAuthError("Stepik token path must not be a symbolic link")
        try:
            file_stat = self.path.stat()
        except OSError as error:
            raise OAuthError("Stepik token file cannot be inspected") from error
        if not stat.S_ISREG(file_stat.st_mode):
            raise OAuthError("Stepik token path must be a regular file")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise OAuthError("Stepik token file permissions are too broad; expected 0600")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise OAuthError("Stepik token file must be owned by the current user")


class StepikOAuthClient:
    def __init__(self, *, base_url: str, client_id: str, client_secret: str) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._client_id = client_id
        self._client_secret = client_secret

    def authorization_url(self, *, redirect_uri: str, state: str) -> str:
        if not state:
            raise ValueError("state must not be empty")
        endpoint = urljoin(self._base_url, "oauth2/authorize/")
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self._client_id,
                "redirect_uri": redirect_uri,
                "state": state,
            }
        )
        return f"{endpoint}?{query}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        timeout_seconds: float = 30.0,
    ) -> OAuthToken:
        import httpx

        endpoint = urljoin(self._base_url, "oauth2/token/")
        try:
            response = httpx.post(
                endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                auth=httpx.BasicAuth(self._client_id, self._client_secret),
                headers={"Accept": "application/json"},
                timeout=timeout_seconds,
            )
        except httpx.RequestError as error:
            raise OAuthError("Could not exchange the Stepik authorization code") from error
        if response.is_error:
            raise OAuthError(f"Stepik token endpoint returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as error:
            raise OAuthError("Stepik token endpoint returned non-JSON data") from error
        if not isinstance(payload, dict):
            raise OAuthError("Stepik token endpoint returned an invalid payload")
        return OAuthToken.from_response(payload)


def parse_callback_query(query: str, *, expected_state: str) -> str:
    parameters = parse_qs(query, keep_blank_values=True)
    if oauth_error := parameters.get("error"):
        error_name = oauth_error[0] or "unknown_error"
        raise OAuthError(f"Stepik denied authorization: {error_name}")

    states = parameters.get("state", [])
    if len(states) != 1 or not hmac.compare_digest(states[0], expected_state):
        raise OAuthError("OAuth state mismatch")

    codes = parameters.get("code", [])
    if len(codes) != 1 or not codes[0]:
        raise OAuthError("OAuth callback has no authorization code")
    return codes[0]


def wait_for_callback(
    *,
    redirect_uri: str,
    expected_state: str,
    timeout_seconds: float,
) -> str:
    parsed_redirect = urlsplit(redirect_uri)
    host = (parsed_redirect.hostname or "").lower()
    if parsed_redirect.scheme != "http" or host not in _LOOPBACK_HOSTS:
        raise OAuthError("OAuth callback must use http://127.0.0.1 or http://localhost")
    if parsed_redirect.port is None:
        raise OAuthError("OAuth callback URI must include a port")
    expected_path = parsed_redirect.path or "/"
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_url = urlsplit(self.path)
            if request_url.path != expected_path:
                self.send_error(404)
                return
            try:
                result["code"] = parse_callback_query(
                    request_url.query,
                    expected_state=expected_state,
                )
                status = 200
                message = "Authorization received. You can return to the terminal."
            except OAuthError as error:
                result["error"] = str(error)
                status = 400
                message = "Authorization failed. You can return to the terminal."
            body = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>Stepik Agent</title></head><body><p>{message}</p></body></html>"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer((host, parsed_redirect.port), CallbackHandler)
    except OSError as error:
        if error.errno in {getattr(socket, "EADDRINUSE", 48), 48, 98}:
            raise OAuthError("OAuth callback port is already in use") from error
        raise OAuthError("Could not start the OAuth callback server") from error

    deadline = time.monotonic() + timeout_seconds
    try:
        while not result:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OAuthError("Timed out waiting for the Stepik OAuth callback")
            server.timeout = min(remaining, 1.0)
            server.handle_request()
    finally:
        server.server_close()

    if error_message := result.get("error"):
        raise OAuthError(error_message)
    return result["code"]
