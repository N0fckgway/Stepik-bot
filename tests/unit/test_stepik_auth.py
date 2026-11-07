import stat
from urllib.parse import parse_qs, urlsplit

import pytest

from stepik_agent.core.errors import OAuthError
from stepik_agent.stepik.auth import (
    OAuthToken,
    StepikOAuthClient,
    TokenStore,
    parse_callback_query,
    wait_for_callback,
)


def test_authorization_url_contains_code_flow_parameters() -> None:
    client = StepikOAuthClient(
        base_url="https://stepik.org",
        client_id="client-id",
        client_secret="client-secret",
    )

    url = client.authorization_url(
        redirect_uri="http://127.0.0.1:8080/callback",
        state="expected-state",
    )

    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "stepik.org"
    assert parsed.path == "/oauth2/authorize/"
    assert parse_qs(parsed.query) == {
        "response_type": ["code"],
        "client_id": ["client-id"],
        "redirect_uri": ["http://127.0.0.1:8080/callback"],
        "state": ["expected-state"],
    }


def test_callback_query_requires_matching_state() -> None:
    code = parse_callback_query("code=abc&state=safe-state", expected_state="safe-state")

    assert code == "abc"

    with pytest.raises(OAuthError, match="state mismatch"):
        parse_callback_query("code=abc&state=wrong", expected_state="safe-state")


def test_callback_rejects_non_loopback_redirect() -> None:
    with pytest.raises(OAuthError, match=r"loopback|127\.0\.0\.1"):
        wait_for_callback(
            redirect_uri="https://example.com/callback",
            expected_state="state",
            timeout_seconds=0.01,
        )


def test_token_store_round_trip_and_permissions(tmp_path) -> None:
    path = tmp_path / "private" / "stepik_token.json"
    token = OAuthToken(
        access_token="access-token",
        token_type="Bearer",
        obtained_at="2026-11-06T00:00:00+00:00",
        expires_in=36000,
        refresh_token="refresh-token",
        scope="read write",
    )
    store = TokenStore(path)

    store.save(token)

    assert store.load() == token
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_invalid_token_response_is_rejected() -> None:
    with pytest.raises(OAuthError, match="access_token"):
        OAuthToken.from_response({"token_type": "Bearer"})


def test_token_store_rejects_broad_permissions(tmp_path) -> None:
    path = tmp_path / "stepik_token.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(OAuthError, match="permissions are too broad"):
        TokenStore(path).load()


def test_token_store_rejects_symbolic_link(tmp_path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "stepik_token.json"
    link.symlink_to(target)

    with pytest.raises(OAuthError, match="symbolic link"):
        TokenStore(link).load()
