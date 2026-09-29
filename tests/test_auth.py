"""
Authentication hardening — X-API-Key enforcement.

The main suite runs unauthenticated (conftest sets AUTH_REQUIRED=false so it can
hit endpoints directly). Here we flip authentication on *in-process* and assert
the contract: state-changing and sensitive routes are rejected without a valid
key, the public allowlist stays reachable, and media sinks accept the key as a
query parameter — all while an invalid key is rejected.
"""
import pytest
from fastapi.testclient import TestClient

from core.config import settings


@pytest.fixture(autouse=True)
def _auth_on(monkeypatch):
    """Force authentication on with a known key for the whole auth test."""
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "AUTH_API_KEY", "test-secret-key")
    yield
    monkeypatch.setattr(settings, "AUTH_REQUIRED", settings.AUTH_REQUIRED)
    monkeypatch.setattr(settings, "AUTH_API_KEY", "")


import pytest
from fastapi.testclient import TestClient

from core.config import settings


@pytest.fixture(autouse=True)
def _auth_on(monkeypatch):
    """Force authentication on with a known key for the whole auth test."""
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "AUTH_API_KEY", "test-secret-key")
    yield
    monkeypatch.setattr(settings, "AUTH_REQUIRED", settings.AUTH_REQUIRED)
    monkeypatch.setattr(settings, "AUTH_API_KEY", "")


@pytest.fixture(scope="module")
def client():
    """A TestClient entered as a context manager so the lifespan (init_db) runs."""
    import api.main as m

    with TestClient(m.app) as c:
        yield c


def test_private_route_requires_key(client):
    r = client.get("/api/cameras")
    # /api/cameras is not in the public allowlist, so it demands the header.
    assert r.status_code == 401
    assert client.get("/api/cameras",
                      headers={"X-API-Key": "test-secret-key"}).status_code == 200


def test_state_change_requires_key(client):
    body = {"name": "x", "url": "0"}
    # Without a key the state change is rejected before it reaches the handler.
    assert client.post("/api/cameras", data=body).status_code == 401
    # With a valid key the request is admitted *past* auth (camera creation may
    # then succeed or fail validation on its own merits — the point is it is no
    # longer blocked as 401 Unauthorized).
    assert client.post(
        "/api/cameras", data=body, headers={"X-API-Key": "test-secret-key"}
    ).status_code != 401


def test_wrong_key_rejected(client):
    assert client.get(
        "/api/cameras", headers={"X-API-Key": "nope"}
    ).status_code == 401


def test_public_allowlist_stays_open(client):
    assert client.get("/api/system/info").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/dashboard").status_code == 200


def test_media_sink_accepts_query_key(client):
    # A camera stream GET with a valid ?api_key= passes auth (the handler then
    # reports the stream is offline / 404 for a fresh DB, not a 401).
    r = client.get("/stream/999999", params={"api_key": "test-secret-key"})
    assert r.status_code != 401
    assert client.get("/stream/999999").status_code == 401


def test_hard_reset_requires_token_even_when_auth_on(client):
    headers = {"headers": {"X-API-Key": "test-secret-key"}}
    # No HARD_RESET_TOKEN configured and not enabled by default path harnessed
    # here — assert the endpoint at least consults auth before anything else.
    r = client.post("/api/system/hard-reset", params={"confirm": "true"},
                    **headers)
    # Either 403 (disabled) or a token error — never a wipe without a token.
    assert r.status_code in (403, 400, 401)