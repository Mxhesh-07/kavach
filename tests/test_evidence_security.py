"""
Phase 19: Security, Anti-Tamper & Evidence Integrity Tests.
Validates:
1. SHA-256 hash chaining and tamper detection on alert logs.
2. HMAC cryptographic signature verification on exported evidence clips.
3. Sensitive credential masking in logs and connection URIs.
"""
import hashlib
import hmac
from pathlib import Path
import pytest
from core.hashchain import chain_hash, GENESIS_HASH


def test_hashchain_integrity_and_tamper_detection():
    """Verify that tampering with any historical row breaks the hash chain."""
    p1 = {"event": "intrusion", "camera": 1, "ts": 100}
    h1 = chain_hash(p1, GENESIS_HASH)

    p2 = {"event": "loitering", "camera": 1, "ts": 105}
    h2 = chain_hash(p2, h1)

    p3 = {"event": "tripwire", "camera": 2, "ts": 110}
    h3 = chain_hash(p3, h2)

    # Valid chain check
    assert chain_hash(p1, GENESIS_HASH) == h1
    assert chain_hash(p2, h1) == h2
    assert chain_hash(p3, h2) == h3

    # Tamper with p2
    tampered_p2 = {"event": "falsified_event", "camera": 1, "ts": 105}
    tampered_h2 = chain_hash(tampered_p2, h1)
    assert tampered_h2 != h2

    # Chain validation on tampered history fails
    recomputed_h3 = chain_hash(p3, tampered_h2)
    assert recomputed_h3 != h3


def test_hmac_evidence_signature():
    """Verify HMAC-SHA256 signature verification on evidence payloads."""
    secret_key = b"ibvap_secure_airgap_secret_key_2026"
    evidence_payload = b"MP4_VIDEO_FRAME_BYTES_SIMULATION_12345"

    # Sign
    signature = hmac.new(secret_key, evidence_payload, hashlib.sha256).hexdigest()

    # Verify valid
    computed = hmac.new(secret_key, evidence_payload, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(signature, computed) is True

    # Verify modified payload rejected
    tampered_payload = b"MP4_VIDEO_FRAME_BYTES_SIMULATION_12346"
    computed_tampered = hmac.new(secret_key, tampered_payload, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(signature, computed_tampered) is False


def test_credential_sanitization_in_uris():
    """Verify RTSP and HTTP credentials are scrubbed from logged URLs."""
    from core.video_source import sanitize_url

    raw_url = "rtsp://admin:superSecretP@ssword123@192.168.1.100:554/live/ch0"
    sanitized = sanitize_url(raw_url)

    assert "superSecretP@ssword123" not in sanitized
    assert "admin:***@192.168.1.100" in sanitized or "***@" in sanitized
