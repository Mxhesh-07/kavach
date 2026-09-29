"""
Verification suite for C2 Operator Triage, Incident Management,
and Evidentiary Hash-Chain Immutability.
"""
import json
import pytest

from fastapi.testclient import TestClient
from core.events import EventManager
from cv.rules import Alert as RuleAlert
from core.hashchain import verify_chain


@pytest.fixture
def client(db):
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_camera(client):
    response = client.post(
        "/api/cameras",
        data={
            "name": "C2-CAM-01",
            "url": "samples/sample_border_scenario.mp4",
            "location": "Sector Alpha Outpost 3",
            "is_active": "false",
        },
    )
    assert response.status_code == 201
    return response.json()


def _create_alert(camera_id, alert_type="entry", object_class="person", track_id=101):
    rule_alert = RuleAlert(
        rule_name="perimeter_fence",
        rule_type="fence",
        track_id=track_id,
        alert_type=alert_type,
        description="Perimeter breach detected in sector alpha",
    )
    return EventManager.get().record(
        camera_id=camera_id,
        rule_alert=rule_alert,
        frame=None,
        object_class=object_class,
        confidence=0.92,
    )


def test_triage_action_lifecycle(client, api_camera, db):
    alert = _create_alert(api_camera["id"])
    alert_id = alert["id"]

    # 1. Initially alert has NEW triage status
    detail_res = client.get(f"/api/alerts/{alert_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["triage_status"] == "NEW"

    # 2. Operator acknowledges the alert
    ack_res = client.post(
        f"/api/alerts/{alert_id}/triage",
        json={
            "action": "ACKNOWLEDGE",
            "operator_id": "OPERATOR-42",
            "notes": "Acknowledged on watch desk",
        },
    )
    assert ack_res.status_code == 201
    ack_data = ack_res.json()
    assert ack_data["action"] == "ACKNOWLEDGE"
    assert ack_data["operator_id"] == "OPERATOR-42"
    assert ack_data["action_hash"] != ""

    # 3. Check triage history
    history_res = client.get(f"/api/alerts/{alert_id}/triage")
    assert history_res.status_code == 200
    history = history_res.json()
    assert len(history) == 1
    assert history[0]["action"] == "ACKNOWLEDGE"

    # 4. Resolve the alert
    res_res = client.post(
        f"/api/alerts/{alert_id}/triage",
        json={
            "action": "RESOLVE",
            "operator_id": "OPERATOR-42",
            "notes": "Situation cleared by field patrol",
        },
    )
    assert res_res.status_code == 201

    # 5. Alert triage status is now RESOLVE
    detail_res2 = client.get(f"/api/alerts/{alert_id}")
    assert detail_res2.json()["triage_status"] == "RESOLVE"

    history2 = client.get(f"/api/alerts/{alert_id}/triage").json()
    assert len(history2) == 2
    assert history2[1]["action"] == "RESOLVE"


def test_triage_invalid_action_rejected(client, api_camera):
    alert = _create_alert(api_camera["id"])
    res = client.post(
        f"/api/alerts/{alert['id']}/triage",
        json={"action": "INVALID_ACTION", "operator_id": "OP-1"},
    )
    assert res.status_code == 400
    assert "Invalid triage action" in res.json()["detail"]


def test_triage_false_alarm_with_reason_codes(client, api_camera):
    alert = _create_alert(api_camera["id"])
    res = client.post(
        f"/api/alerts/{alert['id']}/triage",
        json={
            "action": "FALSE_ALARM",
            "operator_id": "WATCHSTANDER-07",
            "reason_code": "BENIGN_WILDLIFE",
            "notes": "Grazing cattle triggered virtual fence tripwire",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["action"] == "FALSE_ALARM"
    assert data["reason_code"] == "BENIGN_WILDLIFE"
    assert "cattle" in data["notes"]


def test_triage_escalate_creates_incident(client, api_camera):
    alert = _create_alert(api_camera["id"], alert_type="entry")
    res = client.post(
        f"/api/alerts/{alert['id']}/triage",
        json={
            "action": "ESCALATE",
            "operator_id": "COMMANDER-01",
            "notes": "Urgent boundary infiltration requiring quick reaction team",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["action"] == "ESCALATE"
    incident_uid = data["incident_uid"]
    assert incident_uid is not None
    assert incident_uid.startswith("INC-")

    # Fetch incident and verify correlation
    inc_res = client.get(f"/api/incidents/{incident_uid}")
    assert inc_res.status_code == 200
    inc_data = inc_res.json()
    assert inc_data["status"] == "ESCALATED"
    assert alert["id"] in inc_data["alert_ids"]
    assert len(inc_data["alerts"]) == 1
    assert inc_data["alerts"][0]["id"] == alert["id"]


def test_triage_preserves_evidentiary_hash_chain(client, api_camera, db):
    """
    CRITICAL INVARIANT TEST:
    Adding triage decisions to alerts must NEVER alter the sealed alert row
    or break the SHA-256 Merkle hash chain.
    """
    # 1. Create a chain of 3 alerts
    alerts = [_create_alert(api_camera["id"], track_id=i) for i in range(1, 4)]
    original_hashes = [(a["id"], a["hash"], a["prev_hash"], a["timestamp"]) for a in alerts]

    # Verify chain initially valid
    result = verify_chain(db)
    assert result.valid is True, f"Initial chain invalid: {result.message}"

    # 2. Perform various triage actions across all alerts
    client.post(
        f"/api/alerts/{alerts[0]['id']}/triage",
        json={"action": "ACKNOWLEDGE", "operator_id": "OP-1"},
    )
    client.post(
        f"/api/alerts/{alerts[1]['id']}/triage",
        json={"action": "ESCALATE", "operator_id": "OP-2"},
    )
    client.post(
        f"/api/alerts/{alerts[2]['id']}/triage",
        json={"action": "FALSE_ALARM", "reason_code": "ENVIRONMENTAL_WIND"},
    )
    client.post(
        f"/api/alerts/{alerts[0]['id']}/triage",
        json={"action": "RESOLVE", "operator_id": "OP-1"},
    )

    # 3. Check that alert row data and cryptographic hashes are 100% unchanged
    from core.models import Alert
    for aid, h, ph, ts in original_hashes:
        row = db.query(Alert).filter(Alert.id == aid).first()
        assert row.hash == h, f"Alert {aid} hash was modified by triage!"
        assert row.prev_hash == ph, f"Alert {aid} prev_hash was modified by triage!"
        assert row.timestamp == ts, f"Alert {aid} timestamp was modified by triage!"

    # 4. Verify the entire chain with verify_chain
    chain_result = verify_chain(db)
    assert chain_result.valid is True, f"Hash chain corrupted after triage operations: {chain_result.message}"


def test_incident_management_api(client, api_camera):
    # 1. Create incident directly
    create_res = client.post(
        "/api/incidents",
        json={
            "title": "Suspicious Night Infiltration",
            "severity": "CRITICAL",
            "status": "OPEN",
            "lead_operator": "DUTY-OFFICER-03",
            "summary": "Multiple sensor activations near riverine sector",
            "sector": "SECTOR-BRAVO",
            "primary_camera_id": api_camera["id"],
            "alert_ids": [],
        },
    )
    assert create_res.status_code == 201
    inc = create_res.json()
    uid = inc["incident_uid"]
    assert uid.startswith("INC-")

    # 2. List incidents
    list_res = client.get("/api/incidents?status=OPEN")
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] >= 1
    found = any(i["incident_uid"] == uid for i in data["incidents"])
    assert found is True

    # 3. Update incident status to CLOSED
    patch_res = client.patch(
        f"/api/incidents/{uid}",
        json={"status": "CLOSED", "summary": "Threat neutralized by border patrol."},
    )
    assert patch_res.status_code == 200
    closed_inc = patch_res.json()
    assert closed_inc["status"] == "CLOSED"
    assert closed_inc["closed_at"] != ""
    assert closed_inc["closed_at_ist"] != ""
    assert "neutralized" in closed_inc["summary"]
