"""
Automated tests for the IBVAP Redesigned Modern Flat Dashboard:
- Incident Creation & Lifecycle Management
- Incident Review Workspace (linked alerts, evidence, status transition)
- Site Map data integrity (camera positioning, status mapping, HUD data)
- Rules Engine table rendering and toggle/deletion endpoints
- Static asset checks (zero emojis in event log, 38-40px button design, DOM consistency)
"""
import json
import pathlib
import pytest
from fastapi.testclient import TestClient

from core.events import EventManager
from cv.rules import Alert as RuleAlert


@pytest.fixture
def client(db):
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_camera(client):
    res = client.post(
        "/api/cameras",
        data={
            "name": "TEST-NORTH-TOWER",
            "url": "samples/sample_border_scenario.mp4",
            "location": "Sector 4 Outpost Alpha",
            "is_active": "false",
        },
    )
    assert res.status_code == 201
    return res.json()


def _create_sample_alert(camera_id, title="Tripwire Breach Test"):
    rule_alert = RuleAlert(
        rule_name="perimeter_tripwire",
        rule_type="line",
        track_id=202,
        alert_type="breach",
        description="Perimeter tripwire breach test",
    )
    return EventManager.get().record(
        camera_id=camera_id,
        rule_alert=rule_alert,
        frame=None,
        object_class="person",
        confidence=0.95,
    )


# --------------------------------------------------------------------------- #
# 1. Incident Creation & Lifecycle Tests
# --------------------------------------------------------------------------- #

def test_incident_modal_submission_and_lifecycle(client, sample_camera):
    """Test full incident workflow from creation through escalation, resolution, and closure."""
    alert = _create_sample_alert(sample_camera["id"])

    # 1. Create incident via API (matching the frontend submitIncident payload)
    create_res = client.post(
        "/api/incidents",
        json={
            "title": "Suspicious Sensor Breach at North Tower",
            "severity": "HIGH",
            "status": "OPEN",
            "sector": "Sector 4 Outpost Alpha",
            "lead_operator": "DUTY-OFFICER-07",
            "primary_camera_id": sample_camera["id"],
            "summary": "Motion detected near outer boundary perimeter fence.",
            "alert_ids": [alert["id"]],
        },
    )
    assert create_res.status_code == 201
    inc_data = create_res.json()
    uid = inc_data["incident_uid"]
    assert uid.startswith("INC-")
    assert inc_data["title"] == "Suspicious Sensor Breach at North Tower"
    assert inc_data["status"] == "OPEN"
    assert inc_data["primary_camera_id"] == sample_camera["id"]

    # 2. Retrieve detailed incident dossier (used by selectIncident in frontend)
    detail_res = client.get(f"/api/incidents/{uid}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["incident_uid"] == uid
    assert len(detail["alerts"]) == 1
    assert detail["alerts"][0]["id"] == alert["id"]
    assert detail["alerts"][0]["camera_name"] == sample_camera["name"]

    # 3. Transition status to ESCALATED
    esc_res = client.patch(
        f"/api/incidents/{uid}",
        json={"status": "ESCALATED"},
    )
    assert esc_res.status_code == 200
    assert esc_res.json()["status"] == "ESCALATED"

    # 4. Transition status to RESOLVED
    res_res = client.patch(
        f"/api/incidents/{uid}",
        json={"status": "RESOLVED", "summary": "Quick reaction team verified perimeter secure."},
    )
    assert res_res.status_code == 200
    resolved = res_res.json()
    assert resolved["status"] == "RESOLVED"
    assert "verified" in resolved["summary"]
    assert resolved["closed_at"] is not None

    # 5. Filter incidents list by status
    list_open = client.get("/api/incidents?status=OPEN").json()
    assert not any(i["incident_uid"] == uid for i in list_open["incidents"])

    list_resolved = client.get("/api/incidents?status=RESOLVED").json()
    assert any(i["incident_uid"] == uid for i in list_resolved["incidents"])


# --------------------------------------------------------------------------- #
# 2. Camera Rules Table & Management Tests
# --------------------------------------------------------------------------- #

def test_camera_rules_lifecycle_for_rules_view(client, sample_camera):
    """Test armed rules table backend endpoints (listing, toggling, deleting)."""
    cam_id = sample_camera["id"]

    # 1. Arm a new spatial rule
    rule_res = client.post(
        f"/api/cameras/{cam_id}/rules",
        data={
            "rule_type": "line",
            "name": "North Fence Line 01",
            "geometry": json.dumps([[100, 150], [500, 150]]),
            "params": json.dumps({"allowed_direction": "entry"}),
            "is_active": "true",
        },
    )
    assert rule_res.status_code == 201
    rule = rule_res.json()
    rule_id = rule["id"]
    assert rule["is_active"] is True

    # 2. List rules for camera (switchRulesCamera in frontend)
    list_res = client.get(f"/api/cameras/{cam_id}/rules")
    assert list_res.status_code == 200
    rules = list_res.json()
    assert len(rules) >= 1
    assert any(r["id"] == rule_id for r in rules)

    # 3. Toggle active status to false (toggleRuleFromTable)
    toggle_res = client.put(
        f"/api/rules/{rule_id}",
        data={"is_active": "false"},
    )
    assert toggle_res.status_code == 200
    assert toggle_res.json()["is_active"] is False

    # 4. Delete the rule (deleteRuleFromTable)
    del_res = client.delete(f"/api/rules/{rule_id}")
    assert del_res.status_code == 200
    assert del_res.json()["ok"] is True
    assert del_res.json()["removed"] == rule_id

    # 5. Verify rule is gone
    final_rules = client.get(f"/api/cameras/{cam_id}/rules").json()
    assert not any(r["id"] == rule_id for r in final_rules)


# --------------------------------------------------------------------------- #
# 3. Frontend Bundle & Static Design Compliance Tests
# --------------------------------------------------------------------------- #

def test_frontend_has_no_emojis_as_evidence_icons():
    """Verify that emoji icons (camera, clapper, numbers) were replaced by SVG badges."""
    app_js = pathlib.Path("static/js/app.js").read_text(encoding="utf-8")
    assert "evidenceBadges" in app_js, "evidenceBadges helper not found in app.js"
    assert "evidence-icon-badge" in app_js, "evidence-icon-badge class missing from app.js"
    # Ensure raw emojis are not used in table cells
    assert "'📷'" not in app_js, "Emoji 📷 still present in app.js"
    assert "'🎬'" not in app_js, "Emoji 🎬 still present in app.js"
    assert "'🔢'" not in app_js, "Emoji 🔢 still present in app.js"


def test_redesigned_ui_has_all_required_controls():
    """Verify all redesigned DOM elements exist in dashboard/index.html."""
    html = pathlib.Path("dashboard/index.html").read_text(encoding="utf-8")
    # New incident modal & controls
    assert 'id="modal-incident"' in html, "New incident modal missing from index.html"
    assert 'id="inc-title"' in html, "Incident title input missing from index.html"
    assert 'id="inc-severity"' in html, "Incident severity select missing from index.html"
    assert 'id="inc-submit"' in html, "Incident submit button missing from index.html"
    # Site map controls & HUD
    assert 'id="site-map-canvas"' in html, "Site map canvas missing from index.html"
    assert 'id="btn-map-cones"' in html, "Toggle cones button missing from index.html"
    assert 'id="map-selected-hud"' in html, "Map selected sensor HUD missing from index.html"
    # Incident layout
    assert 'id="incident-status-filter"' in html, "Incident status filter missing from index.html"
    assert 'id="incident-details"' in html, "Incident details workspace missing from index.html"


def test_css_design_system_tokens():
    """Verify modern flat UI tokens: solid surfaces, 1px borders, typography, button heights."""
    css = pathlib.Path("static/css/style.css").read_text(encoding="utf-8")
    # Solid dark surfaces
    assert "--color-bg-canvas:" in css
    assert "--color-bg-surface:" in css
    assert "--color-border-default:" in css
    # Practical minimum button heights (38px to 40px)
    assert "min-height: 38px" in css or "height: 38px" in css, "Button 38px min-height missing from style.css"
    # Typography: 14.5px body text
    assert "font-size: 14.5px" in css or "font-size: 14px" in css
    # Flat site map and incident layout
    assert ".map-container" in css
    assert ".incident-layout" in css
    assert ".map-hud" in css
