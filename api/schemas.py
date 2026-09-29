"""
Pydantic request/response schemas.

These decouple the database model from the wire format, allowing us to
hide internal fields (prev_hash, hash) from the public API while still
exposing them on demand.
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------------ #
# Camera
# ------------------------------------------------------------------ #
class CameraBase(BaseModel):
    name: str
    url: str
    location: Optional[str] = None
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraRead(CameraBase):
    id: int
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Rule
# ------------------------------------------------------------------ #
class RuleBase(BaseModel):
    rule_type: str
    geometry: Any          # list of coordinate pairs / line points
    params: Optional[dict] = None
    is_active: bool = True


class RuleCreate(RuleBase):
    pass


class RuleRead(RuleBase):
    id: int
    camera_id: int
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Alert
# ------------------------------------------------------------------ #
class AlertBase(BaseModel):
    alert_type: str
    object_class: Optional[str] = None
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    timestamp: str
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None


class AlertRead(AlertBase):
    id: int
    camera_id: int
    hash: str
    prev_hash: str
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ #
# Stats
# ------------------------------------------------------------------ #
class StatsResponse(BaseModel):
    total_alerts: int
    by_type: dict[str, int]
    by_camera: dict[str, int]
    today: int


# ------------------------------------------------------------------ #
# Operator Triage & Incidents
# ------------------------------------------------------------------ #
class AlertTriageCreate(BaseModel):
    action: str  # ACKNOWLEDGE, ESCALATE, RESOLVE, FALSE_ALARM
    operator_id: Optional[str] = "operator"
    reason_code: Optional[str] = ""
    notes: Optional[str] = ""


class AlertTriageRead(BaseModel):
    id: int
    alert_id: int
    action: str
    operator_id: str
    reason_code: str
    notes: str
    timestamp: str
    timestamp_ist: str
    action_hash: str
    model_config = ConfigDict(from_attributes=True)


class IncidentCreate(BaseModel):
    title: str
    severity: Optional[str] = "MEDIUM"
    status: Optional[str] = "OPEN"
    lead_operator: Optional[str] = "operator"
    summary: Optional[str] = ""
    sector: Optional[str] = "SECTOR-ALPHA"
    primary_camera_id: Optional[int] = None
    alert_ids: Optional[list[int]] = None


class IncidentUpdate(BaseModel):
    title: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    lead_operator: Optional[str] = None
    summary: Optional[str] = None
    sector: Optional[str] = None
    primary_camera_id: Optional[int] = None
    alert_ids: Optional[list[int]] = None


class IncidentRead(BaseModel):
    id: int
    incident_uid: str
    title: str
    severity: str
    status: str
    lead_operator: str
    summary: str
    sector: str
    primary_camera_id: Optional[int] = None
    created_at: str
    created_at_ist: str
    closed_at: Optional[str] = ""
    closed_at_ist: Optional[str] = ""
    alert_ids: list[int] = []
    model_config = ConfigDict(from_attributes=True)

