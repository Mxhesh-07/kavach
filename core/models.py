"""
SQLAlchemy ORM models.

Five tables:

``cameras``        registered video sources (RTSP / HTTP / webcam / file)
``rules``          virtual fences, zones, loiter areas, direction rules
``alerts``         the tamper-evident event log (SHA-256 hash chain)
``checkpoints``    periodic Merkle checkpoints sealing a range of the chain
``watchlist_entries``  enrolled face embeddings
``analysis_sessions``  uploaded-MP4 analysis runs

Timestamps are stored as aware **UTC ISO-8601** strings (sortable, filterable)
alongside a denormalised **IST** display string, so the audit log reads
correctly to an Indian operator without any client-side conversion.
"""
from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import relationship

from core.database import Base

# Alert severities, ordered.
SEVERITY_INFO = "INFO"
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_ORDER = {
    SEVERITY_INFO: 0, SEVERITY_LOW: 1, SEVERITY_MEDIUM: 2,
    SEVERITY_HIGH: 3, SEVERITY_CRITICAL: 4,
}


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    url = Column(String(500), nullable=False)
    location = Column(String(200), default="")
    is_active = Column(Boolean, default=True)
    is_online = Column(Boolean, default=False)
    #: "live"   — RTSP / HTTP / webcam index
    #: "file"   — an MP4 registered as a first-class camera source
    #: "upload" — the pseudo-source that owns offline analysis events
    source_kind = Column(String(20), default="live")
    created_at = Column(String(40), default="")

    #: Soft-delete marker. Removing a camera must not destroy its events: the
    #: alert log is a SHA-256 hash chain, and deleting rows from the middle of
    #: it invalidates every subsequent row, so integrity verification would fail
    #: forever afterwards. A removed camera that still owns events is therefore
    #: archived — hidden from every listing, never auto-started, its stream gone
    #: — while its sealed evidence stays verifiable. A camera with no events is
    #: deleted outright. See ``retire_camera``.
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(String(40), default="")

    #: Rules are configuration and are removed with the camera.
    rules = relationship("Rule", back_populates="camera", cascade="all, delete-orphan")
    #: Alerts are evidence. ``passive_deletes`` keeps SQLAlchemy from issuing a
    #: cascade that would silently shred the audit chain; the delete path checks
    #: for dependants and archives instead.
    alerts = relationship("Alert", back_populates="camera", passive_deletes=True)
    sessions = relationship("AnalysisSession", back_populates="camera",
                            passive_deletes=True)

    def __repr__(self) -> str:
        return f"<Camera {self.id} {self.name}>"

    @property
    def is_file_source(self) -> bool:
        return (self.source_kind or "live") == "file"


class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    #: 'line' | 'zone' | 'loiter' | 'direction'
    rule_type = Column(String(50), nullable=False)
    geometry = Column(Text)      # JSON coordinate list, in frame pixel space
    params = Column(Text)        # JSON, e.g. {"dwell_seconds": 30}
    is_active = Column(Boolean, default=True)
    name = Column(String(200))
    created_at = Column(String(40), default="")

    camera = relationship("Camera", back_populates="rules")

    def __repr__(self) -> str:
        return f"<Rule {self.id} name={self.name} type={self.rule_type}>"


class Alert(Base):
    """
    One sealed event in the audit log.

    ``prev_hash``/``hash`` chain every row to its predecessor: altering any
    hashed column of any row invalidates that row and everything after it.
    """

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), default=SEVERITY_MEDIUM)
    object_class = Column(String(50), default="")
    track_id = Column(Integer, default=0)
    confidence = Column(Float, default=0.0)

    #: Aware UTC ISO-8601 — canonical, sortable, hashed.
    timestamp = Column(String(50), nullable=False)
    #: Denormalised IST display string — what the operator actually reads.
    timestamp_ist = Column(String(64), default="")

    #: Which rule produced this event, and of what kind.
    rule_name = Column(String(200), default="")
    rule_type = Column(String(50), default="")
    #: 'ai_detection' for model output, 'rule_engine' for analytics rules.
    detector = Column(String(40), default="rule_engine")
    #: 'live' or 'upload'
    source_type = Column(String(20), default="live")
    #: Upload analysis session id, when source_type == 'upload'.
    session_id = Column(String(64), default="")
    #: Free-form JSON: plate text, watchlist name, dwell seconds, geometry…
    details_json = Column(Text, default="{}")
    description = Column(Text, default="")

    snapshot_path = Column(String(500), default="")
    clip_path = Column(String(500), default="")

    prev_hash = Column(String(64), nullable=False)
    hash = Column(String(64), nullable=False)

    camera = relationship("Camera", back_populates="alerts")
    triage_actions = relationship(
        "AlertTriageAction",
        back_populates="alert",
        cascade="all, delete-orphan",
        order_by="AlertTriageAction.id",
    )

    __table_args__ = (
        Index("ix_alerts_timestamp", "timestamp"),
        Index("ix_alerts_camera_type", "camera_id", "alert_type"),
        Index("ix_alerts_session", "session_id"),
    )

    def __repr__(self) -> str:
        return f"<Alert {self.id} type={self.alert_type} cam={self.camera_id}>"


class Checkpoint(Base):
    """
    A periodic Merkle checkpoint over a contiguous range of the alert chain.

    This is **not** a public blockchain anchor — nothing is broadcast to any
    network.  It is a locally sealed checkpoint: a Merkle root over the alert
    hashes in ``[first_alert_id, last_alert_id]`` plus the chain tip at seal
    time.  Exported checkpoints let a third party verify a range of the log
    without being handed the whole database.
    """

    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True)
    checkpoint_uid = Column(String(64), nullable=False, unique=True)
    first_alert_id = Column(Integer, default=0)
    last_alert_id = Column(Integer, default=0)
    alert_count = Column(Integer, default=0)
    merkle_root = Column(String(64), nullable=False)
    chain_tip_hash = Column(String(64), nullable=False)
    timestamp = Column(String(50), nullable=False)     # aware UTC ISO-8601
    timestamp_ist = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<Checkpoint {self.id} root={self.merkle_root[:12]}…>"


class WatchlistEntry(Base):
    """Persistent face watchlist entry (normalised ArcFace embedding)."""

    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    embedding_json = Column(Text, nullable=False)
    metadata_json = Column(Text, default="{}")
    created_at = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<WatchlistEntry {self.id} name={self.name}>"


class AnalysisSession(Base):
    """An uploaded-MP4 analysis run, persisted so results survive a restart."""

    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True)
    session_uid = Column(String(64), nullable=False, unique=True)
    filename = Column(String(300), nullable=False)
    stored_path = Column(String(500), nullable=False)
    output_path = Column(String(500), default="")
    #: The camera whose rules this run inherited, or the upload pseudo-source.
    #: This foreign key is what made "Remove Camera" fail with
    #: ``FOREIGN KEY constraint failed``: the ORM had no relationship for it, so
    #: no cascade applied and SQLite rejected the parent DELETE outright.
    camera_id = Column(Integer, ForeignKey("cameras.id"))

    #: queued | running | completed | failed | cancelled
    status = Column(String(20), default="queued")
    error = Column(Text, default="")

    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    analysed_frames = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    source_fps = Column(Float, default=0.0)
    width = Column(Integer, default=0)
    height = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)

    detections_total = Column(Integer, default=0)
    persons_seen = Column(Integer, default=0)
    vehicles_seen = Column(Integer, default=0)
    alerts_generated = Column(Integer, default=0)
    processing_fps = Column(Float, default=0.0)

    created_at = Column(String(50), default="")
    created_at_ist = Column(String(64), default="")
    completed_at = Column(String(50), default="")
    completed_at_ist = Column(String(64), default="")

    camera = relationship("Camera", back_populates="sessions")

    __table_args__ = (Index("ix_sessions_created", "created_at"),)

    def __repr__(self) -> str:
        return f"<AnalysisSession {self.session_uid} {self.status}>"


class ANPRDetection(Base):
    """
    A licence plate reading, stored as structured data rather than only as text.

    The alert log already carries every plate read as a ``anpr_detection``
    event, but an event is a *narrative* row: its payload is JSON and its
    purpose is the audit chain.  Answering the questions an operator actually
    asks of ANPR — "has this plate passed any camera this week", "show every
    read of KA01F1234", "which vehicles crossed after 2 a.m." — against JSON
    text is both slow and unreliable.

    This table is the queryable projection of the same reading: one row per
    published plate, indexed by plate text, camera and time, and linked back to
    the sealed alert it came from.  The alert remains the evidentiary record;
    this is the index over it, which is why ``alert_id`` is not nullable for
    anything the pipeline writes.
    """

    __tablename__ = "anpr_detections"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    #: The sealed event this reading belongs to — the tamper-evident record.
    alert_id = Column(Integer, ForeignKey("alerts.id"))

    timestamp = Column(String(50), nullable=False)
    timestamp_ist = Column(String(64), default="")

    #: Normalised, no spaces: "MH12AB1234".
    plate_text = Column(String(24), nullable=False)
    #: Grouped for display: "MH 12 AB 1234".
    plate_display = Column(String(32), default="")
    #: What OCR returned before grammar correction, kept so the record shows
    #: what the recogniser actually saw rather than only the tidied result.
    plate_raw = Column(String(32), default="")
    confidence = Column(Float, default=0.0)
    #: True when the reading matches the Indian registration grammar.
    format_verified = Column(Boolean, default=False)
    #: How many independent frames agreed on this reading.
    votes = Column(Integer, default=1)
    consensus = Column(Boolean, default=False)

    vehicle_class = Column(String(32), default="")
    vehicle_track_id = Column(Integer, default=0)

    evidence_path = Column(String(500), default="")
    #: SHA-256 of the evidence image, so the crop cannot be swapped silently.
    evidence_sha256 = Column(String(64), default="")

    #: live | upload | system
    source_type = Column(String(20), default="live")
    session_id = Column(String(64), default="")
    #: published | uncertain — an uncertain read is recorded for review but is
    #: never presented as an identified registration.
    processing_status = Column(String(20), default="published")

    camera = relationship("Camera")

    __table_args__ = (
        Index("ix_anpr_plate", "plate_text"),
        Index("ix_anpr_timestamp", "timestamp"),
        Index("ix_anpr_camera_time", "camera_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<ANPRDetection {self.plate_text} cam={self.camera_id}>"


class FaceDetection(Base):
    """
    A detected face, and whether an identity was established for it.

    ``recognition_status`` is deliberately explicit rather than implied by a
    nullable identity column.  The system detects far more faces than it can
    identify, and the difference matters: "a face was seen here" and "this
    person was seen here" are very different claims to put in a border-security
    record.  ``unknown`` means the face did not match any watchlist entry above
    threshold — never that the subject is unidentifiable.
    """

    __tablename__ = "face_detections"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    alert_id = Column(Integer, ForeignKey("alerts.id"))

    timestamp = Column(String(50), nullable=False)
    timestamp_ist = Column(String(64), default="")

    #: SCRFD detector score for the face itself.
    confidence = Column(Float, default=0.0)
    #: YOLO person track this face was associated with, 0 if unassociated.
    track_id = Column(Integer, default=0)

    #: matched | unknown
    recognition_status = Column(String(20), default="unknown")
    identity_id = Column(Integer, ForeignKey("watchlist_entries.id"))
    identity_name = Column(String(120), default="")
    #: Cosine similarity against the best watchlist candidate, matched or not.
    similarity = Column(Float, default=0.0)
    similarity_threshold = Column(Float, default=0.0)

    bbox_json = Column(String(120), default="")
    evidence_path = Column(String(500), default="")
    evidence_sha256 = Column(String(64), default="")

    source_type = Column(String(20), default="live")
    session_id = Column(String(64), default="")

    camera = relationship("Camera")

    __table_args__ = (
        Index("ix_face_timestamp", "timestamp"),
        Index("ix_face_camera_time", "camera_id", "timestamp"),
        Index("ix_face_identity", "identity_id"),
    )

    def __repr__(self) -> str:
        return f"<FaceDetection {self.recognition_status} cam={self.camera_id}>"


class AlertTriageAction(Base):
    """
    Operator triage log entry for an alert.

    CRITICAL INVARIANT: Alerts are sealed in a SHA-256 hash chain and their
    columns must never be modified in-place after insertion. Instead, operator
    triage decisions (ACKNOWLEDGE, ESCALATE, RESOLVE, FALSE_ALARM) are appended
    as distinct, immutable triage actions linked to the alert, maintaining full
    auditability and evidentiary integrity.
    """

    __tablename__ = "alert_triage_actions"

    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    action = Column(String(32), nullable=False)  # ACKNOWLEDGE, ESCALATE, RESOLVE, FALSE_ALARM
    operator_id = Column(String(64), default="operator")
    reason_code = Column(String(64), default="")  # e.g. ENVIRONMENTAL_WIND, BENIGN_WILDLIFE, etc.
    notes = Column(Text, default="")
    timestamp = Column(String(50), nullable=False)  # UTC ISO-8601
    timestamp_ist = Column(String(64), default="")
    action_hash = Column(String(64), default="")

    alert = relationship("Alert", back_populates="triage_actions")

    __table_args__ = (
        Index("ix_triage_alert_id", "alert_id"),
        Index("ix_triage_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<AlertTriageAction {self.id} alert={self.alert_id} action={self.action}>"


class Incident(Base):
    """
    Formal operational incident aggregating one or more correlated alerts
    under lead operator review.
    """

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True)
    incident_uid = Column(String(64), nullable=False, unique=True)
    title = Column(String(200), nullable=False)
    severity = Column(String(20), default=SEVERITY_MEDIUM)
    status = Column(String(32), default="OPEN")  # OPEN, UNDER_INVESTIGATION, ESCALATED, RESOLVED, CLOSED
    lead_operator = Column(String(64), default="operator")
    summary = Column(Text, default="")
    sector = Column(String(100), default="SECTOR-ALPHA")
    primary_camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    created_at = Column(String(50), nullable=False)
    created_at_ist = Column(String(64), default="")
    closed_at = Column(String(50), default="")
    closed_at_ist = Column(String(64), default="")
    alert_ids_json = Column(Text, default="[]")

    camera = relationship("Camera")

    __table_args__ = (
        Index("ix_incidents_uid", "incident_uid"),
        Index("ix_incidents_created", "created_at"),
        Index("ix_incidents_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Incident {self.incident_uid} {self.status} sev={self.severity}>"


class AuditLog(Base):
    """
    Immutable audit trail entry.

    Records every significant operator action: logins, camera changes,
    alert triage, evidence access, configuration changes, etc.
    """

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    username = Column(String(120), default="system")
    action = Column(String(60), nullable=False, index=True)
    resource = Column(String(120), default="")
    resource_id = Column(String(120), default="")
    ip_address = Column(String(64), default="")
    detail_json = Column(Text, default="{}")
    timestamp = Column(String(50), nullable=False)
    timestamp_ist = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<AuditLog {self.id} {self.action} by {self.username}>"


class Notification(Base):
    """
    In-app notification for a user (or all users if username is empty).
    """

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    #: Empty string means broadcast to all users
    username = Column(String(120), default="", index=True)
    #: alert | system | info
    type = Column(String(32), default="info")
    title = Column(String(200), nullable=False)
    message = Column(Text, default="")
    payload_json = Column(Text, default="{}")
    is_read = Column(Integer, default=0)  # 0=unread, 1=read
    timestamp = Column(String(50), nullable=False)
    timestamp_ist = Column(String(64), default="")

    def __repr__(self) -> str:
        return f"<Notification {self.id} {self.type} {self.title[:40]}>"


class SystemSetting(Base):
    """
    A single runtime-tunable system setting.

    Settings are stored as key-value pairs with type information,
    allowing the API to validate and convert values.
    """

    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(120), nullable=False, unique=True, index=True)
    value = Column(Text, default="")
    #: string | boolean | integer | float | json
    value_type = Column(String(20), default="string")
    category = Column(String(60), default="general")
    description = Column(Text, default="")
    updated_at = Column(String(50), default="")
    updated_at_ist = Column(String(64), default="")

    def typed_value(self):
        """Convert the stored string value to its typed Python value."""
        raw = self.value
        try:
            if self.value_type == "boolean":
                return raw and raw.lower() in ("1", "true", "yes", "on")
            if self.value_type == "integer":
                return int(float(raw))
            if self.value_type == "float":
                return float(raw)
            if self.value_type == "json":
                return json.loads(raw) if raw else None
        except (TypeError, ValueError):
            return raw
        return raw

    def __repr__(self) -> str:
        return f"<SystemSetting {self.key}={self.value}>"

