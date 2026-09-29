"""
Dashboard summary endpoint.

Provides an aggregate overview of the entire platform: camera counts,
alert statistics, threat level, detection counts, and system health.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from core import models
from core.timeutil import now_utc, start_of_ist_day, fmt_ist
from core.threat_level import compute_threat_level

log = logging.getLogger("kavach.dashboard")


def get_dashboard_summary(db: Session) -> dict:
    """
    Aggregate dashboard summary.

    Returns a comprehensive overview of platform state.
    """
    now = now_utc()
    one_hour_ago = (now - timedelta(hours=1)).isoformat()
    today_start = start_of_ist_day().isoformat()

    # Camera counts
    cam_total = db.query(func.count(models.Camera.id)).scalar() or 0
    cam_online = db.query(func.count(models.Camera.id)).filter(
        models.Camera.is_online == True
    ).scalar() or 0

    # Alert counts
    total_alerts = db.query(func.count(models.Alert.id)).scalar() or 0
    today_alerts = db.query(func.count(models.Alert.id)).filter(
        models.Alert.timestamp >= today_start
    ).scalar() or 0

    # Alerts by severity today
    sev_rows = (
        db.query(models.Alert.severity, func.count(models.Alert.id))
        .filter(models.Alert.timestamp >= today_start)
        .group_by(models.Alert.severity)
        .all()
    )
    by_severity = {s: c for s, c in sev_rows}

    # Alerts by type today
    type_rows = (
        db.query(models.Alert.alert_type, func.count(models.Alert.id))
        .filter(models.Alert.timestamp >= today_start)
        .group_by(models.Alert.alert_type)
        .all()
    )
    by_type = {t: c for t, c in type_rows}

    # Recent alerts (last hour)
    recent_count = db.query(func.count(models.Alert.id)).filter(
        models.Alert.timestamp >= one_hour_ago
    ).scalar() or 0

    # Open incidents
    open_incidents = (
        db.query(func.count(models.Incident.id))
        .filter(models.Incident.status.in_(["OPEN", "UNDER_INVESTIGATION", "ESCALATED"]))
        .scalar()
        or 0
    )

    # ANPR stats
    anpr_total = db.query(func.count(models.ANPRDetection.id)).scalar() or 0
    anpr_today = db.query(func.count(models.ANPRDetection.id)).filter(
        models.ANPRDetection.timestamp >= today_start
    ).scalar() or 0

    # Face stats
    face_total = db.query(func.count(models.FaceDetection.id)).scalar() or 0
    face_matched = db.query(func.count(models.FaceDetection.id)).filter(
        models.FaceDetection.recognition_status == "matched"
    ).scalar() or 0

    # Watchlist
    watchlist_count = db.query(func.count(models.WatchlistEntry.id)).scalar() or 0

    # Threat level
    threat = compute_threat_level(db)

    # Triage stats
    triage_total = db.query(func.count(models.AlertTriageAction.id)).scalar() or 0

    return {
        "cameras": {
            "total": cam_total,
            "online": cam_online,
            "offline": cam_total - cam_online,
        },
        "alerts": {
            "total": total_alerts,
            "today": today_alerts,
            "last_hour": recent_count,
            "by_severity_today": by_severity,
            "by_type_today": by_type,
        },
        "incidents": {
            "open": open_incidents,
        },
        "anpr": {
            "total": anpr_total,
            "today": anpr_today,
        },
        "faces": {
            "total": face_total,
            "matched": face_matched,
        },
        "watchlist": {
            "entries": watchlist_count,
        },
        "triage": {
            "total_actions": triage_total,
        },
        "threat_level": threat,
        "generated_at_ist": fmt_ist(),
    }
