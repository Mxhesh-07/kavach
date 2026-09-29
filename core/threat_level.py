"""
Dynamic threat level assessment.

Computes an overall platform threat level based on recent alert
severity counts, active incidents, and system health.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from core import models
from core.timeutil import now_utc

log = logging.getLogger("ibvap.threat")


def compute_threat_level(db: Session) -> dict:
    """
    Derive the platform threat level from recent events.

    Rules:
    - CRITICAL: any CRITICAL alert in the last hour
    - HIGH: 3+ HIGH alerts in the last hour
    - MEDIUM: 5+ MEDIUM alerts in the last hour
    - LOW: otherwise
    """
    one_hour_ago = (now_utc() - timedelta(hours=1)).isoformat()

    recent = (
        db.query(models.Alert.severity, func.count(models.Alert.id))
        .filter(models.Alert.timestamp >= one_hour_ago)
        .group_by(models.Alert.severity)
        .all()
    )
    sev_counts = {s: c for s, c in recent}

    critical = sev_counts.get("CRITICAL", 0)
    high = sev_counts.get("HIGH", 0)
    medium = sev_counts.get("MEDIUM", 0)

    if critical >= 1:
        level = "CRITICAL"
    elif high >= 3:
        level = "HIGH"
    elif medium >= 5:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Count open incidents
    open_incidents = (
        db.query(func.count(models.Incident.id))
        .filter(models.Incident.status.in_(["OPEN", "UNDER_INVESTIGATION", "ESCALATED"]))
        .scalar()
        or 0
    )

    # Count active alerts (not resolved/false alarm)
    active_alerts = (
        db.query(func.count(models.Alert.id))
        .filter(
            models.Alert.severity.in_(["HIGH", "CRITICAL"]),
        )
        .scalar()
        or 0
    )

    return {
        "threat_level": level,
        "critical_alerts_last_hour": critical,
        "high_alerts_last_hour": high,
        "medium_alerts_last_hour": medium,
        "open_incidents": open_incidents,
        "active_high_critical_alerts": active_alerts,
        "factors": _threat_factors(critical, high, medium, open_incidents),
    }


def _threat_factors(critical: int, high: int, medium: int, open_incidents: int) -> list[str]:
    """Human-readable explanation of why the threat level is what it is."""
    factors = []
    if critical >= 1:
        factors.append(f"{critical} CRITICAL alert(s) in the last hour")
    if high >= 3:
        factors.append(f"{high} HIGH alerts in the last hour")
    if medium >= 5:
        factors.append(f"{medium} MEDIUM alerts in the last hour")
    if open_incidents > 0:
        factors.append(f"{open_incidents} open incident(s)")
    if not factors:
        factors.append("No significant threat indicators")
    return factors
