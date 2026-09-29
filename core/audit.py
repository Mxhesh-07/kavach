"""
Audit logging for all operator actions.

Every state-changing action in the system can be recorded in an audit log,
providing a complete, tamper-evident trail of who did what, when, and from
where. This is essential for forensic review and accountability.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from core.models import AuditLog
from core.timeutil import utc_iso, fmt_ist

log = logging.getLogger("kavach.audit")


def record_audit(
    db: Session,
    action: str,
    username: str = "system",
    resource: str = "",
    resource_id: str = "",
    ip_address: str = "",
    detail: Optional[dict] = None,
) -> AuditLog:
    """Record an audit log entry."""
    import json

    now_u = utc_iso()
    entry = AuditLog(
        username=username[:120],
        action=action[:60],
        resource=resource[:120],
        resource_id=resource_id[:120],
        ip_address=ip_address[:64],
        detail_json=json.dumps(detail or {}),
        timestamp=now_u,
        timestamp_ist=fmt_ist(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_audit_logs(
    db: Session,
    username: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List audit log entries with optional filters."""
    query = db.query(AuditLog)
    if username:
        query = query.filter(AuditLog.username == username)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource:
        query = query.filter(AuditLog.resource == resource)

    total = query.count()
    rows = query.order_by(desc(AuditLog.id)).offset(offset).limit(limit).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": [
            {
                "id": e.id,
                "username": e.username,
                "action": e.action,
                "resource": e.resource,
                "resource_id": e.resource_id,
                "ip_address": e.ip_address,
                "detail": e.detail_json,
                "timestamp": e.timestamp,
                "timestamp_ist": e.timestamp_ist,
            }
            for e in rows
        ],
    }


def list_audit_actions(db: Session) -> list[str]:
    """Return distinct action values in the audit log."""
    rows = db.query(AuditLog.action).distinct().order_by(AuditLog.action).all()
    return [r[0] for r in rows]
