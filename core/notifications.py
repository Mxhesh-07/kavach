"""
In-app notification system.

Provides user-targeted notifications for alerts, system events, and
informational messages. Supports read/unread tracking and per-user feeds.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from core.models import Notification
from core.timeutil import utc_iso, fmt_ist

log = logging.getLogger("ibvap.notifications")


def create_notification(
    db: Session,
    title: str,
    message: str = "",
    type_: str = "info",
    username: str = "",
    payload: Optional[dict] = None,
) -> Notification:
    """Create a new notification."""
    now_u = utc_iso()
    notif = Notification(
        username=username[:120],
        type=type_[:32],
        title=title[:200],
        message=message,
        payload_json=json.dumps(payload or {}),
        is_read=0,
        timestamp=now_u,
        timestamp_ist=fmt_ist(),
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)
    return notif


def list_notifications(
    db: Session,
    username: str = "",
    unread_only: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """List notifications for a user (or global)."""
    query = db.query(Notification)
    if username:
        query = query.filter(
            (Notification.username == username) | (Notification.username == "")
        )
    if unread_only:
        query = query.filter(Notification.is_read == 0)

    total = query.count()
    rows = query.order_by(desc(Notification.id)).offset(offset).limit(limit).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "notifications": [
            {
                "id": n.id,
                "username": n.username,
                "type": n.type,
                "title": n.title,
                "message": n.message,
                "payload": n.payload_json,
                "is_read": bool(n.is_read),
                "timestamp": n.timestamp,
                "timestamp_ist": n.timestamp_ist,
            }
            for n in rows
        ],
    }


def unread_count(db: Session, username: str = "") -> int:
    """Count unread notifications for a user."""
    query = db.query(func.count(Notification.id)).filter(Notification.is_read == 0)
    if username:
        query = query.filter(
            (Notification.username == username) | (Notification.username == "")
        )
    return query.scalar() or 0


def mark_read(db: Session, notification_id: int, username: str = "") -> bool:
    """Mark a notification as read."""
    query = db.query(Notification).filter(Notification.id == notification_id)
    if username:
        query = query.filter(
            (Notification.username == username) | (Notification.username == "")
        )
    notif = query.first()
    if not notif:
        return False
    notif.is_read = 1
    db.commit()
    return True


def mark_all_read(db: Session, username: str = "") -> int:
    """Mark all notifications as read for a user. Returns count updated."""
    query = db.query(Notification).filter(Notification.is_read == 0)
    if username:
        query = query.filter(
            (Notification.username == username) | (Notification.username == "")
        )
    count = query.update({Notification.is_read: 1})
    db.commit()
    return count
