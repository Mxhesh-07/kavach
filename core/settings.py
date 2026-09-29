"""
Runtime system settings.

Provides a database-backed key-value store for system configuration
that can be updated at runtime through the API, without restarting
the server or editing .env files.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.models import SystemSetting
from core.timeutil import utc_iso, fmt_ist

log = logging.getLogger("kavach.settings")


def get_setting(db: Session, key: str, default: Any = None) -> Any:
    """Get a setting value by key, returning default if not found."""
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting is None:
        return default
    return setting.typed_value()


def get_all_settings(db: Session) -> list[dict]:
    """Return all system settings as a list of dicts."""
    settings = db.query(SystemSetting).order_by(SystemSetting.key).all()
    return [
        {
            "key": s.key,
            "value": s.typed_value(),
            "value_type": s.value_type,
            "category": s.category,
            "description": s.description,
        }
        for s in settings
    ]


def get_settings_map(db: Session) -> dict[str, Any]:
    """Return all settings as a {key: value} dict."""
    settings = db.query(SystemSetting).all()
    return {s.key: s.typed_value() for s in settings}


def update_setting(db: Session, key: str, value: Any) -> Optional[SystemSetting]:
    """Update a single setting. Returns the updated setting or None if key not found."""
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting is None:
        return None

    # Convert value to string for storage
    if isinstance(value, bool):
        setting.value = "true" if value else "false"
        setting.value_type = "boolean"
    elif isinstance(value, int):
        setting.value = str(value)
        setting.value_type = "integer"
    elif isinstance(value, float):
        setting.value = str(value)
        setting.value_type = "float"
    elif isinstance(value, (dict, list)):
        setting.value = json.dumps(value)
        setting.value_type = "json"
    else:
        setting.value = str(value)
        setting.value_type = "string"

    setting.updated_at = utc_iso()
    setting.updated_at_ist = fmt_ist()
    db.commit()
    db.refresh(setting)
    return setting


def update_settings(db: Session, updates: dict[str, Any]) -> list[str]:
    """Bulk-update settings. Returns list of updated keys."""
    updated_keys = []
    for key, value in updates.items():
        setting = update_setting(db, key, value)
        if setting is not None:
            updated_keys.append(key)
    return updated_keys


def seed_default_settings(db: Session) -> None:
    """
    Seed default settings if the table is empty.

    These are sensible defaults that can be overridden at runtime.
    """
    defaults = [
        ("system_name", "KAVACH Command Center", "string", "general", "Display name for this installation"),
        ("operator_name", "", "string", "general", "Name of the current duty operator"),
        ("alert_retention_days", "30", "integer", "evidence", "Days to retain alert evidence"),
        ("max_concurrent_streams", "12", "integer", "streaming", "Maximum concurrent MJPEG viewers per camera"),
        ("enable_audio_alerts", "false", "boolean", "alerts", "Enable audio alerts on new events"),
        ("dashboard_refresh_seconds", "5", "integer", "ui", "Dashboard auto-refresh interval in seconds"),
        ("timezone_display", "Asia/Kolkata", "string", "ui", "Timezone for display purposes"),
        ("theme", "dark", "string", "ui", "UI theme (dark/light)"),
        ("language", "en", "string", "ui", "Interface language"),
        ("session_timeout_minutes", "60", "integer", "security", "Operator session timeout in minutes"),
    ]

    for key, value, vtype, category, description in defaults:
        existing = db.query(SystemSetting).filter(SystemSetting.key == key).first()
        if existing is None:
            setting = SystemSetting(
                key=key,
                value=value,
                value_type=vtype,
                category=category,
                description=description,
                updated_at=utc_iso(),
                updated_at_ist=fmt_ist(),
            )
            db.add(setting)

    db.commit()
