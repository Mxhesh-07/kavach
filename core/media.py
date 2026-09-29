"""
Media / evidence file management.

Provides listing, searching, and deletion of evidence files
(snapshots, clips, ANPR crops, face crops) with proper access control.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from core import models
from core.evidence import is_safe_evidence_path

log = logging.getLogger("kavach.media")


def list_media_files(
    db: Session,
    camera_id: Optional[int] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    List evidence media files with optional filtering.

    Searches across snapshots, clips, ANPR evidence, and face evidence.
    """
    results = []

    # Snapshots from alerts
    alert_query = db.query(models.Alert).filter(
        models.Alert.snapshot_path != ""
    )
    if camera_id:
        alert_query = alert_query.filter(models.Alert.camera_id == camera_id)
    if file_type == "snapshot":
        alert_query = alert_query.filter(models.Alert.snapshot_path != "")
    elif file_type == "clip":
        alert_query = alert_query.filter(models.Alert.clip_path != "")

    for alert in alert_query.order_by(desc(models.Alert.id)).limit(limit).all():
        if alert.snapshot_path and is_safe_evidence_path(alert.snapshot_path):
            results.append({
                "id": f"snap_{alert.id}",
                "type": "snapshot",
                "path": alert.snapshot_path,
                "camera_id": alert.camera_id,
                "alert_id": alert.id,
                "alert_type": alert.alert_type,
                "timestamp": alert.timestamp,
                "timestamp_ist": alert.timestamp_ist,
            })
        if alert.clip_path and is_safe_evidence_path(alert.clip_path):
            results.append({
                "id": f"clip_{alert.id}",
                "type": "clip",
                "path": alert.clip_path,
                "camera_id": alert.camera_id,
                "alert_id": alert.id,
                "alert_type": alert.alert_type,
                "timestamp": alert.timestamp,
                "timestamp_ist": alert.timestamp_ist,
            })

    # ANPR evidence
    anpr_query = db.query(models.ANPRDetection).filter(
        models.ANPRDetection.evidence_path != ""
    )
    if camera_id:
        anpr_query = anpr_query.filter(models.ANPRDetection.camera_id == camera_id)

    for det in anpr_query.order_by(desc(models.ANPRDetection.id)).limit(limit).all():
        if det.evidence_path and is_safe_evidence_path(det.evidence_path):
            results.append({
                "id": f"anpr_{det.id}",
                "type": "anpr_evidence",
                "path": det.evidence_path,
                "camera_id": det.camera_id,
                "alert_id": det.alert_id,
                "plate_text": det.plate_text,
                "timestamp": det.timestamp,
                "timestamp_ist": det.timestamp_ist,
            })

    # Face evidence
    face_query = db.query(models.FaceDetection).filter(
        models.FaceDetection.evidence_path != ""
    )
    if camera_id:
        face_query = face_query.filter(models.FaceDetection.camera_id == camera_id)

    for det in face_query.order_by(desc(models.FaceDetection.id)).limit(limit).all():
        if det.evidence_path and is_safe_evidence_path(det.evidence_path):
            results.append({
                "id": f"face_{det.id}",
                "type": "face_evidence",
                "path": det.evidence_path,
                "camera_id": det.camera_id,
                "alert_id": det.alert_id,
                "identity_name": det.identity_name,
                "timestamp": det.timestamp,
                "timestamp_ist": det.timestamp_ist,
            })

    # Sort by timestamp descending
    results.sort(key=lambda x: x["timestamp"], reverse=True)

    total = len(results)
    paginated = results[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "files": paginated,
    }


def delete_media_file(db: Session, file_id: str) -> dict:
    """
    Delete a media file from disk and remove its reference.

    file_id format: "snap_123", "clip_123", "anpr_123", "face_123"
    """
    parts = file_id.split("_", 1)
    if len(parts) != 2:
        return {"ok": False, "error": "Invalid file ID format"}

    prefix, row_id = parts
    row_id = int(row_id)

    path = None

    if prefix == "snap":
        alert = db.query(models.Alert).filter(models.Alert.id == row_id).first()
        if alert and alert.snapshot_path:
            path = alert.snapshot_path
            alert.snapshot_path = ""
    elif prefix == "clip":
        alert = db.query(models.Alert).filter(models.Alert.id == row_id).first()
        if alert and alert.clip_path:
            path = alert.clip_path
            alert.clip_path = ""
    elif prefix == "anpr":
        det = db.query(models.ANPRDetection).filter(models.ANPRDetection.id == row_id).first()
        if det and det.evidence_path:
            path = det.evidence_path
            det.evidence_path = ""
    elif prefix == "face":
        det = db.query(models.FaceDetection).filter(models.FaceDetection.id == row_id).first()
        if det and det.evidence_path:
            path = det.evidence_path
            det.evidence_path = ""
    else:
        return {"ok": False, "error": f"Unknown file type: {prefix}"}

    if not path:
        return {"ok": False, "error": "File reference not found"}

    if not is_safe_evidence_path(path):
        return {"ok": False, "error": "File path is outside permitted store"}

    # Delete from disk
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            log.info("Deleted media file: %s", path)
    except OSError as exc:
        log.warning("Could not delete file %s: %s", path, exc)
        return {"ok": False, "error": f"Could not delete file: {exc}"}

    db.commit()
    return {"ok": True, "deleted": path}


def get_media_stats(db: Session) -> dict:
    """Get statistics about stored media files."""
    snapshot_count = db.query(models.Alert).filter(models.Alert.snapshot_path != "").count()
    clip_count = db.query(models.Alert).filter(models.Alert.clip_path != "").count()
    anpr_count = db.query(models.ANPRDetection).filter(models.ANPRDetection.evidence_path != "").count()
    face_count = db.query(models.FaceDetection).filter(models.FaceDetection.evidence_path != "").count()

    return {
        "snapshots": snapshot_count,
        "clips": clip_count,
        "anpr_evidence": anpr_count,
        "face_evidence": face_count,
        "total": snapshot_count + clip_count + anpr_count + face_count,
    }
