"""
Per-component system health monitoring.

Provides detailed health status for each subsystem: database, cameras,
inference engine, face recognition, ANPR, disk space, and more.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from core import models
from core.config import settings
from core.database import engine
from core.evidence import measure_evidence_usage
from core.timeutil import fmt_ist

log = logging.getLogger("ibvap.health")


def _probe_system_metrics() -> dict:
    """Best-effort CPU / RAM / GPU metrics."""
    cpu = ram = gpu = None
    try:
        import psutil
        cpu = round(psutil.cpu_percent(interval=0.1), 1)
        ram = round(psutil.virtual_memory().percent, 1)
    except ImportError:
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu = round(util.gpu, 1)
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return {"cpu_percent": cpu, "ram_percent": ram, "gpu_percent": gpu}


def check_database_health() -> dict:
    """Check database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(func.count(models.Camera.id))
        return {"status": "HEALTHY", "detail": "Database query succeeded"}
    except Exception as exc:
        return {"status": "CRITICAL", "detail": f"Database error: {exc}"}


def check_disk_health() -> dict:
    """Check disk space on evidence storage."""
    try:
        usage = shutil.disk_usage(settings.ALERTS_DIR)
        free_pct = round((usage.free / usage.total) * 100, 1)
        status = "HEALTHY" if free_pct > 20 else ("WARNING" if free_pct > 10 else "CRITICAL")
        return {
            "status": status,
            "detail": f"{free_pct}% free ({usage.free // (1024**3)} GB of {usage.total // (1024**3)} GB)",
        }
    except Exception as exc:
        return {"status": "CRITICAL", "detail": f"Disk check failed: {exc}"}


def get_system_health(db: Session) -> list[dict]:
    """
    Comprehensive per-component health check.

    Returns a list of component health records.
    """
    components = []
    now = datetime.now(timezone.utc).isoformat()

    # Database
    db_health = check_database_health()
    components.append({
        "component": "database",
        "status": db_health["status"],
        "value": settings.DATABASE_URL.split(":///")[-1] if ":///" in settings.DATABASE_URL else "local",
        "detail": db_health["detail"],
        "updated_at": now,
    })

    # Disk
    disk_health = check_disk_health()
    components.append({
        "component": "disk",
        "status": disk_health["status"],
        "value": f"{settings.ALERTS_DIR}",
        "detail": disk_health["detail"],
        "updated_at": now,
    })

    # CPU / RAM
    metrics = _probe_system_metrics()
    cpu = metrics["cpu_percent"]
    ram = metrics["ram_percent"]
    cpu_status = "HEALTHY" if cpu is None or cpu < 80 else ("WARNING" if cpu < 95 else "CRITICAL")
    components.append({
        "component": "cpu",
        "status": cpu_status,
        "value": cpu,
        "detail": f"CPU usage: {cpu}%" if cpu is not None else "psutil not available",
        "updated_at": now,
    })
    ram_status = "HEALTHY" if ram is None or ram < 80 else ("WARNING" if ram < 95 else "CRITICAL")
    components.append({
        "component": "memory",
        "status": ram_status,
        "value": ram,
        "detail": f"RAM usage: {ram}%" if ram is not None else "psutil not available",
        "updated_at": now,
    })

    # GPU
    gpu = metrics["gpu_percent"]
    if gpu is not None:
        gpu_status = "HEALTHY" if gpu < 80 else ("WARNING" if gpu < 95 else "CRITICAL")
        components.append({
            "component": "gpu",
            "status": gpu_status,
            "value": gpu,
            "detail": f"GPU usage: {gpu}%",
            "updated_at": now,
        })

    # Cameras
    cam_total = db.query(func.count(models.Camera.id)).scalar() or 0
    cam_online = db.query(func.count(models.Camera.id)).filter(
        models.Camera.is_online == True
    ).scalar() or 0
    cam_status = "HEALTHY" if cam_online == cam_total else ("WARNING" if cam_online > 0 else "CRITICAL")
    components.append({
        "component": "cameras",
        "status": cam_status,
        "value": f"{cam_online}/{cam_total}",
        "detail": f"{cam_online} online, {cam_total - cam_online} offline",
        "updated_at": now,
    })

    # Inference engine
    try:
        from cv.detector import Detector
        det = Detector.get()
        det_metrics = det.metrics()
        components.append({
            "component": "inference",
            "status": "HEALTHY",
            "value": det_metrics.get("model", "unknown"),
            "detail": f"Model: {det_metrics.get('model', 'unknown')}, Device: {det_metrics.get('device', 'unknown')}",
            "updated_at": now,
        })
    except Exception as exc:
        components.append({
            "component": "inference",
            "status": "CRITICAL",
            "value": "unavailable",
            "detail": str(exc),
            "updated_at": now,
        })

    # Face recognition
    try:
        from cv.face import get_face_recognizer
        face = get_face_recognizer()
        face_metrics = face.get_metrics()
        face_status = "HEALTHY" if face_metrics.get("enabled") else "WARNING"
        components.append({
            "component": "face_recognition",
            "status": face_status,
            "value": "enabled" if face_metrics.get("enabled") else "disabled",
            "detail": f"Watchlist entries: {face_metrics.get('watchlist_count', 0)}",
            "updated_at": now,
        })
    except Exception as exc:
        components.append({
            "component": "face_recognition",
            "status": "CRITICAL",
            "value": "unavailable",
            "detail": str(exc),
            "updated_at": now,
        })

    # ANPR
    try:
        from cv.anpr import get_anpr_processor
        anpr = get_anpr_processor()
        anpr_metrics = anpr.get_metrics()
        anpr_status = "HEALTHY" if anpr_metrics.get("available") else "WARNING"
        components.append({
            "component": "anpr",
            "status": anpr_status,
            "value": "available" if anpr_metrics.get("available") else "unavailable",
            "detail": f"Languages: {anpr_metrics.get('languages', 'unknown')}",
            "updated_at": now,
        })
    except Exception as exc:
        components.append({
            "component": "anpr",
            "status": "CRITICAL",
            "value": "unavailable",
            "detail": str(exc),
            "updated_at": now,
        })

    # Evidence storage
    try:
        usage = measure_evidence_usage()
        components.append({
            "component": "evidence_storage",
            "status": "HEALTHY",
            "value": f"{usage.get('total_mb', 0):.1f} MB",
            "detail": f"{usage.get('file_count', 0)} files, {usage.get('total_mb', 0):.1f} MB total",
            "updated_at": now,
        })
    except Exception as exc:
        components.append({
            "component": "evidence_storage",
            "status": "WARNING",
            "value": "unknown",
            "detail": str(exc),
            "updated_at": now,
        })

    return components


def get_overall_status(components: list[dict]) -> str:
    """Derive overall system status from component statuses."""
    statuses = [c["status"] for c in components]
    if "CRITICAL" in statuses:
        return "CRITICAL"
    if "WARNING" in statuses:
        return "WARNING"
    return "HEALTHY"
