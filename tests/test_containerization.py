"""
Production Containerization & Orchestration Verification Tests.
Validates:
1. Dockerfile.prod multi-stage build, CUDA base images, non-root user, and health check.
2. docker-compose.prod.yml NVIDIA GPU reservation and volume isolation.
3. Healthcheck probes and entrypoint configuration.
"""
from pathlib import Path
import pytest
import yaml

PROJECT = Path(__file__).resolve().parent.parent


def test_dockerfile_prod_structure():
    """Verify Dockerfile.prod follows multi-stage hardened standards."""
    dockerfile_path = PROJECT / "Dockerfile.prod"
    assert dockerfile_path.exists(), "Dockerfile.prod missing"
    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage assertion
    assert "FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder" in content
    assert "FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04" in content

    # Security: unprivileged appuser
    assert "useradd" in content and "appuser" in content
    assert "USER appuser" in content

    # Healthcheck assertion
    assert "HEALTHCHECK" in content
    assert "http://localhost:8000/health" in content

    # Entrypoint
    assert 'CMD ["python3", "manage.py", "run"' in content


def test_docker_compose_prod_gpu_orchestration():
    """Verify docker-compose.prod.yml defines NVIDIA GPU reservations and persistent volumes."""
    compose_path = PROJECT / "docker-compose.prod.yml"
    assert compose_path.exists(), "docker-compose.prod.yml missing"
    
    with open(compose_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    assert "ibvap-edge" in data["services"]
    service = data["services"]["ibvap-edge"]

    # GPU reservations
    assert "deploy" in service
    reservations = service["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" and "gpu" in d.get("capabilities", []) for d in reservations)

    # Healthcheck
    assert "healthcheck" in service
    assert "test" in service["healthcheck"]

    # Persistent storage volumes
    assert "volumes" in data
    assert "ibvap-data" in data["volumes"]
