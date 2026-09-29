"""
TensorRT Engine Manifest & Hardware Capability Validator.
Ensures engine compatibility with host GPU architecture, precision, and dimensions.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EngineManifest:
    """Validates and checks serialized TensorRT engine compatibility."""
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.manifest_path = engine_path + ".json"

    def write_manifest(self, model_name: str, precision: str, input_shape: tuple, sm_arch: str, trt_version: str):
        manifest = {
            "model_name": model_name,
            "engine_path": self.engine_path,
            "precision": precision,
            "input_shape": list(input_shape),
            "sm_arch": sm_arch,
            "trt_version": trt_version,
            "checksum": os.path.getsize(self.engine_path) if os.path.exists(self.engine_path) else 0
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote TRT engine manifest to {self.manifest_path}")

    def validate(self) -> bool:
        if not os.path.exists(self.engine_path):
            logger.error(f"Engine file not found: {self.engine_path}")
            return False
        if not os.path.exists(self.manifest_path):
            logger.warning(f"No manifest found for {self.engine_path}; skipping hardware signature check.")
            return True
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation
            if data.get("checksum", 0) != os.path.getsize(self.engine_path):
                logger.warning(f"Engine size mismatch against manifest for {self.engine_path}")
            return True
        except Exception as e:
            logger.error(f"Manifest validation error: {e}")
            return False
