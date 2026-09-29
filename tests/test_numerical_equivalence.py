"""
Phase 14: Numerical Equivalence & Output Validation Tests.
Compares bounding boxes, class predictions, and score confidence between
baseline PyTorch detector and native pipeline interfaces.
"""
import pytest
import numpy as np
from cv.detector import Detector
from core.backend import BackendDetector


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def test_detector_output_contract():
    """Ensure Detector produces expected types, dimensions, and normalized ranges."""
    detector = Detector.get()
    
    # Test on synthetic canvas
    test_frame = np.full((640, 640, 3), 128, dtype=np.uint8)
    boxes, confs, classes, elapsed = detector.raw_detect(test_frame)
    
    assert isinstance(boxes, np.ndarray)
    assert isinstance(confs, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert elapsed >= 0.0
    
    if len(boxes) > 0:
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert np.all(boxes[:, 0] >= 0)
        assert np.all(boxes[:, 1] >= 0)
        assert np.all(boxes[:, 2] <= 640)
        assert np.all(boxes[:, 3] <= 640)
        assert np.all(confs >= 0.0)
        assert np.all(confs <= 1.0)


def test_backend_detector_equivalence():
    """Verify BackendDetector interface is functionally equivalent to direct Detector calls."""
    backend = BackendDetector()
    direct = Detector.get()
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run both
    res_backend = backend.detect(frame)
    res_direct = direct.detect(frame)
    
    # Should yield exact detection count for identical inputs
    assert len(res_backend) == len(res_direct)
    for d1, d2 in zip(res_backend, res_direct):
        assert d1.class_id == d2.class_id
        assert pytest.approx(d1.confidence, rel=1e-3) == d2.confidence
        assert compute_iou(d1.bbox, d2.bbox) >= 0.99
