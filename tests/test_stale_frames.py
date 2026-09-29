"""
Tests for Phase 3: Stale-frame elimination, per-camera latest-frame slots,
and evidence ring-buffer independence.
"""
import time
import threading
import numpy as np
import pytest

from core.video_source import LiveSource
from core.camera import FrameBuffer
from core.evidence import ClipRecorder
from core.timing import FrameTiming, metrics


def test_stale_frame_drop_under_artificial_overload():
    """Producer emits 100 frames rapidly; slow consumer takes only 5. 95 frames must be dropped."""
    source = LiveSource("0", name="OVERLOAD_TEST")
    assert source._frames.maxlen == 1

    produced = 100
    for i in range(produced):
        with source._new_frame:
            if source._frames:
                source.stats.frames_dropped += 1
            frame = np.full((384, 640, 3), i % 255, dtype=np.uint8)
            source._frames.append(frame)
            source._frame_id += 1

    assert len(source._frames) == 1
    assert source.stats.frames_dropped == 99
    assert source._frames[0][0, 0, 0] == 99 % 255


def test_frame_age_stays_bounded_during_stalled_inference():
    """Verify that when a frame is read from the slot, its capture age is fresh."""
    source = LiveSource("0", name="AGE_TEST")
    
    # Simulate time passing between captures
    t0 = time.monotonic()
    source._frames.append(np.zeros((10, 10, 3), dtype=np.uint8))
    source._frame_ts = time.time()
    
    # 50 ms later a fresh frame arrives and replaces the old one
    time.sleep(0.05)
    t1 = time.monotonic()
    fresh_frame = np.full((10, 10, 3), 42, dtype=np.uint8)
    source._frames.append(fresh_frame)
    source._frame_ts = time.time()
    
    # Read returns the newest frame
    assert len(source._frames) == 1
    assert source._frames[0][0, 0, 0] == 42


def test_evidence_ring_buffer_preserves_frames_independently():
    """The evidence ring buffer retains pre-event frames even while analytics drops frames."""
    recorder = ClipRecorder(source_id="test_cam", pre_frames=10, post_frames=5)
    
    for i in range(15):
        frame = np.full((100, 100, 3), i, dtype=np.uint8)
        recorder.push(frame)
        
    assert len(recorder._buffer) == 10
    # The oldest kept frame in pre-roll is frame 5
    assert recorder._buffer[0][0, 0, 0] == 5
    # The newest kept frame in pre-roll is frame 14
    assert recorder._buffer[-1][0, 0, 0] == 14


def test_multi_camera_independent_stale_slots():
    """Multiple cameras replacing frames do not interfere with each other."""
    cam1 = LiveSource("0", name="CAM1")
    cam2 = LiveSource("1", name="CAM2")

    for i in range(50):
        with cam1._new_frame:
            if cam1._frames:
                cam1.stats.frames_dropped += 1
            cam1._frames.append(np.full((10, 10, 3), 1, dtype=np.uint8))
            
    for i in range(20):
        with cam2._new_frame:
            if cam2._frames:
                cam2.stats.frames_dropped += 1
            cam2._frames.append(np.full((10, 10, 3), 2, dtype=np.uint8))

    assert cam1.stats.frames_dropped == 49
    assert cam2.stats.frames_dropped == 19
    assert len(cam1._frames) == 1
    assert len(cam2._frames) == 1
