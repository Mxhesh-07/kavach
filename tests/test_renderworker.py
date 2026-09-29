"""
RenderWorker: drop-oldest backpressure and fast teardown.
"""
import time

import numpy as np

from core.config import settings


def _make_processor(camera_id=9001):
    from core.database import SessionLocal, init_db
    init_db()  # CameraProcessor.__init__ → reload_rules() queries the rules table.
    from core.camera import CameraProcessor

    proc = CameraProcessor(camera_id=camera_id, url="0", name="RW")
    proc._stop_event.clear()
    return proc


def test_worker_start_join_fast(monkeypatch):
    """start() then stop()/join() must exit in microseconds, not seconds."""
    from core.camera import FrameBuffer

    proc = _make_processor()
    FrameBuffer.get().open(proc.camera_id)
    try:
        proc._render_worker.start()
        assert proc._render_worker.alive
        t0 = time.perf_counter()
        proc._render_worker.stop()
        joined = proc._render_worker.join(timeout=2.0)
        elapsed = time.perf_counter() - t0
        assert joined
        assert elapsed < 1.5  # generous; a drop-oldest worker wakes instantly
    finally:
        proc._render_worker.join(timeout=1.0)
        FrameBuffer.get().drop(proc.camera_id)


def test_submit_renders_and_publishes():
    """A submitted frame ends up encoded and in the FrameBuffer."""
    from core.camera import CameraProcessor, FrameBuffer

    proc = _make_processor()
    FrameBuffer.get().open(proc.camera_id)
    try:
        frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 120, np.uint8)
        result = proc.analyzer.analyse(frame, timestamp=1000.0)
        buffer = FrameBuffer.get()
        # Worker not alive -> synchronous fallback leaves a JPEG immediately.
        proc._publish(buffer, result, captured_at=1000.0)
        assert FrameBuffer.get().get_jpeg(proc.camera_id) is not None
        assert FrameBuffer.get().get_clean_frame(proc.camera_id) is not None
    finally:
        proc._render_worker.join(timeout=1.0)
        FrameBuffer.get().drop(proc.camera_id)


def test_teardown_wakes_and_joins_without_waiting_for_slow_slot():
    """stop() must discard the pending slot so join returns immediately."""
    import threading

    from core.camera import FrameBuffer

    proc = _make_processor()
    FrameBuffer.get().open(proc.camera_id)
    worker = proc._render_worker
    started = threading.Event()

    # Park a synthetic slow "render" in the slot by hand: set a slot, then
    # stop() — since _loop blocks on the condition only when the slot is empty,
    # a full slot read overlaps stop, which is fine.
    worker.start()
    try:
        frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 5, np.uint8)
        result = proc.analyzer.analyse(frame, timestamp=1.0)
        buffer = FrameBuffer.get()
        with worker._cond:
            worker._slot = (result, 1.0, buffer)
            worker._cond.notify_all()
        time.sleep(0.05)  # let the loop pick it up if it will
        t0 = time.perf_counter()
        worker.stop()
        ok = worker.join(timeout=1.0)
        elapsed = time.perf_counter() - t0
        assert ok
        assert elapsed < 1.0
    finally:
        worker.join(timeout=1.0)
        FrameBuffer.get().drop(proc.camera_id)


def test_render_frame_with_all_armed_rules():
    """render_frame must never freeze or drop frames when rules are armed."""
    from core.analytics import FrameAnalyzer
    from core.camera import FrameBuffer

    analyzer = FrameAnalyzer(source_id="9002", display_name="TestCam")
    analyzer.set_rules([
        {"id": 1, "rule_type": "line", "geometry": [[50, 50], [200, 200]], "name": "Tripwire"},
        {"id": 2, "rule_type": "zone", "geometry": [[100, 100], [300, 100], [300, 300], [100, 300]], "name": "Restricted"},
        {"id": 3, "rule_type": "loiter", "geometry": [[350, 100], [500, 100], [500, 300], [350, 300]], "name": "Lobby"},
        {"id": 4, "rule_type": "direction", "geometry": [[50, 300], [200, 450]], "name": "OneWay"},
    ])

    frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 100, np.uint8)
    result = analyzer.analyse(frame, timestamp=100.0, annotate=False)

    assert result.zones is not None
    assert len(result.zones) >= 2  # zone and loiter rules report occupancy

    # Must render without KeyError or any exception
    annotated = analyzer.render_frame(result)
    assert annotated is not None
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3)


def test_submit_with_armed_rules_publishes_to_framebuffer():
    """Worker renders and publishes to FrameBuffer when camera has rules configured."""
    from core.camera import FrameBuffer

    proc = _make_processor(camera_id=9003)
    FrameBuffer.get().open(proc.camera_id)
    try:
        proc.analyzer.set_rules([
            {"id": 10, "rule_type": "zone", "geometry": [[20, 20], [200, 20], [200, 200]], "name": "ZoneA"},
            {"id": 11, "rule_type": "line", "geometry": [[10, 50], [500, 50]], "name": "FenceB"},
        ])
        proc._render_worker.start()

        frame = np.full((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), 80, np.uint8)
        result = proc.analyzer.analyse(frame, timestamp=500.0, annotate=False)
        buffer = FrameBuffer.get()

        proc._publish(buffer, result, captured_at=500.0)

        # Give worker a moment to process the slot
        for _ in range(20):
            if buffer.get_jpeg(proc.camera_id) is not None:
                break
            time.sleep(0.05)

        assert buffer.get_jpeg(proc.camera_id) is not None
        assert buffer.get_clean_frame(proc.camera_id) is not None
        assert buffer.sequence(proc.camera_id) > 0
    finally:
        proc._render_worker.stop()
        proc._render_worker.join(timeout=1.0)
        FrameBuffer.get().drop(proc.camera_id)