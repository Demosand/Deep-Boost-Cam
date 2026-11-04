"""Last stage of Live: upscaler -> frame interpolation -> virtual camera + preview.

Shared by the processing worker and the reprojection presenter. With the
upscaler or interpolation on, this stage runs in its own thread, so it works
on frame N while the swap already processes N+1 (in the caller's thread the
two were measured to cost the swap a third of its rate). Without them frames
are delivered directly, as before. Overlay text (FPS, tracker state) is drawn
only on the preview copy, never into the virtual camera.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional, Sequence, Tuple

import cv2

import modules.globals as G
from modules import virtual_camera

Overlay = List[Tuple[str, int, float]]   # (text, y, font scale)


def _put_latest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class LiveOutput:
    def __init__(self, display_queue: queue.Queue, camera_fps: float, on_error=None):
        self._dq = display_queue
        self._fps = camera_fps
        self._on_error = on_error
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def send(self, frame, bboxes: Optional[Sequence] = None, overlay: Optional[Overlay] = None) -> None:
        if not (G.upscaler or G.frame_interpolation):
            self.close()
            self._deliver(frame, overlay)
            return
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="live-output", daemon=True)
            self._thread.start()
        _put_latest(self._q, (frame, bboxes, overlay, time.perf_counter()))

    def _run(self) -> None:
        prev = prev_t = None
        while not self._stop.is_set():
            try:
                frame, bboxes, overlay, t = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            # The midpoint is made from the frames before upscaling and only
            # resized: RIFE on the x2 picture cost ~34 ms and halved the output
            # rate. The in-between frame is on screen for half a frame.
            raw = frame
            if G.upscaler:
                from modules.processors.frame import upscaler
                frame = upscaler.upscale(frame, bboxes)
            if (G.frame_interpolation and prev is not None and prev.shape == raw.shape
                    and 0 < t - prev_t < 0.25):
                from modules.frame_interp import midpoint
                mid = midpoint(prev, raw)
                if mid is not None and mid.shape != frame.shape:
                    mid = cv2.resize(mid, frame.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                if mid is not None:
                    self._deliver(mid, overlay)
                    # Half a frame between the midpoint and the frame; release
                    # early if the next frame is already waiting (no backlog).
                    deadline = time.perf_counter() + (t - prev_t) / 2
                    while (time.perf_counter() < deadline and self._q.empty()
                           and not self._stop.is_set()):
                        time.sleep(0.001)
            self._deliver(frame, overlay)
            prev, prev_t = raw, t

    def _deliver(self, frame, overlay: Optional[Overlay]) -> None:
        if G.virtual_camera:
            fps = self._fps * (2 if G.frame_interpolation else 1)
            virtual_camera.send(frame, fps)
            if not G.virtual_camera and virtual_camera.last_error() and self._on_error:
                self._on_error(virtual_camera.last_error())
        if overlay:
            frame = frame.copy()
            for text, y, scale in overlay:
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
        _put_latest(self._dq, frame)

    def close(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
