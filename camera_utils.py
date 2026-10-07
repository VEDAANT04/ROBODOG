"""
Threaded camera capture.

A plain `cv2.VideoCapture.read()` call blocks the calling thread until the
driver hands over a frame. In a script that also runs detection/recognition
in the same loop, I/O and processing happen back-to-back instead of in
parallel, which is the main source of "laggy" video.

VideoStream runs the blocking read() in a background thread and always
keeps the latest frame ready, so the main thread never waits on the camera.
"""

import sys
import threading
import time

import cv2


class VideoStream:

    def __init__(self, src=0, width=640, height=480, fps=30, warmup_frames=10):

        # CAP_DSHOW opens noticeably faster than the default backend on Windows.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

        self.stream = cv2.VideoCapture(src, backend)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = False
        self._thread = None

        if self.stream.isOpened():
            for _ in range(warmup_frames):
                self.stream.read()
            ret, frame = self.stream.read()
            self._ret, self._frame = ret, frame

    def isOpened(self):
        return self.stream.isOpened()

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while self._running:
            ret, frame = self.stream.read()
            with self._lock:
                self._ret, self._frame = ret, frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.stream.release()

    # Alias kept for drop-in compatibility with cv2.VideoCapture call sites.
    def release(self):
        self.stop()
