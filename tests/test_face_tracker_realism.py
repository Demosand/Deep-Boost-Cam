import unittest

import cv2
import numpy as np
from insightface.app.common import Face

import modules.globals as G
from modules.face_tracker import FaceTracker
from modules.processors.frame import realism


def _scene(dx: int) -> np.ndarray:
    """Textured 'face' patch on a textured background, shifted by dx pixels."""
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
    patch = rng.integers(80, 255, (160, 160, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    frame = bg.copy()
    frame[160:320, 200 + dx:360 + dx] = patch
    return frame


def _face(dx: float = 0.0, score: float = 0.9) -> Face:
    kps = np.array([[240, 200], [320, 200], [280, 240], [250, 280], [310, 280]], np.float32)
    kps[:, 0] += dx
    return Face(bbox=np.array([200 + dx, 160, 360 + dx, 320], np.float32),
                kps=kps, det_score=np.float32(score))


class FaceTrackerTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(G, k) for k in dir(G) if k.startswith("tracking_")}
        G.tracking_optical_flow = True
        G.tracking_smoothing = False
        G.tracking_grace_frames = 3
        G.tracking_auto_redetect = True

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(G, k, v)

    def test_flow_follows_motion_between_detections(self):
        t = FaceTracker()
        t.update(_scene(0), True, _face(0))
        out = t.update(_scene(6), False, None)
        self.assertEqual(t.state, "flow")
        self.assertAlmostEqual(float(out.kps[:, 0].mean() - _face(0).kps[:, 0].mean()), 6.0, delta=1.0)

    def test_grace_period_then_reset(self):
        t = FaceTracker()
        t.update(_scene(0), True, _face(0))
        blank = np.zeros((480, 640, 3), np.uint8)
        held = [t.update(blank, True, None) is not None for _ in range(5)]
        # 3 grace frames held, then dropped (never held forever).
        self.assertEqual(held, [True, True, True, False, False])
        self.assertIsNone(t.face)

    def test_motion_limit_rejects_jump(self):
        G.tracking_max_motion = 0.05
        t = FaceTracker()
        t.update(_scene(0), True, _face(0))
        t.update(_scene(30), False, None)   # 30px of a 160px face > 5%
        self.assertEqual(t.state, "hold")
        self.assertTrue(t.need_detect)

    def test_low_confidence_counts_as_lost(self):
        G.tracking_min_confidence = 0.5
        t = FaceTracker()
        self.assertIsNone(t.update(_scene(0), True, _face(0, score=0.3)))

    def test_smoothing_reduces_jitter(self):
        G.tracking_smoothing = True
        G.tracking_smoothing_strength = 1.0
        G.tracking_optical_flow = False
        t = FaceTracker()
        rng = np.random.default_rng(1)
        xs = []
        for _ in range(40):
            out = t.update(_scene(0), True, _face(float(rng.normal(0, 2))))
            xs.append(float(out.kps[:, 0].mean()))
        self.assertLess(np.std(xs[10:]), 1.0)


class RealismTest(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(G, k) for k in (
            "color_matching", "temporal_stabilization", "eye_preservation",
            "mouth_preservation", "adaptive_face_mask")}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(G, k, v)
        realism.reset_state()

    def test_all_off_is_noop(self):
        for k in self.saved:
            setattr(G, k, False)
        self.assertFalse(realism.any_enabled())

    def test_refine_shapes_and_color_shift(self):
        for k in self.saved:
            setattr(G, k, True)
        frame = np.full((480, 640, 3), (60, 90, 160), np.uint8)
        fake = np.full((128, 128, 3), (200, 200, 200), np.uint8)
        face = _face()
        face.landmark_2d_106 = np.random.default_rng(2).uniform(220, 340, (106, 2)).astype(np.float32)
        M = np.array([[0.8, 0, -150], [0, 0.8, -120]], np.float32)
        ellipse = np.full((128, 128), 255, np.uint8)
        out, alpha = realism.refine(fake, frame, face, M, ellipse)
        self.assertEqual(out.shape, fake.shape)
        self.assertEqual(alpha.shape, (128, 128))
        # Colour matching pulls the grey fake toward the target's skin tone.
        self.assertGreater(out[64, 64, 2] - out[64, 64, 0], 20)


if __name__ == "__main__":
    unittest.main()
