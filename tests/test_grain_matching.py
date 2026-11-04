import unittest

import cv2
import numpy as np

import modules.globals as G
from modules.processors.frame import realism


def _textured(seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(60, 200, (240, 240, 3)).astype(np.float32), (0, 0), 3)
    return np.clip(base + rng.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def _noise_sigma(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))
    return 1.4826 * float(np.median(r))


class GrainMatchingTest(unittest.TestCase):
    def setUp(self):
        self.saved = (G.grain_amount, G.grain_sharpness_amount, G.many_faces)
        G.grain_amount, G.grain_sharpness_amount, G.many_faces = 1.0, 0.7, False
        realism._grain_state["stats"] = None

    def tearDown(self):
        G.grain_amount, G.grain_sharpness_amount, G.many_faces = self.saved
        realism._grain_state["stats"] = None

    def test_smooth_face_gets_camera_grain(self):
        target = _textured(1, noise=6.0)              # noisy webcam frame
        fake = cv2.GaussianBlur(_textured(2, 0.0), (0, 0), 1.5)   # smooth swap output
        alpha = np.full((240, 240), 255, np.uint8)
        out = realism.match_grain(fake, target, alpha)
        self.assertLess(_noise_sigma(fake), _noise_sigma(target) * 0.5)
        ratio = _noise_sigma(out) / _noise_sigma(target)
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.5)

    def test_zero_amounts_leave_face_unchanged(self):
        G.grain_amount, G.grain_sharpness_amount = 0.0, 0.0
        target = _textured(1, noise=6.0)
        fake = _textured(2, 0.0)
        out = realism.match_grain(fake, target, np.full((240, 240), 255, np.uint8))
        self.assertTrue(np.array_equal(out, fake))


if __name__ == "__main__":
    unittest.main()
