import unittest

import cv2
import numpy as np
from insightface.app.common import Face

from modules import source_identity
from modules.live_pipeline import Reprojector, shift_faces


def _scene(dx: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 60, (480, 640, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(rng.integers(80, 255, (160, 160, 3), dtype=np.uint8), (3, 3), 0)
    frame[160:320, 200 + dx:360 + dx] = patch
    return frame


def _face(dx: float = 0.0) -> Face:
    kps = np.array([[240, 200], [320, 200], [280, 240], [250, 280], [310, 280]], np.float32)
    kps[:, 0] += dx
    return Face(bbox=np.array([200 + dx, 160, 360 + dx, 320], np.float32), kps=kps,
                det_score=np.float32(0.9))


class ShiftFacesTest(unittest.TestCase):
    def test_detection_carried_to_newer_frame(self):
        moved = shift_faces(_face(0), _scene(0), _scene(8))
        self.assertAlmostEqual(float(moved.kps[:, 0].mean() - _face(0).kps[:, 0].mean()), 8.0, delta=1.0)


class ReprojectorTest(unittest.TestCase):
    def test_processed_face_follows_newest_frame(self):
        r = Reprojector()
        r.on_raw(1, _scene(0))
        processed = _scene(0)
        processed[180:300, 220:340] = (0, 0, 255)          # "swapped" face region
        r.publish(1, processed, _face(0), raw=_scene(0))
        r.on_raw(2, _scene(6))
        r.on_raw(3, _scene(12))
        out = r.compose(3, _scene(12))
        red = (out[:, :, 2] > 200) & (out[:, :, 0] < 50)
        xs = np.where(red.any(axis=0))[0]
        # The red region moved with the face by ~12 px.
        self.assertAlmostEqual((xs.min() + xs.max()) / 2, 280 + 12, delta=4)

    def test_no_face_shows_processed_frame(self):
        r = Reprojector()
        r.on_raw(1, _scene(0))
        frozen = np.zeros((480, 640, 3), np.uint8)
        r.publish(1, frozen, None)
        self.assertIs(r.compose(2, _scene(5)), frozen)


class SourceIdentityTest(unittest.TestCase):
    def _src(self, yaw_px, emb):
        f = _face(0)
        f.kps[2, 0] += yaw_px                 # nose shift = head turn proxy
        f.embedding = np.asarray(emb, np.float32)
        return f

    def test_weights_prefer_matching_pose(self):
        front = self._src(0, np.r_[1.0, np.zeros(511)])
        turned = self._src(25, np.r_[0.0, 1.0, np.zeros(510)])
        ident = source_identity.SourceIdentity([front, turned], ["a", "b"])
        w_front = ident.weights(_face(0).kps)
        target = _face(0)
        target.kps[2, 0] += 25
        w_turned = ident.weights(target.kps)
        self.assertGreater(w_front[0], w_front[1])
        self.assertGreater(w_turned[1], w_turned[0])
        self.assertAlmostEqual(float(w_front.sum()), 1.0, places=5)

    def test_blend_keeps_unit_normed_embedding(self):
        a = self._src(0, np.r_[3.0, np.zeros(511)])
        b = self._src(10, np.r_[0.0, 3.0, np.zeros(510)])
        face = source_identity.SourceIdentity([a, b], ["a", "b"]).face_for(_face(0))
        self.assertAlmostEqual(float(np.linalg.norm(face.normed_embedding)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
