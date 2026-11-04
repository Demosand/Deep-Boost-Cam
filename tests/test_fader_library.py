import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from insightface.app.common import Face

import modules.globals as G
from modules import face_fader, face_library


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FaderTest(unittest.TestCase):
    def setUp(self):
        self.saved = (G.face_fader, G.fader_seconds, G.swap_enabled, G.profile_fade)
        G.face_fader, G.fader_seconds, G.swap_enabled, G.profile_fade = True, 1.0, True, False
        self.clock = _Clock()
        self.p = patch.object(face_fader.time, "perf_counter", self.clock)
        self.p.start()
        face_fader.reset()

    def tearDown(self):
        self.p.stop()
        G.face_fader, G.fader_seconds, G.swap_enabled, G.profile_fade = self.saved
        face_fader.reset()

    def test_swap_off_ramps_down(self):
        self.assertEqual(face_fader.strength(), 1.0)
        G.swap_enabled = False
        values = []
        for _ in range(5):
            self.clock.t += 0.25
            values.append(face_fader.strength())
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertGreater(values[1], 0.0)
        self.assertEqual(values[-1], 0.0)

    def test_fader_off_is_instant(self):
        G.face_fader = False
        G.swap_enabled = False
        self.assertEqual(face_fader.strength(), 0.0)

    def test_identity_morphs_between_people(self):
        a = Face(embedding=np.r_[20.0, np.zeros(511)].astype(np.float32))
        b = Face(embedding=np.r_[0.0, 20.0, np.zeros(510)].astype(np.float32))
        face_fader.blend_identity(a)
        start = face_fader.blend_identity(b).normed_embedding   # morph starts now
        self.assertAlmostEqual(float(start[0]), 1.0, places=3)
        self.clock.t += 0.5
        mid = face_fader.blend_identity(b).normed_embedding
        self.assertGreater(mid[0], 0.1)
        self.assertGreater(mid[1], 0.1)
        self.clock.t += 1.0
        end = face_fader.blend_identity(b).normed_embedding
        self.assertAlmostEqual(float(end[1]), 1.0, places=3)

    def test_pose_factor_falls_with_yaw(self):
        G.profile_fade = True
        f = Face(bbox=np.zeros(4), kps=np.zeros((5, 2)))
        f.pose = np.array([0.0, 20.0, 0.0])
        self.assertEqual(face_fader.pose_factor(None, f), 1.0)
        f.pose = np.array([0.0, 47.5, 0.0])
        self.assertAlmostEqual(face_fader.pose_factor(None, f), 0.5, places=2)
        f.pose = np.array([0.0, -70.0, 0.0])
        self.assertEqual(face_fader.pose_factor(None, f), 0.0)


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = patch.object(face_library, "LIB_DIR", os.path.join(self.tmp, "lib"))
        self.p.start()

    def tearDown(self):
        self.p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_load_replace_delete(self):
        imgs = []
        for i in range(2):
            path = os.path.join(self.tmp, f"p{i}.jpg")
            cv2.imwrite(path, np.full((8, 8, 3), i * 50, np.uint8))
            imgs.append(path)
        name = face_library.save_set("Anna 1/", imgs)
        self.assertEqual(name, "Anna 1")
        self.assertEqual(len(face_library.photos(name)), 2)
        face_library.save_set(name, imgs[:1])            # replace
        self.assertEqual(len(face_library.photos(name)), 1)
        face_library.delete_set(name)
        self.assertEqual(face_library.list_sets(), [])

    def test_empty_set_rejected(self):
        with self.assertRaises(ValueError):
            face_library.save_set("x", [])


class UpscalerFaceBoxTest(unittest.TestCase):
    def test_boxes_clamped_and_square_ish(self):
        from modules.processors.frame import upscaler
        boxes = upscaler._face_boxes([np.array([10, 10, 110, 130])], 200, 150)
        x0, y0, x1, y1 = boxes[0]
        self.assertEqual((x0, y0), (0, 0))
        self.assertLessEqual(x1, 200)
        self.assertLessEqual(y1, 150)


if __name__ == "__main__":
    unittest.main()
