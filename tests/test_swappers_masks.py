import unittest

import numpy as np

import modules.globals as G
from modules.processors.frame import face_regions, swapper_models


class SwapperRegistryTest(unittest.TestCase):
    def test_default_and_fallback(self):
        self.assertEqual(swapper_models.spec_for("nope").key, swapper_models.DEFAULT_KEY)
        self.assertEqual(swapper_models.spec_for("hyperswap_1b_256").size, 256)

    def test_all_requested_models_registered(self):
        keys = {s.key for s in swapper_models.SWAPPERS}
        self.assertTrue({"inswapper_128", "reswapper_256", "hyperswap_1a_256",
                         "hyperswap_1b_256", "hyperswap_1c_256"} <= keys)


class MaskCacheTest(unittest.TestCase):
    def setUp(self):
        self.saved = (G.mask_interval, G.many_faces)
        face_regions.reset_cache()

    def tearDown(self):
        G.mask_interval, G.many_faces = self.saved
        face_regions.reset_cache()

    def _sequence(self, n):
        hits = []
        for _ in range(n):
            hit = face_regions._cached("parts", 64)
            hits.append(hit is not None)
            if hit is None:
                face_regions._store("parts", np.ones((64, 64), np.float32))
        return hits

    def test_interval_reuses_between_updates(self):
        G.mask_interval, G.many_faces = 3, False
        self.assertEqual(self._sequence(6), [False, True, True, False, True, True])

    def test_interval_one_never_reuses(self):
        G.mask_interval, G.many_faces = 1, False
        self.assertEqual(self._sequence(3), [False, False, False])

    def test_many_faces_never_reuses(self):
        G.mask_interval, G.many_faces = 3, True
        self.assertEqual(self._sequence(3), [False, False, False])


if __name__ == "__main__":
    unittest.main()
