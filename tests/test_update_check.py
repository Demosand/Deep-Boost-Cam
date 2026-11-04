import unittest

from modules import update_check


class UpdateCompareTest(unittest.TestCase):
    def test_newer(self):
        self.assertTrue(update_check.is_newer("v1.0.1", "1.0.0"))
        self.assertTrue(update_check.is_newer("v2.0", "1.9.9"))

    def test_not_newer(self):
        self.assertFalse(update_check.is_newer("v1.0.0", "1.0.0"))
        self.assertFalse(update_check.is_newer("v0.9.9", "1.0.0"))
        self.assertFalse(update_check.is_newer("", "1.0.0"))


if __name__ == "__main__":
    unittest.main()
