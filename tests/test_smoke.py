"""End-to-end checks using a public-domain NASA test photograph."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from app import _to_bgr, classify_face
from face_type_pipeline import FACE_TYPE_LABELS


FIXTURE = Path(__file__).parent / "fixtures" / "astronaut-face.png"


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bgr = cv2.imread(str(FIXTURE))
        assert bgr is not None, f"Missing fixture: {FIXTURE}"
        cls.face_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def test_valid_face_through_ui_callback(self):
        label, details = classify_face(self.face_rgb)
        self.assertIn(label.removeprefix("Face Type: "), FACE_TYPE_LABELS)
        self.assertIn("Heuristic match score:", details)

    def test_rotated_and_large_phone_photo(self):
        for image in (
            cv2.rotate(self.face_rgb, cv2.ROTATE_90_CLOCKWISE),
            cv2.resize(self.face_rgb, (2200, 2400)),
        ):
            with self.subTest(shape=image.shape):
                label, _ = classify_face(image)
                self.assertIn(label.removeprefix("Face Type: "), FACE_TYPE_LABELS)

    def test_blank_image_is_rejected(self):
        label, details = classify_face(np.zeros((400, 400, 3), dtype=np.uint8))
        self.assertEqual(label, "No result")
        self.assertIn("No face detected", details)

    def test_invalid_inputs_are_actionable(self):
        self.assertIn("upload", classify_face(None)[1])
        with self.assertRaisesRegex(ValueError, "RGB or RGBA"):
            _to_bgr(np.zeros((10, 10, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _to_bgr(np.zeros((0, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
