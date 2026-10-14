from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


FACE_TYPE_LABELS = (
    "Oval",
    "Round",
    "Square",
    "Rectangle",
    "Oblong",
    "Diamond",
    "Heart",
    "Inverted Triangle",
    "Triangle",
    "Pear",
    "Tapered Oval",
    "Tapered Square",
)

STABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "Oval": ("Oval", "Tapered Oval"),
    "Round": ("Round",),
    "Square": ("Square", "Tapered Square"),
    "Rectangle": ("Rectangle", "Oblong"),
    "Diamond": ("Diamond",),
    "Heart": ("Heart", "Inverted Triangle"),
    "Triangle": ("Triangle", "Pear"),
}

ROOT = Path(__file__).resolve().parent
YU_NET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
LBF_MODEL_PATH = ROOT / "models" / "lbfmodel.yaml"

MIN_FACE_SIZE_PX = 70
MIN_FACE_AREA_RATIO = 0.05
MIN_INTEROCULAR_PX = 22.0
MAX_ABS_ROLL_DEG = 18.0
MAX_ABS_YAW_RATIO = 0.38
MAX_ABS_PITCH_RATIO = 0.45
# Blur is measured on a face crop resized to BLUR_REFERENCE_PX so the threshold
# is independent of the original photo resolution. Calibrated so a clear photo
# scores well above the threshold while genuinely soft/out-of-focus faces fall
# below it.
BLUR_REFERENCE_PX = 200
BLUR_THRESHOLD = 25.0
DOMINANT_FACE_RATIO = 0.50
ALIGNED_SIZE = 256
# YuNet fails to detect faces in very large images (e.g. multi-megapixel phone
# photos), so detection runs on a copy whose longest side is capped here. Phone
# uploads are routinely 3000-4000px and otherwise return "no face detected".
MAX_DETECT_SIDE = 1024


@dataclass(frozen=True)
class Detection:
    bbox_xywh: np.ndarray
    landmarks5: np.ndarray
    score: float

    @property
    def area(self) -> float:
        return float(self.bbox_xywh[2] * self.bbox_xywh[3])


@dataclass(frozen=True)
class QualityMetrics:
    face_width_px: float
    face_height_px: float
    face_area_ratio: float
    interocular_px: float
    blur_score: float
    roll_deg: float
    yaw_ratio: float
    pitch_ratio: float
    quality_score: float


def _euclidean(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0).astype(np.float32)


def _arc_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.linalg.norm(direction))
    if denominator < 1e-6:
        return 0.0
    offset = point - start
    cross_magnitude = abs(direction[0] * offset[1] - direction[1] * offset[0])
    return float(cross_magnitude / denominator)


def _soft_closeness(value: float, target: float, tolerance: float) -> float:
    return float(np.exp(-0.5 * ((value - target) / max(tolerance, 1e-6)) ** 2))


def _bounded_score(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        distance = lower - value
    else:
        distance = value - upper
    spread = max((upper - lower) * 0.5, 1e-6)
    return float(np.exp(-0.5 * (distance / spread) ** 2))


def _blur_score(image_bgr: np.ndarray, bbox_xywh: np.ndarray) -> float:
    x, y, w, h = [int(round(value)) for value in bbox_xywh]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image_bgr.shape[1], x + w)
    y1 = min(image_bgr.shape[0], y + h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Normalize the crop to a fixed size before measuring sharpness. Laplacian
    # variance falls off as a crop grows, so without this a large, perfectly
    # sharp face reads as "blurry" purely because of its resolution.
    normalized = cv2.resize(gray, (BLUR_REFERENCE_PX, BLUR_REFERENCE_PX), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(normalized, cv2.CV_64F).var())


def _roll_deg(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    delta = right_eye - left_eye
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _yaw_ratio(landmarks68: np.ndarray) -> float:
    nose_tip = landmarks68[30]
    left_face = landmarks68[0]
    right_face = landmarks68[16]
    left_dist = abs(nose_tip[0] - left_face[0])
    right_dist = abs(right_face[0] - nose_tip[0])
    denominator = max(left_dist + right_dist, 1e-6)
    return float((left_dist - right_dist) / denominator)


def _pitch_ratio(landmarks68: np.ndarray) -> float:
    brow_center = _centroid(np.vstack([landmarks68[21], landmarks68[22]]))
    nose_tip = landmarks68[30]
    chin = landmarks68[8]
    upper = _euclidean(brow_center, nose_tip)
    lower = _euclidean(nose_tip, chin)
    denominator = max(upper + lower, 1e-6)
    return float((upper - lower) / denominator)


def _canonical_alignment_template(output_size: int) -> np.ndarray:
    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    return template * (output_size / 112.0)


def _extract_alignment_points(landmarks68: np.ndarray) -> np.ndarray:
    left_eye = _centroid(landmarks68[36:42])
    right_eye = _centroid(landmarks68[42:48])
    nose_tip = landmarks68[30]
    left_mouth = landmarks68[48]
    right_mouth = landmarks68[54]
    return np.stack([left_eye, right_eye, nose_tip, left_mouth, right_mouth]).astype(np.float32)


def _align_face(image_bgr: np.ndarray, landmarks68: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = _extract_alignment_points(landmarks68)
    destination = _canonical_alignment_template(ALIGNED_SIZE)
    transform, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("Face alignment failed. Please try a clearer front-facing image.")

    aligned_image = cv2.warpAffine(
        image_bgr,
        transform,
        (ALIGNED_SIZE, ALIGNED_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    homogeneous = np.hstack([landmarks68, np.ones((landmarks68.shape[0], 1), dtype=np.float32)])
    aligned_landmarks = (transform @ homogeneous.T).T.astype(np.float32)
    return aligned_image, aligned_landmarks


class FaceTypePipeline:
    def __init__(self) -> None:
        if not YU_NET_PATH.exists() or not LBF_MODEL_PATH.exists():
            raise FileNotFoundError("Required model files are missing from the models directory.")
        if YU_NET_PATH.stat().st_size < 100_000 or LBF_MODEL_PATH.stat().st_size < 1_000_000:
            raise RuntimeError("Model files appear incomplete. Run `git lfs pull` before starting the app.")

        self.detector = cv2.FaceDetectorYN.create(
            str(YU_NET_PATH),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.landmarker = cv2.face.createFacemarkLBF()
        self.landmarker.loadModel(str(LBF_MODEL_PATH))

    def predict(self, image_bgr: np.ndarray) -> dict[str, object]:
        image_bgr = self._validate_image(image_bgr)
        image_bgr, detections = self._detect_with_recovery(image_bgr)
        detection = self._select_primary_face(detections)
        landmarks68 = self._fit_landmarks(image_bgr, detection)
        quality = self._quality_checks(image_bgr, detection, landmarks68)
        aligned_image, aligned_landmarks = _align_face(image_bgr, landmarks68)
        geometry = self._geometry_features(aligned_landmarks)
        face_type, confidence, second_best = self._classify(geometry, quality)

        note = "Best match based on stable facial proportions and contour geometry."
        if confidence < 0.5:
            note = (
                "Best match based on stable facial proportions and contour geometry. "
                "Use a clearer straight-on photo for a stronger match."
            )

        return {
            "face_type": face_type,
            "confidence": confidence,
            "note": note,
            "second_best": second_best,
        }

    def _validate_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("Please upload a supported image or take a photo.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Please upload a standard color face image.")
        return image_bgr

    def _resize_for_detection(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        longest = max(height, width)
        if longest <= MAX_DETECT_SIDE:
            return image_bgr
        scale = MAX_DETECT_SIDE / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

    def _detect_with_recovery(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """Detect faces robustly: downscale oversized images and try the four
        orientations so EXIF-rotated phone photos still work. Returns the image
        in the orientation that detected a face, plus its detections."""
        working = self._resize_for_detection(image_bgr)
        rotations = (
            (None, working),
            (cv2.ROTATE_90_CLOCKWISE, cv2.rotate(working, cv2.ROTATE_90_CLOCKWISE)),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.rotate(working, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            (cv2.ROTATE_180, cv2.rotate(working, cv2.ROTATE_180)),
        )

        best_image = working
        best_detections: list[Detection] = []
        best_score = -1.0
        for _, candidate in rotations:
            detections = self._detect_faces(candidate)
            if not detections:
                continue
            top_score = max(detection.score for detection in detections)
            if top_score > best_score:
                best_score = top_score
                best_image = candidate
                best_detections = detections

        return best_image, best_detections

    def _detect_faces(self, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image_bgr)
        if faces is None:
            return []
        return [
            Detection(
                bbox_xywh=row[:4].astype(np.float32),
                landmarks5=row[4:14].reshape(5, 2).astype(np.float32),
                score=float(row[14]),
            )
            for row in faces
        ]

    def _select_primary_face(self, detections: list[Detection]) -> Detection:
        if not detections:
            raise ValueError("No face detected. Please use a clear image with one visible face.")

        ranked = sorted(detections, key=lambda item: item.area, reverse=True)
        if len(ranked) > 1 and ranked[1].area / max(ranked[0].area, 1e-6) >= DOMINANT_FACE_RATIO:
            raise ValueError("Multiple faces detected. Please use an image with one visible face.")
        return ranked[0]

    def _fit_landmarks(self, image_bgr: np.ndarray, detection: Detection) -> np.ndarray:
        rect = np.array(
            [[
                int(round(detection.bbox_xywh[0])),
                int(round(detection.bbox_xywh[1])),
                int(round(detection.bbox_xywh[2])),
                int(round(detection.bbox_xywh[3])),
            ]],
            dtype=np.int32,
        )
        ok, landmarks = self.landmarker.fit(image_bgr, rect)
        if not ok or landmarks is None:
            raise ValueError("Face landmarks could not be extracted. Please try a clearer image.")
        # OpenCV 4 returns (1, 68, 2), while OpenCV 5 returns (68, 1, 2).
        points = np.asarray(landmarks[0], dtype=np.float32)
        if points.size != 68 * 2:
            raise ValueError("Face landmarks could not be extracted. Please try a clearer image.")
        return points.reshape(68, 2)

    def _quality_checks(
        self,
        image_bgr: np.ndarray,
        detection: Detection,
        landmarks68: np.ndarray,
    ) -> QualityMetrics:
        left_eye = _centroid(landmarks68[36:42])
        right_eye = _centroid(landmarks68[42:48])
        brow_center = _centroid(np.vstack([landmarks68[21], landmarks68[22]]))
        chin = landmarks68[8]

        interocular_px = _euclidean(left_eye, right_eye)
        blur_score = _blur_score(image_bgr, detection.bbox_xywh)
        roll_deg = _roll_deg(left_eye, right_eye)
        yaw_ratio = _yaw_ratio(landmarks68)
        pitch_ratio = _pitch_ratio(landmarks68)
        face_width_px = float(detection.bbox_xywh[2])
        face_height_px = float(detection.bbox_xywh[3])
        face_area_ratio = float(detection.area / max(image_bgr.shape[0] * image_bgr.shape[1], 1))

        checks = [
            (
                face_width_px >= MIN_FACE_SIZE_PX and face_height_px >= MIN_FACE_SIZE_PX,
                "Face is too small. Please move closer to the camera.",
            ),
            (
                face_area_ratio >= MIN_FACE_AREA_RATIO,
                "Face occupies too little of the image. Please use a closer crop.",
            ),
            (
                interocular_px >= MIN_INTEROCULAR_PX,
                "Face is too small or too far away for a stable result.",
            ),
            (
                abs(roll_deg) <= MAX_ABS_ROLL_DEG,
                "Head tilt is too strong. Please use a straight-on photo.",
            ),
            (
                abs(yaw_ratio) <= MAX_ABS_YAW_RATIO,
                "Face angle is too extreme. Please use a front-facing photo.",
            ),
            (
                abs(pitch_ratio) <= MAX_ABS_PITCH_RATIO,
                "Face angle is too extreme. Please keep the camera level.",
            ),
            (
                blur_score >= BLUR_THRESHOLD,
                "Image is too blurry. Please use a sharper photo.",
            ),
        ]

        for passed, message in checks:
            if not passed:
                raise ValueError(message)

        upper_face_height = _euclidean(brow_center, landmarks68[33])
        lower_face_height = _euclidean(landmarks68[33], chin)
        lower_balance = lower_face_height / max(upper_face_height, 1e-6)

        quality_score = np.mean(
            [
                _bounded_score(blur_score, BLUR_THRESHOLD, BLUR_THRESHOLD * 8.0),
                _bounded_score(interocular_px, MIN_INTEROCULAR_PX, MIN_INTEROCULAR_PX * 7.0),
                _bounded_score(abs(roll_deg), 0.0, MAX_ABS_ROLL_DEG * 0.55),
                _bounded_score(abs(yaw_ratio), 0.0, MAX_ABS_YAW_RATIO * 0.65),
                _bounded_score(abs(pitch_ratio), 0.0, MAX_ABS_PITCH_RATIO * 0.65),
                _bounded_score(lower_balance, 0.85, 1.55),
            ]
        )

        return QualityMetrics(
            face_width_px=face_width_px,
            face_height_px=face_height_px,
            face_area_ratio=face_area_ratio,
            interocular_px=interocular_px,
            blur_score=blur_score,
            roll_deg=roll_deg,
            yaw_ratio=yaw_ratio,
            pitch_ratio=pitch_ratio,
            quality_score=float(np.clip(quality_score, 0.0, 1.0)),
        )

    def _geometry_features(self, landmarks68: np.ndarray) -> dict[str, float]:
        brow_center = _centroid(np.vstack([landmarks68[21], landmarks68[22]]))
        nose_base = landmarks68[33]
        chin = landmarks68[8]

        upper_width = _euclidean(landmarks68[17], landmarks68[26])
        cheekbone_width = _euclidean(landmarks68[3], landmarks68[13])
        jaw_width = _euclidean(landmarks68[4], landmarks68[12])
        chin_width = _euclidean(landmarks68[6], landmarks68[10])
        dominant_width = max(upper_width, cheekbone_width, jaw_width, 1e-6)
        upper_face_height = _euclidean(brow_center, nose_base)
        lower_face_height = _euclidean(nose_base, chin)
        face_length = _euclidean(brow_center, chin) + 0.5 * upper_face_height

        mid_x = 0.5 * (landmarks68[0, 0] + landmarks68[16, 0])
        mirrored_offsets = []
        symmetric_pairs = [
            (0, 16),
            (1, 15),
            (2, 14),
            (3, 13),
            (4, 12),
            (5, 11),
            (6, 10),
            (7, 9),
            (17, 26),
            (18, 25),
            (19, 24),
            (20, 23),
            (21, 22),
            (31, 35),
            (32, 34),
            (36, 45),
            (37, 44),
            (38, 43),
            (39, 42),
            (40, 47),
            (41, 46),
            (48, 54),
            (49, 53),
            (50, 52),
            (59, 55),
            (58, 56),
        ]
        for left_idx, right_idx in symmetric_pairs:
            left = landmarks68[left_idx]
            right = landmarks68[right_idx]
            mirrored_gap = abs((mid_x - left[0]) - (right[0] - mid_x))
            vertical_gap = abs(left[1] - right[1])
            mirrored_offsets.append(np.hypot(mirrored_gap, vertical_gap))

        jaw_arc_ratio = _arc_length(landmarks68[3:14]) / max(_euclidean(landmarks68[3], landmarks68[13]), 1e-6)
        chin_depth = _point_line_distance(chin, landmarks68[5], landmarks68[11]) / max(face_length, 1e-6)
        chin_taper = 1.0 - (chin_width / max(jaw_width, 1e-6))
        jaw_squareness = 0.5 * (jaw_width / max(cheekbone_width, 1e-6)) + 0.5 * (chin_width / max(jaw_width, 1e-6))
        landmark_asymmetry = float(np.mean(mirrored_offsets) / max(_euclidean(landmarks68[36], landmarks68[45]), 1e-6))

        return {
            "aspect_ratio": face_length / dominant_width,
            "forehead_to_jaw": upper_width / max(jaw_width, 1e-6),
            "cheek_to_jaw": cheekbone_width / max(jaw_width, 1e-6),
            "cheek_to_forehead": cheekbone_width / max(upper_width, 1e-6),
            "chin_taper": chin_taper,
            "chin_depth": chin_depth,
            "jaw_curve": jaw_arc_ratio,
            "jaw_squareness": jaw_squareness,
            "lower_balance": lower_face_height / max(upper_face_height, 1e-6),
            "symmetry": float(np.clip(1.0 - landmark_asymmetry, 0.0, 1.0)),
        }

    def _resolve_stable_label(
        self,
        geometry: dict[str, float],
        scores: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        family_scores: dict[str, float] = {}
        for family, members in STABLE_GROUPS.items():
            member_scores = [scores[member] for member in members]
            family_scores[family] = 0.7 * max(member_scores) + 0.3 * float(np.mean(member_scores))

        ranked_families = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        chosen_family = ranked_families[0][0]
        chosen_label = chosen_family

        if chosen_family == "Oval":
            tapered_margin = scores["Tapered Oval"] - scores["Oval"]
            if (
                geometry["aspect_ratio"] >= 1.50
                and geometry["chin_taper"] >= 0.49
                and tapered_margin >= 0.035
            ):
                chosen_label = "Tapered Oval"
        elif chosen_family == "Square":
            tapered_margin = scores["Tapered Square"] - scores["Square"]
            if (
                geometry["chin_taper"] >= 0.43
                and geometry["jaw_squareness"] >= 0.75
                and tapered_margin >= 0.035
            ):
                chosen_label = "Tapered Square"
        elif chosen_family == "Rectangle":
            oblong_margin = scores["Oblong"] - scores["Rectangle"]
            if geometry["aspect_ratio"] >= 1.58 and oblong_margin >= 0.03:
                chosen_label = "Oblong"
        elif chosen_family == "Heart":
            inverted_margin = scores["Inverted Triangle"] - scores["Heart"]
            if (
                geometry["forehead_to_jaw"] >= 1.20
                and geometry["cheek_to_forehead"] <= 0.97
                and inverted_margin >= 0.03
            ):
                chosen_label = "Inverted Triangle"
        elif chosen_family == "Triangle":
            pear_margin = scores["Pear"] - scores["Triangle"]
            if (
                geometry["lower_balance"] >= 1.14
                and geometry["jaw_curve"] >= 1.53
                and pear_margin >= 0.03
            ):
                chosen_label = "Pear"

        return chosen_label, family_scores

    def _classify(self, geometry: dict[str, float], quality: QualityMetrics) -> tuple[str, float, str]:
        prototypes = {
            "Oval": {
                "aspect_ratio": (1.40, 0.16, 1.35),
                "forehead_to_jaw": (1.08, 0.10, 1.0),
                "cheek_to_jaw": (1.10, 0.08, 1.15),
                "chin_taper": (0.44, 0.05, 1.0),
                "jaw_curve": (1.56, 0.10, 0.85),
                "jaw_squareness": (0.72, 0.07, 0.8),
            },
            "Round": {
                "aspect_ratio": (1.24, 0.14, 1.45),
                "forehead_to_jaw": (1.06, 0.10, 0.95),
                "cheek_to_jaw": (1.10, 0.08, 1.15),
                "chin_taper": (0.42, 0.05, 0.9),
                "jaw_curve": (1.60, 0.10, 1.45),
                "jaw_squareness": (0.71, 0.07, 0.8),
            },
            "Square": {
                "aspect_ratio": (1.23, 0.14, 1.35),
                "forehead_to_jaw": (1.00, 0.08, 1.05),
                "cheek_to_jaw": (1.04, 0.07, 1.1),
                "chin_taper": (0.35, 0.06, 0.95),
                "jaw_curve": (1.44, 0.08, 1.1),
                "jaw_squareness": (0.80, 0.06, 1.45),
            },
            "Rectangle": {
                "aspect_ratio": (1.48, 0.16, 1.45),
                "forehead_to_jaw": (1.00, 0.08, 1.0),
                "cheek_to_jaw": (1.04, 0.07, 1.0),
                "chin_taper": (0.35, 0.06, 0.95),
                "jaw_curve": (1.44, 0.08, 1.0),
                "jaw_squareness": (0.80, 0.06, 1.45),
            },
            "Oblong": {
                "aspect_ratio": (1.60, 0.18, 1.55),
                "forehead_to_jaw": (1.08, 0.10, 0.95),
                "cheek_to_jaw": (1.08, 0.08, 1.0),
                "chin_taper": (0.40, 0.05, 0.85),
                "jaw_curve": (1.52, 0.10, 0.95),
                "jaw_squareness": (0.74, 0.07, 0.85),
            },
            "Diamond": {
                "aspect_ratio": (1.40, 0.16, 1.15),
                "cheek_to_jaw": (1.18, 0.08, 1.35),
                "cheek_to_forehead": (1.14, 0.08, 1.3),
                "chin_taper": (0.49, 0.05, 1.25),
                "chin_depth": (0.18, 0.04, 0.9),
                "jaw_curve": (1.60, 0.10, 0.9),
            },
            "Heart": {
                "aspect_ratio": (1.36, 0.14, 1.2),
                "forehead_to_jaw": (1.16, 0.10, 1.35),
                "cheek_to_jaw": (1.12, 0.08, 1.1),
                "chin_taper": (0.48, 0.05, 1.2),
                "chin_depth": (0.18, 0.04, 1.0),
                "jaw_curve": (1.58, 0.10, 0.9),
            },
            "Inverted Triangle": {
                "aspect_ratio": (1.30, 0.14, 1.0),
                "forehead_to_jaw": (1.22, 0.10, 1.45),
                "cheek_to_jaw": (1.08, 0.08, 1.0),
                "cheek_to_forehead": (0.92, 0.07, 1.1),
                "chin_taper": (0.47, 0.05, 1.2),
                "chin_depth": (0.18, 0.04, 1.0),
            },
            "Triangle": {
                "aspect_ratio": (1.24, 0.14, 1.1),
                "forehead_to_jaw": (0.88, 0.08, 1.4),
                "cheek_to_jaw": (0.96, 0.07, 1.15),
                "chin_taper": (0.36, 0.06, 0.95),
                "jaw_curve": (1.44, 0.08, 1.0),
                "jaw_squareness": (0.80, 0.06, 1.25),
            },
            "Pear": {
                "aspect_ratio": (1.28, 0.14, 1.0),
                "forehead_to_jaw": (0.85, 0.08, 1.4),
                "cheek_to_jaw": (1.00, 0.07, 1.0),
                "chin_taper": (0.40, 0.05, 0.9),
                "jaw_curve": (1.56, 0.10, 1.1),
                "lower_balance": (1.18, 0.10, 1.15),
            },
            "Tapered Oval": {
                "aspect_ratio": (1.52, 0.16, 1.3),
                "forehead_to_jaw": (1.14, 0.10, 1.2),
                "cheek_to_jaw": (1.10, 0.08, 1.0),
                "chin_taper": (0.50, 0.05, 1.35),
                "chin_depth": (0.19, 0.04, 1.05),
                "jaw_squareness": (0.69, 0.07, 0.95),
            },
            "Tapered Square": {
                "aspect_ratio": (1.32, 0.14, 1.2),
                "forehead_to_jaw": (1.06, 0.08, 1.0),
                "cheek_to_jaw": (1.06, 0.07, 1.0),
                "chin_taper": (0.42, 0.05, 1.15),
                "jaw_curve": (1.48, 0.08, 1.0),
                "jaw_squareness": (0.78, 0.06, 1.3),
            },
        }

        scores: dict[str, float] = {}
        for label, prototype in prototypes.items():
            total_weight = 0.0
            weighted_sum = 0.0
            for feature_name, (target, tolerance, weight) in prototype.items():
                weighted_sum += weight * _soft_closeness(geometry[feature_name], target, tolerance)
                total_weight += weight

            if label in {"Round", "Square", "Rectangle", "Oblong"}:
                weighted_sum += 0.4 * geometry["symmetry"]
                total_weight += 0.4
            elif label in {"Heart", "Diamond", "Inverted Triangle", "Tapered Oval"}:
                weighted_sum += 0.25 * geometry["symmetry"]
                total_weight += 0.25

            scores[label] = weighted_sum / max(total_weight, 1e-6)

        top_label, family_scores = self._resolve_stable_label(geometry, scores)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_families = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = scores[top_label]
        second_label = ranked[0][0] if ranked[0][0] != top_label else ranked[1][0]
        second_score = scores[second_label]
        family_margin = float(np.clip(ranked_families[0][1] - ranked_families[1][1], 0.0, 1.0))

        logits = np.array([score * (5.0 + 2.0 * quality.quality_score) for _, score in ranked], dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top_probability = float(np.clip(scores[top_label], 0.0, 1.0))
        margin = float(np.clip(top_score - second_score, 0.0, 1.0))
        confidence = 0.40 * top_probability + 0.22 * top_score + 0.20 * quality.quality_score
        confidence += 0.10 * margin + 0.10 * family_margin
        confidence = float(np.clip(confidence, 0.35, 0.98))
        return top_label, confidence, second_label


@lru_cache(maxsize=1)
def _get_pipeline() -> FaceTypePipeline:
    return FaceTypePipeline()


def predict_face_type(image_bgr: np.ndarray) -> dict[str, object]:
    """
    Returns a dict:
    {
        "face_type": "Oval",
        "confidence": 0.91,
        "note": "Best match based on facial proportions and contour geometry."
    }
    """
    return _get_pipeline().predict(image_bgr)
