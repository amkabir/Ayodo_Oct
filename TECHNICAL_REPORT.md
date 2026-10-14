# Ayodo Face Type Technical Report

## Current implementation note

The runnable app is pinned to OpenCV 5.0.0.93. A regression test checks the OpenCV 5 landmark-array shape, which differs from OpenCV 4. The displayed score is labeled a **heuristic match score**, not a probability or validated accuracy estimate. This report describes the existing face-shape demo; it does not document the proposed mQR credential system as implemented.

## 1. Executive Summary

`Ayodo Face Type` is a rule-based facial geometry classification system that predicts a face-shape label from a single image. It is not a face recognition system, not an identity verification model, and not a metric-learning or embedding pipeline. Its purpose is to accept one visible face, verify that the photo is usable, extract stable geometric measurements from facial landmarks, and map those measurements to one of 12 predefined face-type labels.

The system is intentionally designed around interpretable geometry rather than opaque learned embeddings. That design choice makes the pipeline easier to explain, easier to debug, and more controllable when the goal is face-shape categorization rather than person identification.

The 12 output labels are:

- `Oval`
- `Round`
- `Square`
- `Rectangle`
- `Oblong`
- `Diamond`
- `Heart`
- `Inverted Triangle`
- `Triangle`
- `Pear`
- `Tapered Oval`
- `Tapered Square`

In short, the system thinks in terms of:

- Is there exactly one usable face?
- Is the image good enough for geometry to be trustworthy?
- What are the stable proportions of the forehead, cheekbones, jaw, chin, and face length?
- Which predefined face-shape prototype is the closest match?
- Is the result stable enough to present with confidence?


## 2. What the Image Test Is

In this project, the "image test" is the full screening and classification procedure applied to an input image before a face-type label is returned.

It includes five major stages:

1. Image validation
2. Face detection and single-face selection
3. Landmark fitting and pose/quality gating
4. Face alignment and geometry extraction
5. Prototype scoring, stable-label resolution, and confidence estimation

This means the system does not simply "run a model and return a class." Instead, it performs a sequence of increasingly strict checks to make sure the image is appropriate for geometric classification.

The core principle is that a bad image should be rejected early rather than producing a misleading label.


## 3. Design Philosophy and Thinking Behind the Pipeline

The thinking behind the pipeline is conservative and geometry-first.

### 3.1 Why geometry instead of identity embeddings

Face-shape classification is fundamentally different from face recognition.

- Face recognition asks: "Whose face is this?"
- Face-type classification asks: "What structural facial shape does this face resemble?"

Recognition systems often use deep embeddings that compress identity-related information into dense vectors. That is useful for matching people, but it is not ideal if the product goal is to explain face shape in interpretable terms.

This pipeline instead uses measurable facial structure:

- forehead width
- cheekbone width
- jaw width
- chin width
- face length
- chin taper
- jaw curvature
- symmetry
- upper/lower face balance

Those features map directly to how people already describe face shapes, which makes the system more transparent.

### 3.2 Why reject images aggressively

Geometric classification is sensitive to image conditions. If the face is too small, too blurry, too tilted, or too far from frontal pose, the landmark geometry becomes unreliable.

So the pipeline is built with an explicit quality philosophy:

- accept only images that are usable for stable measurement
- reject ambiguous or weak samples before classification
- favor reliability over coverage

That is why the code contains hard thresholds for size, blur, yaw, pitch, roll, and multi-face ambiguity.

### 3.3 Why use stable families first, then fine subtypes

Several face-shape labels are close relatives. For example:

- `Oval` and `Tapered Oval`
- `Square` and `Tapered Square`
- `Rectangle` and `Oblong`
- `Heart` and `Inverted Triangle`
- `Triangle` and `Pear`

The pipeline first resolves a broader stable family, then applies extra subtype rules. This reduces label flicker and prevents tiny geometric noise from causing unstable label changes.


## 4. End-to-End Pipeline

## 4.1 Input handling

The Gradio app accepts one image from:

- file upload
- webcam capture

The app converts the image into OpenCV BGR format:

- grayscale images are expanded to BGR
- RGBA images are converted to RGB, then BGR
- invalid or non-standard arrays are rejected

At this point, the system has not yet decided whether the image is acceptable. It has only standardized the input format.


## 4.2 Face detection

The first true inference stage is face detection using OpenCV's `FaceDetectorYN` with the YuNet model:

- model path: `models/face_detection_yunet_2023mar.onnx`
- detector score threshold: `0.6`
- NMS threshold: `0.3`
- `top_k`: `5000`

The detector returns:

- a face bounding box
- 5 coarse landmarks
- a detection score

If no face is detected, the pipeline rejects the image.


## 4.3 Primary-face selection

The app is deliberately single-face only.

Detected faces are ranked by bounding-box area. The largest face is considered the candidate primary face. However, if the second-largest face is too similar in size to the largest one, the image is rejected as ambiguous.

The rule is:

- `DOMINANT_FACE_RATIO = 0.50`

Interpretation:

- if `second_face_area / largest_face_area >= 0.50`, the image is rejected
- this means the primary face must clearly dominate the scene

This is important because a multi-face scene creates ambiguity about which face should be classified.


## 4.4 Landmark fitting

After selecting the primary face, the pipeline fits 68 facial landmarks using OpenCV Facemark LBF:

- model path: `models/lbfmodel.yaml`

These 68 points define facial contour, brows, eyes, nose, and mouth. They are the basis for both:

- quality measurements
- face-shape geometry features

If landmarks cannot be extracted, the image is rejected.


## 4.5 Quality gating

This stage is central to the "image test." The system computes several quality and pose measurements and enforces hard acceptance thresholds.

The measurements are:

- face width in pixels
- face height in pixels
- face area ratio relative to the full image
- interocular distance
- blur score
- roll angle
- yaw ratio
- pitch ratio

The thresholds in the current implementation are:

- `MIN_FACE_SIZE_PX = 70`
- `MIN_FACE_AREA_RATIO = 0.05`
- `MIN_INTEROCULAR_PX = 22.0`
- `MAX_ABS_ROLL_DEG = 18.0`
- `MAX_ABS_YAW_RATIO = 0.38`
- `MAX_ABS_PITCH_RATIO = 0.45`
- `BLUR_THRESHOLD = 25.0`

Acceptance logic:

- face width and height must both be at least `70 px`
- face must occupy at least `5%` of the image area
- distance between eye centers must be at least `22 px`
- absolute head roll must be at most `18 degrees`
- absolute yaw ratio must be at most `0.38`
- absolute pitch ratio must be at most `0.45`
- blur score must be at least `25`

If any one of these checks fails, the image is rejected immediately.

This means acceptance is conjunction-based:

- the image is accepted only if all quality gates pass

### 4.5.1 What each threshold means

#### Face size threshold

The `70 px` minimum width/height prevents classification from running on tiny faces that cannot support stable landmark geometry.

#### Face area threshold

The `0.05` area ratio means the face must occupy at least 5% of the frame. This avoids classifying faces that are technically detectable but too small relative to the overall image.

#### Interocular threshold

The `22 px` interocular minimum is a practical geometric stability check. Eye-center distance is a strong proxy for how much facial detail is actually available.

#### Roll threshold

Roll is estimated from eye tilt. If the head is rotated too far clockwise or counterclockwise, the measured geometry can be distorted relative to the canonical shape assumptions.

#### Yaw threshold

Yaw measures left-right turning of the face using nose position relative to the left and right face edges. Large yaw means the face is not sufficiently frontal.

#### Pitch threshold

Pitch measures up-down tilt using brow, nose, and chin geometry. Large pitch means the camera angle is too high or too low for stable shape extraction.

#### Blur threshold

Blur is computed from the Laplacian variance of the detected face crop. Low variance means the crop lacks edge detail and is likely blurry. This is important because poor sharpness degrades both landmark stability and contour geometry.


## 4.6 Quality score

In addition to hard rejection thresholds, the pipeline computes a softer `quality_score` in the range `[0, 1]`.

This score is the mean of bounded scores for:

- blur
- interocular distance
- roll
- yaw
- pitch
- lower-face balance

Important distinction:

- hard thresholds determine acceptance or rejection
- `quality_score` affects confidence, not acceptance, once an image has already passed the hard gates

So the system has two levels of quality reasoning:

1. binary gate: "Is this image acceptable at all?"
2. continuous score: "How strong and stable is this accepted sample?"


## 4.7 Face alignment

Once the sample passes quality checks, the face is aligned to a canonical coordinate system.

The alignment uses five derived anchor points:

- left eye center
- right eye center
- nose tip
- left mouth corner
- right mouth corner

These points are mapped to a standard template, and an affine transform is estimated with `cv2.estimateAffinePartial2D` using `LMEDS`.

The aligned output size is:

- `ALIGNED_SIZE = 256`

Why alignment matters:

- it reduces nuisance variation from translation, scale, and mild rotation
- it makes facial measurements more comparable across images
- it lets geometry features reflect intrinsic structure rather than camera framing


## 4.8 Geometry feature extraction

After alignment, the pipeline computes interpretable geometric features from the 68 aligned landmarks.

The features are:

- `aspect_ratio`
- `forehead_to_jaw`
- `cheek_to_jaw`
- `cheek_to_forehead`
- `chin_taper`
- `chin_depth`
- `jaw_curve`
- `jaw_squareness`
- `lower_balance`
- `symmetry`

### 4.8.1 Meaning of the main features

#### `aspect_ratio`

Approximate face length divided by dominant width. This separates shorter/wider shapes from longer/narrower shapes.

#### `forehead_to_jaw`

Upper-face width divided by jaw width. This helps distinguish top-heavy faces from bottom-heavy ones.

#### `cheek_to_jaw`

Cheekbone width divided by jaw width. This helps identify whether the face is widest around the cheeks or the jaw.

#### `cheek_to_forehead`

Cheekbone width divided by forehead width. This is especially useful in distinguishing `Diamond`, `Heart`, and `Inverted Triangle` style shapes.

#### `chin_taper`

Defined as `1 - (chin_width / jaw_width)`. Higher values indicate a more tapered chin.

#### `chin_depth`

Distance from the chin point to a line across the lower jaw, normalized by face length. This captures how pointed or projecting the chin is.

#### `jaw_curve`

Arc length of the jaw contour relative to chord length. Higher values mean a rounder or more curved jawline.

#### `jaw_squareness`

A combined measure of jaw width and chin width relative to cheekbone width. Higher values correspond to squarer lower-face structure.

#### `lower_balance`

Lower-face height relative to upper-face height. This helps detect shapes with stronger lower-face dominance, such as `Pear`.

#### `symmetry`

Estimated by comparing mirrored landmark pairs around the midline. It is clipped to `[0, 1]` and acts as a stabilizing factor for some classes.


## 4.9 Prototype-based classification

The classifier is not trained in this repository. Instead, it uses manually defined class prototypes.

Each face-type label has a set of expected feature targets with:

- a target value
- a tolerance
- a feature weight

For each feature, the system computes a Gaussian-like closeness score:

- exact or near-exact matches score highest
- farther values score progressively lower

For a class, the weighted closeness values are averaged into a class score.

This means each class behaves like a handcrafted prototype in feature space.

Example interpretation:

- `Square` expects lower aspect ratio, lower chin taper, flatter jaw curve, and high jaw squareness
- `Oval` expects moderate length, moderate taper, smooth jaw curve, and balanced proportions
- `Diamond` expects wider cheekbones relative to both forehead and jaw, plus a tapered and somewhat pointed chin


## 4.10 Stable-family resolution

Raw class scores are not used directly as the final label in every case.

The code groups related labels into stable families:

- `Oval`: `Oval`, `Tapered Oval`
- `Round`: `Round`
- `Square`: `Square`, `Tapered Square`
- `Rectangle`: `Rectangle`, `Oblong`
- `Diamond`: `Diamond`
- `Heart`: `Heart`, `Inverted Triangle`
- `Triangle`: `Triangle`, `Pear`

For each family, a family score is computed as:

- `0.7 * max(member_scores) + 0.3 * mean(member_scores)`

This gives priority to the best matching subtype while still rewarding family-level consistency.

The highest-scoring family is chosen first. Then the pipeline may specialize to a subtype if additional geometric conditions are met.

### 4.10.1 Subtype-switch conditions

These are the explicit subtype refinement rules in the code.

#### `Oval` to `Tapered Oval`

Requires:

- `aspect_ratio >= 1.50`
- `chin_taper >= 0.49`
- `score(Tapered Oval) - score(Oval) >= 0.035`

#### `Square` to `Tapered Square`

Requires:

- `chin_taper >= 0.43`
- `jaw_squareness >= 0.75`
- `score(Tapered Square) - score(Square) >= 0.035`

#### `Rectangle` to `Oblong`

Requires:

- `aspect_ratio >= 1.58`
- `score(Oblong) - score(Rectangle) >= 0.03`

#### `Heart` to `Inverted Triangle`

Requires:

- `forehead_to_jaw >= 1.20`
- `cheek_to_forehead <= 0.97`
- `score(Inverted Triangle) - score(Heart) >= 0.03`

#### `Triangle` to `Pear`

Requires:

- `lower_balance >= 1.14`
- `jaw_curve >= 1.53`
- `score(Pear) - score(Triangle) >= 0.03`

This is one of the most important parts of the system's "thinking." It does not overreact to small differences. It first decides the broad morphological family, then upgrades to a finer subtype only when there is enough structural evidence.


## 4.11 Confidence computation

The pipeline returns a confidence value, but it is important to understand what this number means.

It is not a calibrated probability from a trained classifier.

It is a heuristic confidence score derived from:

- the strength of the winning class score
- the quality of the image
- the margin over the second-best class
- the margin over the second-best family

The implemented formula is:

- base from clipped winning score
- plus contribution from winning prototype score
- plus contribution from `quality_score`
- plus contribution from class margin
- plus contribution from family margin
- clipped to `[0.35, 0.98]`

More specifically:

- `0.40 * top_probability`
- `0.22 * top_score`
- `0.20 * quality_score`
- `0.10 * class_margin`
- `0.10 * family_margin`

Final clipping:

- minimum confidence shown: `0.35`
- maximum confidence shown: `0.98`

### 4.11.1 How to interpret confidence operationally

This score should be treated as a strength-of-match indicator, not as a statistically calibrated posterior probability.

Practical interpretation:

- high confidence means the sample passed all gates and strongly matches one prototype family
- medium confidence means the sample passed quality gates but sits closer to neighboring shapes
- lower confidence means the system still found a best match, but the image or geometry is less decisive

The UI adds a cautionary note when:

- `confidence < 0.5`

That note tells the user to use a clearer straight-on photo for a stronger match.


## 5. Acceptance and Rejection Logic

Acceptance is binary.

An image is accepted only if all of the following are true:

- image format is valid
- at least one face is detected
- exactly one dominant face is present
- landmarks are successfully fitted
- face size threshold passes
- face area threshold passes
- interocular threshold passes
- roll threshold passes
- yaw threshold passes
- pitch threshold passes
- blur threshold passes
- alignment succeeds

Otherwise, the system rejects the image with a user-facing error message.

The rejection philosophy is "fail closed." If the geometry is likely to be unstable, the pipeline prefers to ask for a better image rather than guess.


## 6. What the Thresholds Are Really Doing

The thresholds serve three different roles.

### 6.1 Reliability thresholds

These ensure there is enough signal to measure:

- `MIN_FACE_SIZE_PX`
- `MIN_FACE_AREA_RATIO`
- `MIN_INTEROCULAR_PX`
- `BLUR_THRESHOLD`

### 6.2 Pose-control thresholds

These ensure the face is close enough to frontal:

- `MAX_ABS_ROLL_DEG`
- `MAX_ABS_YAW_RATIO`
- `MAX_ABS_PITCH_RATIO`

### 6.3 Ambiguity-control thresholds

These ensure the sample is not semantically ambiguous:

- `DOMINANT_FACE_RATIO`
- subtype score margins such as `0.03` and `0.035`

Taken together, the thresholds are not arbitrary filters. They encode a clear stance:

- geometry-based classification should only happen when the pose, scale, and sample clarity support trustworthy landmark proportions


## 7. How the System "Thinks"

If we describe the runtime reasoning informally, it is approximately this:

1. "Do I have one clear face to work with?"
2. "Is the face large enough, sharp enough, and frontal enough?"
3. "Can I trust the landmark structure?"
4. "What are the ratios of top width, cheek width, jaw width, chin taper, face length, and jaw contour?"
5. "Which face-shape prototype best matches those ratios?"
6. "Is the best match only barely ahead, or clearly ahead?"
7. "Should I stay with the broader family label, or refine to a subtype?"

That is the conceptual thinking behind the pipeline.

This is not learned latent reasoning. It is engineered morphological reasoning.


## 8. What UMAP Is

UMAP stands for `Uniform Manifold Approximation and Projection`.

It is a nonlinear dimensionality-reduction technique commonly used to project high-dimensional data into 2D or 3D while preserving local neighborhood structure as much as possible.

In practical machine learning workflows, UMAP is often used for:

- visualizing embeddings
- exploring cluster structure
- diagnosing whether classes separate cleanly
- inspecting overlap between categories
- spotting outliers or mislabeled samples

If you have many geometric feature vectors or neural embeddings, UMAP can help answer questions like:

- Do `Oval` and `Tapered Oval` form nearby but separable clusters?
- Are `Rectangle` and `Oblong` strongly overlapping?
- Are some rejected samples sitting between classes?
- Are certain thresholds excluding points that lie in noisy transition zones?


## 9. What UMAP Is Not Doing in This Project

There is no UMAP code in the current implementation.

That means:

- UMAP is not part of inference
- UMAP is not part of the acceptance pipeline
- UMAP is not used to compute confidence
- UMAP is not used to choose thresholds
- UMAP is not used to assign classes at runtime

So in this repository, UMAP should be described as an analytical or research tool that could support development, not as a production inference component.


## 10. How UMAP Would Fit Into the Broader Technical Story

Even though UMAP is not implemented here, it fits naturally into the methodology around this type of system.

### 10.1 UMAP as a research and validation tool

If the team collects a development dataset with:

- aligned landmark-derived feature vectors
- human-reviewed face-type labels
- quality metadata

then UMAP can be applied to those feature vectors to visualize the morphology space.

That can help with:

- checking whether handcrafted prototypes are well placed
- understanding which classes are naturally close
- validating whether stable families make sense
- tuning subtype margins
- identifying regions where confidence should be reduced

### 10.2 UMAP as a threshold-tuning aid

UMAP cannot directly tell you the correct threshold values, but it can help you inspect the geometry of the feature space after thresholds are applied.

For example, you could compare:

- all detected samples
- samples that pass hard quality gates
- samples with high final confidence
- samples that flip between related subtype labels

If UMAP shows that rejected samples cluster in noisy overlap zones, that supports the rejection policy. If it shows many high-quality accepted samples are still intermingled across certain labels, that suggests the prototypes or taxonomy may need refinement.

### 10.3 UMAP as a communication tool

UMAP is also useful in reports because it translates high-dimensional structure into a visual narrative:

- compact clusters imply stable prototypes
- elongated overlap implies ambiguous boundaries
- isolated points imply outliers or poor-quality inputs

For stakeholders, UMAP can make the system's behavior easier to explain than raw feature tables alone.


## 11. Suggested Technical Positioning of This System

The most accurate technical description of the current project is:

"A deterministic face-shape classification pipeline based on face detection, landmark fitting, image-quality gating, canonical alignment, interpretable geometric feature extraction, and prototype-based scoring across a 12-label taxonomy."

It should not be described as:

- a face recognition engine
- a deep identity embedding system
- a UMAP-based classifier
- a trained end-to-end neural face-shape model


## 12. Strengths of the Current Approach

- Highly interpretable
- Easy to explain feature-by-feature
- Strong quality control
- Conservative acceptance policy
- Stable family-first decision logic
- Lightweight runtime dependencies
- Suitable for demos and controlled classification workflows


## 13. Limitations of the Current Approach

- Prototype parameters are handcrafted rather than learned from labeled data
- Confidence is heuristic rather than statistically calibrated
- Performance depends heavily on landmark quality
- Strong pose restrictions reduce coverage on unconstrained images
- Face-shape categories are inherently fuzzy and partly subjective
- UMAP-backed empirical validation is not currently present in code


## 14. Bottom Line

The image test in this project is a gated, geometry-driven classification pipeline. It works by accepting only one clear frontal face, extracting stable landmarks, aligning the face, computing interpretable facial proportions, and comparing those proportions to handcrafted prototypes for 12 face-shape labels. Thresholds are used primarily to reject images that are too small, blurry, off-angle, or ambiguous. Acceptance is binary, while confidence is a softer heuristic score describing how strong the accepted match is.

UMAP is not part of the runtime system in the current repository. However, it is highly relevant as a technical analysis tool for studying the geometry feature space, validating label families, visualizing class overlap, tuning thresholds, and supporting a more rigorous research narrative around the pipeline.
