from __future__ import annotations

import traceback
from typing import Tuple

import cv2
import gradio as gr
import numpy as np

from face_type_pipeline import _get_pipeline, predict_face_type


APP_TITLE = "Ayodo Face Type"
APP_DESCRIPTION = (
    "Upload a face photo or use your camera, then press Submit to get the predicted face type. "
    "This tool classifies face shape only and does not identify a person."
)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Please upload an image or take a photo.")

    if not isinstance(image, np.ndarray):
        raise ValueError("Please upload a supported image file.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3:
        raise ValueError("Please upload a standard color image.")

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] != 3:
        raise ValueError("Please upload a standard RGB or RGBA image.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Please upload a non-empty image.")

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def classify_face(image: np.ndarray) -> Tuple[str, str]:
    if image is None:
        return "No result", "Please upload an image or take a photo."

    try:
        image_bgr = _to_bgr(image)
        result = predict_face_type(image_bgr)

        face_type = result.get("face_type", "Unknown")
        confidence = result.get("confidence")
        note = result.get("note", "Prediction completed.")

        face_type_text = f"Face Type: {face_type}"
        details = note
        if confidence is not None:
            details = f"Heuristic match score: {confidence:.2f} (not a probability)\n{note}"

        return face_type_text, details
    except ValueError as exc:
        return "No result", str(exc)
    except Exception:
        traceback.print_exc()
        return (
            "No result",
            "Something went wrong while processing the image. Please try a clear front-facing photo with one visible face.",
        )


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    image_input = gr.Image(
        type="numpy",
        sources=["upload", "webcam"],
        label="Face Image",
    )
    submit_button = gr.Button("Submit", variant="primary")

    face_type_output = gr.Textbox(label="Result", lines=1)
    details_output = gr.Textbox(label="Details", lines=3)

    submit_button.click(
        fn=classify_face,
        inputs=image_input,
        outputs=[face_type_output, details_output],
    )

    gr.Markdown("Use a clear, front-facing image with one visible face for the most stable result.")


if __name__ == "__main__":
    _get_pipeline()  # Fail at startup if a model file is missing or incompatible.
    demo.launch()
