# Test image provenance

`astronaut-face.png` is a crop of `skimage.data.astronaut()`, a photograph of NASA astronaut Eileen Collins. The scikit-image documentation identifies the source as the NASA Great Images database and says the image is in the public domain: https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut

The crop is used only to verify that face detection, 68-point landmark fitting, and the UI callback complete under the pinned dependencies. Its predicted label is not a ground-truth face-shape label or an accuracy benchmark.
