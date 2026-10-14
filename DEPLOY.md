# Server setup

This repository and the server ZIP contain the complete runnable face-shape demo, including both model files. Python dependencies install during the Docker build. Git LFS is not needed.

## Docker (recommended)

On a server with Docker installed, clone the repository:

```bash
git clone https://github.com/amkabir/Ayodo_Oct.git
cd Ayodo_Oct
```

Or, if you received the ZIP instead:

```bash
unzip ayodo-face-type-server.zip
cd ayodo-face-type-server
```

Then build and run:

```bash
docker build -t ayodo-face-type:1.0 .
docker run -d --name ayodo-face-type --restart unless-stopped -p 7860:7860 ayodo-face-type:1.0
docker logs -f ayodo-face-type
```

Open `http://SERVER_IP:7860/` to test it. For a public deployment, put the app behind a domain with HTTPS; browser webcam capture generally requires HTTPS. Restrict access if this is a private review demo.

Run the packaged smoke tests in the same image:

```bash
docker run --rm ayodo-face-type:1.0 python -m unittest discover -s tests -v
```

To stop or update it:

```bash
docker stop ayodo-face-type
docker rm ayodo-face-type
```

## Python without Docker

Use Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 .venv/bin/python app.py
```

Keep the process running under the server's process manager. Expose port 7860 through a reverse proxy with HTTPS for public use.

## Scope

The app predicts a handcrafted face-shape label and displays an uncalibrated heuristic match score. It is a demonstration only. It does not verify identity or implement the mQR credential system described in the separate competition proposal. The packaged sample image checks software execution, not prediction accuracy.

Before public redistribution, confirm rights to the bundled `models/lbfmodel.yaml` landmark model. Its commonly cited upstream repository does not state a license.
