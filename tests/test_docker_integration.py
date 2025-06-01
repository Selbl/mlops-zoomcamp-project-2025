"""
Integration test: build the Docker image and make sure the /predict
endpoint answers with the expected JSON structure.

Run with:  pytest -m docker
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from textwrap import dedent

import docker  # pip install docker
import pytest
import requests


# ───────────────────────── Helpers ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
IMAGE_TAG = "gradeclass-service:test"  # local, disposable tag


def free_host_port() -> int:
    """Return an unused local TCP port (race-safe enough for tests)."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ────────────────────────── Pytest fixtures ───────────────────────────────────
@pytest.fixture(scope="session")
def docker_client() -> docker.DockerClient:  # type: ignore
    return docker.from_env()


@pytest.fixture(scope="session")
def built_image(docker_client) -> str:
    """Build the image once for the whole session."""
    print("\n[build]   docker build -t", IMAGE_TAG, PROJECT_ROOT)
    # stream build output so CI logs show progress
    for line in docker_client.api.build(
        path=str(PROJECT_ROOT),
        tag=IMAGE_TAG,
        rm=True,
        decode=True,
    ):
        if "stream" in line:  # prettify output
            print(line["stream"].rstrip())
        elif "error" in line:
            pytest.fail("Docker build failed:\n" + json.dumps(line, indent=2))
    return IMAGE_TAG


@pytest.fixture(scope="module")
def running_container(docker_client, built_image):
    """Start the container on a random host port, yield the public URL."""
    host_port = free_host_port()
    print(f"[run]     docker run -p {host_port}:9696 {built_image}")

    container = docker_client.containers.run(
        built_image,
        detach=True,
        auto_remove=True,
        ports={"9696/tcp": host_port},  # map internal → external
        environment={  # minimal env if your image needs it
            "PREFECT_API_MODE": "ephemeral",
        },
    )

    # Give the Flask+Gunicorn app a few seconds to start
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{host_port}/health")
            if r.ok:
                break
        except requests.ConnectionError:
            time.sleep(0.5)
    else:  # pragma: no cover
        logs = container.logs().decode()[-800:]  # last ~800 chars for context
        container.stop()
        pytest.fail("Container never became healthy.\nLogs tail:\n" + logs)

    yield f"http://127.0.0.1:{host_port}"

    # ─── teardown ─────────────────────────────────────────────────────────
    container.stop(timeout=2)


# ──────────────────────────── The actual test ─────────────────────────────────
@pytest.mark.docker
def test_predict_endpoint(running_container):
    payload = {
        "Age": 16,
        "Gender": 1,
        "Ethnicity": 2,
        "ParentalEducation": 3,
        "Tutoring": 0,
        "ParentalSupport": 2,
        "Extracurricular": 1,
        "Sports": 0,
        "Music": 1,
        "Volunteering": 0,
        "StudentID": 123,
    }

    url = f"{running_container}/predict"
    resp = requests.post(url, json=payload, timeout=10)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "grade_class" in data, dedent(
        f"""
        Unexpected response JSON:

            {json.dumps(data, indent=2)}
        """
    )
