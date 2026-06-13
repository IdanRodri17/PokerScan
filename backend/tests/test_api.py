"""
API-hardening / smoke tests for the FastAPI backend (main.py + image_processor):
HEIC support, upload validation (size / content-type / unreadable), the CORS
config, and the /version endpoint. These drive the real app via TestClient.
"""

import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from main import app, MAX_UPLOAD_BYTES, allow_origins_setting

client = TestClient(app)


def _png_bytes(w=64, h=64):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 128, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_version_endpoint():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert {"version", "model_file", "model_loaded", "device"}.issubset(body)
    assert body["version"] == app.version


def test_sample_png_upload_succeeds():
    r = client.post("/upload", files={"file": ("hand.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_upload_rejects_too_large_with_413():
    big = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    r = client.post("/upload", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


def test_upload_rejects_unreadable_image_with_400():
    r = client.post("/upload", files={"file": ("bad.png", b"definitely not an image", "image/png")})
    assert r.status_code == 400


def test_upload_accepts_octet_stream_content_type():
    # Mobile/HEIC uploads often arrive as application/octet-stream; a real image must still work.
    r = client.post("/upload", files={"file": ("hand", _png_bytes(), "application/octet-stream")})
    assert r.status_code == 200


def test_heic_formats_registered():
    from main import image_processor
    from services.image_processor import HEIF_SUPPORTED
    formats = set(image_processor.get_supported_formats())
    if HEIF_SUPPORTED:
        assert {'.heic', '.heif'}.issubset(formats)


def test_heic_upload_round_trips_when_supported():
    from services.image_processor import HEIF_SUPPORTED
    if not HEIF_SUPPORTED:
        pytest.skip("pillow_heif not available")
    buf = io.BytesIO()
    try:
        Image.new("RGB", (64, 64), (10, 100, 30)).save(buf, format="HEIF")
    except Exception:
        pytest.skip("HEIF encoding not available in this build")
    r = client.post("/upload", files={"file": ("hand.heic", buf.getvalue(), "image/heic")})
    assert r.status_code == 200


def test_cors_uses_explicit_origins_not_wildcard():
    assert "*" not in allow_origins_setting
    assert "https://pokervision.netlify.app" in allow_origins_setting
