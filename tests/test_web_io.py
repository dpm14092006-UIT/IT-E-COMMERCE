import io
import json

import pytest
from PIL import Image

from web.storage import read_registry, write_registry
from web.uploads import decode_upload


def image_bytes(image, format="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=format, **kwargs)
    return stream.getvalue()


def test_registry_roundtrip_preserves_vietnamese_and_next_id(tmp_path):
    path = tmp_path / "nested" / "styles.json"
    assert read_registry(path) == {"styles": []}
    value = {"next_id": 9, "styles": [{"id": "N8", "name": "Áo khoác xanh"}]}
    write_registry(path, value)
    assert read_registry(path) == value
    assert list(path.parent.iterdir()) == [path]


def test_failed_registry_replace_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "styles.json"
    write_registry(path, {"styles": [{"id": "N1"}]})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("simulated disk error")

    monkeypatch.setattr("web.storage.os.replace", fail)
    with pytest.raises(OSError):
        write_registry(path, {"styles": []})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("content", ["not json", '[]', '{"styles": {}}'])
def test_invalid_registry_is_not_silently_replaced(tmp_path, content):
    path = tmp_path / "styles.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        read_registry(path)
    assert path.read_text(encoding="utf-8") == content


def test_non_finite_registry_is_not_written(tmp_path):
    path = tmp_path / "styles.json"
    with pytest.raises(ValueError):
        write_registry(path, {"styles": [{"proto": float("nan")}]})
    assert not path.exists()


def test_upload_composites_transparency_onto_training_background():
    source = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    source.putpixel((1, 0), (200, 20, 50, 255))
    rgb, size, mode = decode_upload(image_bytes(source))
    assert (size, mode, rgb.mode) == ((2, 1), "RGBA", "RGB")
    assert rgb.getpixel((0, 0)) == (255, 255, 255)
    assert rgb.getpixel((1, 0)) == (200, 20, 50)


def test_upload_applies_camera_orientation():
    source = Image.new("RGB", (4, 2), "red")
    exif = source.getexif()
    exif[274] = 6
    rgb, original_size, _ = decode_upload(image_bytes(source, "JPEG", exif=exif))
    assert original_size == (4, 2)
    assert rgb.size == (2, 4)


@pytest.mark.parametrize("raw", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_upload_is_rejected(raw):
    with pytest.raises(ValueError, match="file ảnh"):
        decode_upload(raw)


def test_upload_pixel_limit_checked_before_full_decode(monkeypatch):
    monkeypatch.setattr("web.uploads.MAX_PIXELS", 16)
    with pytest.raises(ValueError, match="điểm ảnh"):
        decode_upload(image_bytes(Image.new("RGB", (5, 5))))
