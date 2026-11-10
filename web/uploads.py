"""Decode uploads without creating temporary files, preserving alpha and camera orientation."""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 20_000_000


def decode_upload(raw, background="white"):
    try:
        with Image.open(io.BytesIO(raw)) as source:
            size, mode = source.size, source.mode
            if size[0] * size[1] > MAX_PIXELS:
                raise ValueError("Ảnh vượt quá 20 triệu điểm ảnh. Hãy giảm kích thước ảnh.")
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", image.size, background), image)
            return image.convert("RGB"), size, mode
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Không đọc được file ảnh. Hãy chọn ảnh JPG, PNG hoặc WEBP hợp lệ.") from error
