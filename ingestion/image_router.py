from io import BytesIO
from PIL import Image

from ingestion.ocr_extractor import extract_text_from_image, has_meaningful_text
from ingestion.vlm_describer import describe_image


def _image_looks_visual(img: Image.Image) -> bool:
    """
    Heuristic: does this image likely contain non-text visual content
    (charts, diagrams, photos)? Used to decide if VLM is worth calling.
    """
    # Simple heuristic: if the image is wider than 400px AND has
    # significant color variance, it probably has visuals worth describing.
    if img.width < 400 or img.height < 300:
        return False

    # Convert to RGB, sample a few pixels, measure color spread
    rgb = img.convert("RGB")
    small = rgb.resize((50, 50))  # downsample for speed
    pixels = list(small.getdata())

    # Count distinct-ish colors (rough bucket: //32)
    buckets = set((r // 32, g // 32, b // 32) for r, g, b in pixels)

    # If many distinct color buckets → colored diagram/photo
    # If few (mostly black/white) → probably pure text
    return len(buckets) >= 6


def process_image(image_input) -> dict:
    """
    Route an image through OCR / VLM / both based on content.

    Returns:
        {
            "ocr_text": str,
            "vlm_description": str,
            "combined_text": str,
            "route": str   # "ocr" | "vlm" | "ocr+vlm" | "none"
        }
    """
    # Load image for heuristics
    if isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))
    else:
        img = Image.open(image_input)

    # Step 1: OCR
    ocr_text = extract_text_from_image(image_input)
    ocr_ok = has_meaningful_text(ocr_text)

    # Step 2: Decide if VLM is needed
    visual = _image_looks_visual(img)
    vlm_desc = ""

    if visual and (ocr_ok or not ocr_ok):  # VLM called if visuals detected
        vlm_desc = describe_image(image_input)

    # Step 3: Combine
    parts = []
    if ocr_text:
        parts.append(f"[OCR]\n{ocr_text}")
    if vlm_desc:
        parts.append(f"[VLM]\n{vlm_desc}")

    combined = "\n\n".join(parts).strip()

    # Route label
    if ocr_ok and vlm_desc:
        route = "ocr+vlm"
    elif ocr_ok:
        route = "ocr"
    elif vlm_desc:
        route = "vlm"
    else:
        route = "none"

    return {
        "ocr_text": ocr_text,
        "vlm_description": vlm_desc,
        "combined_text": combined,
        "route": route,
    }