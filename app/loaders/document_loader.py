"""
Template-agnostic document loader.

Converts any supported input file (.docx or scanned image .png/.jpg)
into raw plain text, so the extraction layer downstream never has to
know or care what the source format or template was.

Design note: OCR and docx-parsing are both treated as "text acquisition"
steps only. No field-specific logic, no regex, no keyword search lives
here -- this module's only job is: file -> clean text.
"""

from __future__ import annotations

import os
from pathlib import Path

import platform

import pytesseract
from PIL import Image
from docx import Document

# On Windows, Tesseract usually isn't auto-added to PATH by the
# installer. If the default install location exists, point
# pytesseract at it directly so users don't have to fight PATH.
# Override with the TESSERACT_CMD env var if installed elsewhere.
_default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif platform.system() == "Windows" and os.path.exists(_default_windows_path):
    pytesseract.pytesseract.tesseract_cmd = _default_windows_path


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
SUPPORTED_DOCX_EXTS = {".docx"}
_ALL_KNOWN_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_DOCX_EXTS | {".pdf", ".doc"}


def _clean_stem(path: Path) -> str:
    """Strip ALL trailing known-extension suffixes, not just the last one.

    Some provided filenames are double-suffixed, e.g.
    "228094620-Rental-Agreement.pdf.docx" -- pathlib's `.stem` only
    strips the outermost ".docx", leaving ".pdf" stuck on the end
    ("228094620-Rental-Agreement.pdf"). That mismatches the "File Name"
    column in train.csv/test.csv ("228094620-Rental-Agreement"), which
    means predict.py's output and evaluate.py's ground-truth join key
    don't line up. Since evaluate.py does an (inner) merge on File Name,
    any row that doesn't match is SILENTLY dropped from the recall
    calculation entirely -- not counted as a miss, just excluded. Loop
    until no known extension remains so this can't happen again with a
    triple-suffixed name either.
    """
    stem = path.stem
    while Path(stem).suffix.lower() in _ALL_KNOWN_EXTS:
        stem = Path(stem).stem
    return stem


def load_docx_text(path: str | Path) -> str:
    """Extract plain text from a .docx file, including table cells,
    since some rental agreements place party/date info in tables."""
    doc = Document(path)
    parts: list[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())

    return "\n".join(parts)


def load_image_text(path: str | Path) -> str:
    """OCR a scanned document image into plain text using Tesseract.

    Tesseract is used purely as a text-acquisition engine (like a
    scanner), not as part of the extraction logic itself -- the actual
    field understanding happens later, in the LLM extraction layer.
    """
    image = Image.open(path)
    # Simple preprocessing: convert to grayscale, which measurably
    # improves Tesseract accuracy on scanned contract images.
    image = image.convert("L")
    text = pytesseract.image_to_string(image)
    return text.strip()


def load_document_text(path: str | Path) -> str:
    """Dispatch on file extension and return extracted plain text.

    Raises ValueError for unsupported file types so failures are loud
    and explicit rather than silently returning empty text.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in SUPPORTED_DOCX_EXTS:
        return load_docx_text(path)
    elif ext in SUPPORTED_IMAGE_EXTS:
        return load_image_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}' for {path.name}. "
            f"Supported: {SUPPORTED_DOCX_EXTS | SUPPORTED_IMAGE_EXTS}"
        )


def load_documents_from_dir(directory: str | Path) -> dict[str, str]:
    """Load every supported document in a directory.

    Returns a dict mapping the file's stem (no extension) to its
    extracted text, so results line up with the "File Name" column
    used in train.csv / test.csv.
    """
    directory = Path(directory)
    results: dict[str, str] = {}

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in (SUPPORTED_DOCX_EXTS | SUPPORTED_IMAGE_EXTS):
            try:
                text = load_document_text(file_path)
                results[_clean_stem(file_path)] = text
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] Failed to load {file_path.name}: {exc}")
                results[_clean_stem(file_path)] = ""

    return results
