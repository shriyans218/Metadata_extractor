"""
REST API wrapper around the extraction pipeline (assignment's optional
bonus requirement).

Run with:
    uvicorn app.api.main:app --reload --port 8000

Then POST a .docx or .png file to /extract and get back the six
metadata fields.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.loaders import load_document_text
from app.extraction import extract_fields

load_dotenv(override=True)

app = FastAPI(
    title="Rental Agreement Metadata Extraction API",
    description=(
        "Extracts Agreement Value, Start/End Dates, Renewal Notice "
        "(Days), and both parties from a rental agreement, regardless "
        "of its template, using an LLM-based (non rule-based) pipeline."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_EXTS = {".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = load_document_text(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the document (empty or unreadable scan).",
            )
        result = extract_fields(text)
        return {
            "file_name": file.filename,
            "fields": result.to_dict(),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
