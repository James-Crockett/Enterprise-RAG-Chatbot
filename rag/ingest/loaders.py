from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]


def infer_source_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "paper"
    if ext in [".md", ".markdown"]:
        return "policy"
    return "text"


ACCESS_TIERS = {"public", "internal", "restricted", "confidential"}


def infer_department(path: Path) -> str:
    # layout is <tier>/<department>/file; files directly under a tier are general.
    folders = [p.lower() for p in path.parent.parts]
    for i, folder in enumerate(folders):
        if folder in ACCESS_TIERS and i + 1 < len(folders):
            return folders[i + 1]
    return "general"


def infer_confidentiality(path: Path) -> str:
    lowered = [p.lower() for p in path.parts]
    if "restricted" in lowered or "confidential" in lowered:
        return "restricted"
    if "public" in lowered:
        return "public"
    return "internal"


def load_markdown(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "source_path": str(path).replace("\\", "/"),
        "source_type": infer_source_type(path),
        "department": infer_department(path),
        "confidentiality": infer_confidentiality(path),
        "page": None,
        "title": path.stem,
    }
    return [Document(text=text, metadata=meta)]


def load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "source_path": str(path).replace("\\", "/"),
        "source_type": infer_source_type(path),
        "department": infer_department(path),
        "confidentiality": infer_confidentiality(path),
        "page": None,
        "title": path.stem,
    }
    return [Document(text=text, metadata=meta)]


def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    docs: List[Document] = []

    for i, page in enumerate(reader.pages):
        extracted = page.extract_text() or ""
        extracted = extracted.strip()
        if not extracted:
            continue

        meta = {
            "source_path": str(path).replace("\\", "/"),
            "source_type": infer_source_type(path),
            "department": infer_department(path),
            "confidentiality": infer_confidentiality(path),
            "page": i + 1,  # human-friendly pages
            "title": path.stem,
        }
        docs.append(Document(text=extracted, metadata=meta))

    return docs


def load_documents(input_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in [".md", ".markdown"]:
            docs.extend(load_markdown(path))
        elif ext in [".txt"]:
            docs.extend(load_text(path))
        elif ext == ".pdf":
            docs.extend(load_pdf(path))
    return docs
