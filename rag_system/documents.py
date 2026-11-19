from __future__ import annotations

from pathlib import Path
import re

from docx import Document as WordDocument
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def parse_document(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"不支持的格式：{suffix or '无扩展名'}")
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        pages = [(text, None)]
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    else:
        doc = WordDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            text += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        pages = [(text, None)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    output: list[tuple[str, int | None]] = []
    for text, page in pages:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            continue
        output.extend((chunk.strip(), page) for chunk in splitter.split_text(text) if chunk.strip())
    if not output:
        raise ValueError("文档中没有可索引的文本，扫描版 PDF 请先进行 OCR")
    return output

