from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

WS_RE = re.compile(r"\s+")
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

IMPORTANT_SECTION_KEYWORDS = {
    "indications and usage",
    "dosage and administration",
    "contraindications",
    "warnings",
    "warnings and precautions",
    "boxed warning",
    "adverse reactions",
    "drug interactions",
    "use in specific populations",
    "overdosage",
    "how supplied",
    "drug facts",
    "purpose",
    "uses",
    "directions",
    "other information",
    "inactive ingredients",
    "active ingredient",
    "do not use",
    "ask a doctor before use",
    "stop use and ask a doctor",
}

IGNORE_SECTION_KEYWORDS = {
    "spl product data elements section",
    "table of contents",
    "recent major changes",
    "full prescribing information: contents",
}


def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def is_useful_section(title: str) -> bool:
    lower = title.lower().strip()
    if not lower:
        return False
    if lower in IGNORE_SECTION_KEYWORDS:
        return False
    return any(key in lower for key in IMPORTANT_SECTION_KEYWORDS)


def split_into_chunks(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def find_first_with_attr(root: ET.Element, tag_name: str, attr_name: str) -> str:
    for el in root.iter():
        if local_name(el.tag) == tag_name and attr_name in el.attrib:
            return el.attrib[attr_name]
    return ""


def find_first_text(root: ET.Element, target_tags: set[str]) -> str:
    for el in root.iter():
        if local_name(el.tag) in target_tags:
            text = normalize_text(" ".join(el.itertext()))
            if text:
                return text
    return ""


def guess_drug_name_from_xml(root: ET.Element, fallback: str) -> str:
    for tags in ({"name"}, {"title"}):
        text = find_first_text(root, tags)
        if text and len(text) < 180:
            return text
    return fallback


def extract_sections_from_xml(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    root = ET.parse(file_path).getroot()

    set_id = find_first_with_attr(root, "setid", "root") or file_path.stem
    document_id = find_first_with_attr(root, "id", "root") or set_id
    effective_time = find_first_with_attr(root, "effectivetime", "value")
    drug_name = guess_drug_name_from_xml(root, file_path.stem)

    sections: list[dict] = []
    for sec in root.iter():
        if local_name(sec.tag) != "section":
            continue

        title = ""
        for child in sec.iter():
            if local_name(child.tag) == "title":
                title = normalize_text(" ".join(child.itertext()))
                if title:
                    break

        if not is_useful_section(title):
            continue

        text = normalize_text(" ".join(sec.itertext()))
        if not text or len(text) < 150:
            continue
        if text.lower().startswith(title.lower()):
            text = normalize_text(text[len(title):])
        if len(text) < 100:
            continue

        sections.append({"section_title": title, "text": text})

    return {
        "set_id": set_id,
        "document_id": document_id,
        "drug_name": drug_name,
        "effective_time": effective_time,
        "source_path": str(file_path),
        "sections": sections,
    }


def looks_like_heading(line: str) -> bool:
    cleaned = normalize_text(line)
    if not cleaned:
        return False
    lower = cleaned.lower().strip(":")
    if lower in IGNORE_SECTION_KEYWORDS:
        return False
    if any(key in lower for key in IMPORTANT_SECTION_KEYWORDS):
        return True
    is_upperish = cleaned.upper() == cleaned and 2 <= len(cleaned.split()) <= 12 and len(cleaned) <= 90
    return is_upperish


def split_text_into_sections(text: str) -> list[dict]:
    lines = [line.strip() for line in text.splitlines()]
    sections: list[dict] = []
    current_title = "Full Text"
    current_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        if looks_like_heading(line):
            body = normalize_text(" ".join(current_lines))
            if body and len(body) >= 100:
                sections.append({"section_title": current_title, "text": body})
            current_title = normalize_text(line).strip(":")
            current_lines = []
        else:
            current_lines.append(line)

    body = normalize_text(" ".join(current_lines))
    if body and len(body) >= 100:
        sections.append({"section_title": current_title, "text": body})

    useful_sections = [sec for sec in sections if is_useful_section(sec["section_title"])]
    if useful_sections:
        return useful_sections

    full_text = normalize_text(text)
    if len(full_text) >= 100:
        return [{"section_title": "Full Text", "text": full_text}]
    return []


def guess_set_id_from_text(file_path: Path, text: str) -> str:
    match = UUID_RE.search(text)
    if match:
        return match.group(0)
    return file_path.stem


def guess_drug_name_from_text(file_path: Path, text: str) -> str:
    for line in text.splitlines():
        line = normalize_text(line)
        if line and len(line) < 120:
            return line
    return file_path.stem


def extract_sections_from_text(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "set_id": guess_set_id_from_text(file_path, text),
        "document_id": file_path.stem,
        "drug_name": guess_drug_name_from_text(file_path, text),
        "effective_time": "",
        "source_path": str(file_path),
        "sections": split_text_into_sections(text),
    }


def extract_sections(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".xml":
        return extract_sections_from_xml(file_path)
    return extract_sections_from_text(file_path)


def iter_dailymed_chunks(
    files: Iterable[str | Path],
    max_words: int = 220,
    overlap_words: int = 40,
) -> Iterable[dict]:
    chunk_id = 0
    for file_path in files:
        parsed = extract_sections(file_path)
        for sec in parsed["sections"]:
            for part_idx, chunk_text in enumerate(
                split_into_chunks(sec["text"], max_words=max_words, overlap_words=overlap_words)
            ):
                yield {
                    "chunk_id": chunk_id,
                    "drug_name": parsed["drug_name"],
                    "set_id": parsed["set_id"],
                    "document_id": parsed["document_id"],
                    "effective_time": parsed["effective_time"],
                    "section_title": sec["section_title"],
                    "part_index": part_idx,
                    "source_path": parsed["source_path"],
                    "content": chunk_text,
                }
                chunk_id += 1


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
