from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from storage import sqlite_store
from storage.sqlite_store import DEFAULT_CONFIG_PATH, load_config
from tools.filename_utils import safe_filename


@dataclass
class PaperOrganizationResult:
    paper_id: str
    title: str
    paper_type: str
    primary_topic: str
    year: int | None
    tags: list[dict[str, Any]]
    collections: list[dict[str, Any]]
    original_path: str | None
    organized_path: str | None
    filename_title_source: str | None
    dry_run: bool
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrganizerAgent:
    def __init__(self, sqlite_store_module: Any | None = None, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.sqlite_store = sqlite_store_module or sqlite_store
        self.config_path = Path(config_path)
        self.config = load_config(config_path)
        self.organizer_config = self.config.get("organizer", {})

    def organize_paper(
        self,
        paper_id: str,
        dry_run: bool = True,
        mode: str | None = None,
    ) -> PaperOrganizationResult:
        paper = self.sqlite_store.get_paper_by_id(paper_id, config_path=self.config_path)
        if paper is None:
            raise ValueError(f"Paper not found: {paper_id}")
        chunks = self.sqlite_store.get_chunks_for_paper(paper_id, limit=10, config_path=self.config_path)
        original_path = Path(paper.get("original_path") or paper.get("source_path") or "")
        title, title_source = self.select_existing_title_for_filename(paper, original_path)
        classification = self.classify_paper(paper, chunks)
        action = mode or str(self.organizer_config.get("default_mode") or "copy")

        result = PaperOrganizationResult(
            paper_id=str(paper_id),
            title=title,
            paper_type=classification["paper_type"],
            primary_topic=classification["primary_topic"],
            year=classification["year"],
            tags=[],
            collections=[],
            original_path=str(original_path) if str(original_path) else None,
            organized_path=None,
            filename_title_source=title_source,
            dry_run=dry_run,
            action=action,
            reason=classification["reason"],
        )
        result.tags = self.suggest_tags_for_paper(paper, chunks)
        result.collections = self.suggest_collections(result)
        if result.original_path:
            planned_path = self.build_organized_path(result, result.original_path)
            result.organized_path = str(
                self.apply_file_operation(
                    Path(result.original_path),
                    planned_path,
                    mode=action,
                    dry_run=dry_run,
                )
            )
        if not dry_run:
            self.apply_result(result)
        return result

    def organize_all_papers(
        self,
        dry_run: bool = True,
        mode: str | None = None,
        limit: int | None = None,
    ) -> list[PaperOrganizationResult]:
        papers = self.sqlite_store.list_papers(limit=limit, config_path=self.config_path)
        return [self.organize_paper(str(paper["id"]), dry_run=dry_run, mode=mode) for paper in papers]

    def classify_paper(self, paper: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        text = self._classification_text(paper, chunks)
        paper_type = _match_rules(
            text,
            [
                ("survey", ("survey", "review", "overview", "taxonomy", "comprehensive study")),
                ("benchmark", ("benchmark", "evaluation", "leaderboard", "compare", "comparison")),
                ("dataset", ("dataset", "corpus", "data set")),
                ("system", ("framework", "system", "platform", "toolkit", "pipeline")),
                ("theory", ("theory", "theoretical", "proof", "bound", "convergence")),
                ("application", ("application", "applied", "case study")),
                ("method", ("propose", "proposed", "method", "model", "architecture", "algorithm", "approach")),
            ],
            default="uncategorized",
        )
        primary_topic = _match_rules(
            text,
            [
                ("transformer", ("transformer", "attention", "self-attention")),
                ("rag", ("retrieval-augmented", "retrieval augmented", "rag", "retrieval")),
                ("diffusion", ("diffusion", "denoising")),
                ("graph_neural_network", ("graph neural network", "gnn", "graph")),
                ("reinforcement_learning", ("reinforcement learning", "policy", "reward")),
                ("large_language_model", ("large language model", "language model", "llm")),
                ("multimodal", ("multimodal", "vision-language", "vlm")),
                ("computer_vision", ("computer vision", "object detection", "segmentation", "image")),
                ("nlp", ("natural language processing", "nlp", "text")),
                ("medical_ai", ("medical", "clinical", "healthcare")),
                ("recommender_system", ("recommendation", "recommender")),
                ("time_series", ("time series", "forecasting")),
            ],
            default="unknown",
        )
        year = paper.get("year") or _extract_year(text)
        return {
            "paper_type": paper_type,
            "primary_topic": primary_topic,
            "year": int(year) if year else None,
            "reason": "classified by organizer keyword rules",
        }

    def suggest_tags_for_paper(self, paper: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        classification = self.classify_paper(paper, chunks)
        tags = [
            _tag(classification["paper_type"], "paper_type", 0.9),
            _tag(classification["primary_topic"], "topic", 0.9),
        ]
        text = self._classification_text(paper, chunks)
        for keyword in ("transformer", "rag", "dataset", "benchmark", "architecture", "object detection"):
            if keyword in text and keyword not in {tag["tag"] for tag in tags}:
                tags.append(_tag(keyword.replace(" ", "_"), "keyword", 0.75))
        return tags

    def suggest_collections(self, result: PaperOrganizationResult) -> list[dict[str, Any]]:
        collections: list[dict[str, Any]] = []
        type_names = {
            "survey": "Survey Papers",
            "method": "Method Papers",
            "benchmark": "Benchmark Papers",
            "dataset": "Dataset Papers",
        }
        topic_names = {
            "transformer": "Transformer Papers",
            "rag": "RAG Papers",
            "large_language_model": "Large Language Model Papers",
        }
        if result.paper_type in type_names:
            collections.append(
                {
                    "collection_id": f"type_{result.paper_type}",
                    "name": type_names[result.paper_type],
                    "description": "",
                    "rule": f"paper_type={result.paper_type}",
                    "reason": f"paper_type={result.paper_type}",
                }
            )
        if result.primary_topic in topic_names:
            collections.append(
                {
                    "collection_id": f"topic_{result.primary_topic}",
                    "name": topic_names[result.primary_topic],
                    "description": "",
                    "rule": f"primary_topic={result.primary_topic}",
                    "reason": f"primary_topic={result.primary_topic}",
                }
            )
        if result.year:
            collections.append(
                {
                    "collection_id": f"year_{result.year}",
                    "name": f"Papers from {result.year}",
                    "description": "",
                    "rule": f"year={result.year}",
                    "reason": f"year={result.year}",
                }
            )
        return collections

    def select_existing_title_for_filename(
        self,
        paper: dict[str, Any],
        original_path: Path,
    ) -> tuple[str, str]:
        for key in ("title", "extracted_title", "metadata_title", "paper_title"):
            value = paper.get(key)
            if _valid_title(value):
                return str(value).strip(), key
        metadata = paper.get("metadata") or {}
        pdf_metadata = metadata.get("pdf_metadata") if isinstance(metadata, dict) else None
        if isinstance(pdf_metadata, dict) and _valid_title(pdf_metadata.get("title")):
            return str(pdf_metadata["title"]).strip(), "metadata.pdf_metadata.title"
        return original_path.stem, "original_path.stem"

    def build_organized_path(self, result: PaperOrganizationResult, original_path: str | Path) -> Path:
        library_root = Path(str(self.organizer_config.get("library_root") or "data/library"))
        if not library_root.is_absolute():
            library_root = self.config_path.resolve().parent / library_root
        year_value = str(result.year) if result.year else str(self.organizer_config.get("unknown_year") or "unknown_year")
        max_length = int(self.organizer_config.get("safe_filename_max_length") or 120)
        file_name = f"{safe_filename(result.title, max_length=max_length)}.pdf"
        folder = str(self.organizer_config.get("folder_template") or "{primary_topic}/{year}").format(
            paper_type=result.paper_type,
            primary_topic=result.primary_topic,
            year=year_value,
        )
        return _deduplicate_path(library_root / folder / file_name)

    def apply_file_operation(
        self,
        original_path: Path,
        target_path: Path,
        mode: str = "copy",
        dry_run: bool = True,
    ) -> Path:
        if original_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files can be organized: {original_path}")
        target = _deduplicate_path(target_path)
        if dry_run:
            return target
        if not original_path.exists():
            raise FileNotFoundError(f"PDF file not found: {original_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(original_path, target)
        elif mode == "move":
            try:
                original_path.rename(target)
            except OSError:
                shutil.copy2(original_path, target)
                original_path.unlink()
        else:
            raise ValueError(f"Unsupported organize mode: {mode}")
        return target

    def apply_result(self, result: PaperOrganizationResult) -> None:
        self.sqlite_store.update_paper_organization(
            result.paper_id,
            paper_type=result.paper_type,
            primary_topic=result.primary_topic,
            year=result.year,
            original_path=result.original_path,
            organized_path=result.organized_path,
            filename_title_source=result.filename_title_source,
            organization_status="organized",
            config_path=self.config_path,
        )
        self.sqlite_store.insert_paper_tags(result.paper_id, result.tags, config_path=self.config_path)
        for collection in result.collections:
            self.sqlite_store.create_collection(
                collection["collection_id"],
                collection["name"],
                collection.get("description", ""),
                collection.get("rule", ""),
                config_path=self.config_path,
            )
            self.sqlite_store.add_paper_to_collection(
                collection["collection_id"],
                result.paper_id,
                collection.get("reason", ""),
                config_path=self.config_path,
            )

    def _classification_text(self, paper: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
        parts = [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("source_path") or ""),
        ]
        parts.extend(str(chunk.get("chunk_text") or chunk.get("text") or "") for chunk in chunks[:5])
        return "\n".join(parts).lower()


def _match_rules(text: str, rules: list[tuple[str, tuple[str, ...]]], default: str) -> str:
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return default


def _extract_year(text: str) -> int | None:
    for match in re.finditer(r"\b(19\d{2}|20\d{2})\b", text):
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


def _valid_title(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if len(text) < 5:
        return False
    return text.lower() not in {"unknown", "untitled", "none", "null"}


def _tag(tag: str, tag_type: str, confidence: float) -> dict[str, Any]:
    return {
        "tag": tag,
        "tag_type": tag_type,
        "confidence": confidence,
        "source": "organizer",
    }


def _deduplicate_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available path for {path}")
