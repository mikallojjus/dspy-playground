"""Tag query service for API endpoints."""

import json
import re
from datetime import datetime
from typing import Dict, Tuple

import faiss
import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from src.api.utils import llm_model
from src.cli.tag_query import TagQueryService
from src.config.prompts.merge_tag_prompt import MERGE_TAG_PROMPT
from src.config.settings import settings
from src.database.models import Tag
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
faiss.omp_set_num_threads(1)


class TagService:
    """Service wrapper around CLI TagQueryService for API usage."""

    def __init__(self, db_session: Session):
        self._query_service = TagQueryService(db_session)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._similarity_threshold = settings.embedding_similarity_threshold

    def fetch_tags_in_range(self, start_datetime: datetime, end_datetime: datetime) -> list[Tag]:
        """Return tags created within the provided datetime window."""
        logger.info(
            "Fetching tags in range",
            extra={"start_datetime": start_datetime, "end_datetime": end_datetime},
        )
        return self._query_service.get_tags_created_between(start_datetime, end_datetime)

    def fetch_all_tags(self) -> list[Tag]:
        """Return all tags ordered by creation date."""
        logger.info("Fetching all tags")
        return self._query_service.get_all_tags()

    def fetch_tag_merge_snapshot(
        self, start_datetime: datetime, end_datetime: datetime
    ) -> Tuple[list[Dict[str, int]], list[int]]:
        """
        Return merge directives (by id) and tags that should be deleted (by id).
        """
        tags_in_range = self.fetch_tags_in_range(start_datetime, end_datetime)
        all_tags = self.fetch_all_tags()

        synonym_suggestions: list[Dict[str, list[str]]] = []
        candidate_map: Dict[int, list[Tag]] = {}
        try:
            candidate_map = self._find_synonym_candidates(tags_in_range, all_tags)
            synonym_suggestions = self._score_candidates_with_llm(tags_in_range, candidate_map)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to build tag synonym suggestions", extra={"error": str(exc)})
            synonym_suggestions = self._build_fallback_suggestions(tags_in_range, candidate_map)

        merges, deletes = self._build_merge_plan(synonym_suggestions, tags_in_range, all_tags)
        return merges, deletes

    def _find_synonym_candidates(
        self,
        tags_in_range: list[Tag],
        all_tags: list[Tag],
        top_k: int = 10,
    ) -> Dict[int, list[Tag]]:
        """
        Use FAISS to find likely synonym candidates between range tags and the global set.
        Returns a mapping of tag_id -> candidate tags.
        """
        if not tags_in_range or not all_tags:
            return {}

        tag_texts = [self._tag_to_text(tag) for tag in all_tags]
        all_embeddings = self._encoder.encode(
            tag_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings = np.asarray(all_embeddings, dtype="float32")
        faiss.normalize_L2(all_embeddings)

        if all_embeddings.size == 0:
            return {}

        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings)

        # Prefer the embedding we already computed if the tag exists in all_tags.
        tag_index_by_id = {tag.id: idx for idx, tag in enumerate(all_tags)}
        search_k = min(len(all_tags), top_k + 1)  # +1 to allow skipping self-match.

        candidates_by_tag: Dict[int, list[Tag]] = {}

        for source_tag in tags_in_range:
            if source_tag.id in tag_index_by_id:
                query_vector = all_embeddings[tag_index_by_id[source_tag.id]].reshape(1, -1)
            else:
                query_vector = self._encoder.encode(
                    [self._tag_to_text(source_tag)],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")
                faiss.normalize_L2(query_vector)

            similarities, neighbor_indices = index.search(query_vector, search_k)

            seen_candidates: set[int] = set()
            for score, neighbor_idx in zip(similarities[0], neighbor_indices[0]):
                if neighbor_idx == -1:
                    continue
                candidate_tag = all_tags[neighbor_idx]
                if candidate_tag.id == source_tag.id:
                    continue
                if score < self._similarity_threshold:
                    continue
                if candidate_tag.id in seen_candidates:
                    continue

                seen_candidates.add(candidate_tag.id)
                candidates_by_tag.setdefault(source_tag.id, []).append(candidate_tag)

            candidates_by_tag.setdefault(source_tag.id, [])

        return candidates_by_tag

    def _score_candidates_with_llm(
        self,
        tags_in_range: list[Tag],
        candidates_by_tag: Dict[int, list[Tag]],
    ) -> list[Dict[str, list[str]]]:
        """
        Run the merge prompt to pick exact synonym matches from the embedding candidates.
        """
        batch_size = 5
        prompt_payload = [
            {
                "keyword": tag.name,
                "candidate_tags": [candidate.name for candidate in candidates_by_tag.get(tag.id, [])],
            }
            for tag in tags_in_range
        ]

        if not prompt_payload:
            return []

        if not any(entry["candidate_tags"] for entry in prompt_payload):
            return [{"keyword": entry["keyword"], "exact_synonyms": []} for entry in prompt_payload]

        chain = llm_model.build_chain(prompt=MERGE_TAG_PROMPT)
        results: list[Dict[str, list[str]]] = []

        for start_idx in range(0, len(prompt_payload), batch_size):
            batch = prompt_payload[start_idx : start_idx + batch_size]
            raw_response = chain.invoke(
                {"input_section": json.dumps(batch, ensure_ascii=False, indent=2)}
            )
            parsed_response = self._parse_llm_json_response(raw_response)
            results.extend(self._normalize_llm_output(batch, parsed_response))

        return results

    @staticmethod
    def _parse_llm_json_response(raw_response: str) -> list:
        """
        Parse the LLM response into JSON, handling optional fenced code blocks.
        """
        try:
            return json.loads(raw_response)
        except Exception:
            response_text = raw_response.strip()
            fenced_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE
            )
            if fenced_match:
                response_text = fenced_match.group(1)
            return json.loads(response_text)

    def _normalize_llm_output(
        self,
        prompt_payload: list[dict[str, list[str]]],
        parsed_response: list,
    ) -> list[Dict[str, list[str]]]:
        """
        Keep only synonyms that were present in the candidate list and preserve ordering.
        """
        if not isinstance(parsed_response, list):
            parsed_response = []

        candidates_by_keyword = {
            entry["keyword"]: set(entry.get("candidate_tags", [])) for entry in prompt_payload
        }
        keyword_order = [entry["keyword"] for entry in prompt_payload]

        selections: Dict[str, list[str]] = {keyword: [] for keyword in keyword_order}

        for item in parsed_response or []:
            keyword = item.get("keyword")
            if not keyword or keyword not in candidates_by_keyword:
                continue
            exact_synonyms = item.get("exact_synonyms", [])
            if not isinstance(exact_synonyms, list):
                continue
            filtered = [
                synonym
                for synonym in exact_synonyms
                if isinstance(synonym, str) and synonym in candidates_by_keyword[keyword]
            ]
            selections[keyword] = filtered

        return [
            {"keyword": keyword, "exact_synonyms": selections.get(keyword, [])}
            for keyword in keyword_order
        ]

    @staticmethod
    def _build_fallback_suggestions(
        tags_in_range: list[Tag],
        candidates_by_tag: Dict[int, list[Tag]],
    ) -> list[Dict[str, list[str]]]:
        """Fallback to the raw embedding candidates if LLM scoring fails."""
        return [
            {
                "keyword": tag.name,
                "exact_synonyms": [candidate.name for candidate in candidates_by_tag.get(tag.id, [])],
            }
            for tag in tags_in_range
        ]

    @staticmethod
    def _build_merge_plan(
        synonym_suggestions: list[Dict[str, list[str]]],
        tags_in_range: list[Tag],
        all_tags: list[Tag],
    ) -> Tuple[list[Dict[str, int]], list[int]]:
        """
        Build a list of merge directives (source -> target) and tags to delete,
        flattening transitive chains (e.g., 1->2->3 becomes 1->3,2->3).
        """
        if not synonym_suggestions:
            return [], []

        tag_by_name: Dict[str, Tag] = {tag.name: tag for tag in all_tags}
        tag_by_id: Dict[int, Tag] = {tag.id: tag for tag in all_tags}
        tag_by_name.update({tag.name: tag for tag in tags_in_range})
        tag_by_id.update({tag.id: tag for tag in tags_in_range})
        source_by_name: Dict[str, Tag] = {tag.name: tag for tag in tags_in_range}

        direct_mapping: Dict[int, int] = {}

        for suggestion in synonym_suggestions:
            keyword = suggestion.get("keyword")
            if not keyword:
                continue
            source_tag = source_by_name.get(keyword)
            if not source_tag:
                continue
            exact_synonyms = suggestion.get("exact_synonyms", [])
            if not isinstance(exact_synonyms, list):
                continue

            target_tag = None
            for candidate_name in exact_synonyms:
                if not isinstance(candidate_name, str):
                    continue
                candidate_tag = tag_by_name.get(candidate_name)
                if candidate_tag:
                    target_tag = candidate_tag
                    break
            if not target_tag:
                continue

            direct_mapping[source_tag.id] = target_tag.id

        if not direct_mapping:
            return [], []

        def find_root(tag_id: int) -> int:
            seen: set[int] = set()
            current = tag_id
            while current in direct_mapping and current not in seen:
                seen.add(current)
                current = direct_mapping[current]
            root = current
            for visited in seen:
                direct_mapping[visited] = root
            return root

        merges: list[Dict[str, int]] = []
        merges_seen: set[Tuple[int, int]] = set()
        delete_ids: set[int] = set()

        for source_id in list(direct_mapping.keys()):
            target_id = find_root(source_id)
            if target_id == source_id:
                continue
            delete_ids.add(source_id)
            pair_key = (source_id, target_id)
            if pair_key in merges_seen:
                continue
            merges_seen.add(pair_key)
            source_tag = tag_by_id.get(source_id)
            target_tag = tag_by_id.get(target_id)
            if source_tag and target_tag:
                merges.append({"source_tag_id": source_tag.id, "target_tag_id": target_tag.id})

        # Include intermediate sources that became part of a chain in delete set.
        delete_ids = {tag_id for tag_id in delete_ids if tag_id in tag_by_id}
        return merges, sorted(delete_ids)

    @staticmethod
    def _tag_to_text(tag: Tag) -> str:
        """Create an embedding-friendly text representation of a tag."""
        if tag.description:
            return f"{tag.name}: {tag.description}"
        return tag.name

    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using SentenceTransformer."""
        embedding = self._encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype="float32").tolist()
