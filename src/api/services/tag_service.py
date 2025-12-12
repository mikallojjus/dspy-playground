"""Tag query service for API endpoints."""

import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from src.api.utils import llm_model
from src.cli.tag_query import TagQueryService
from src.config.prompts.merge_tag_prompt import MERGE_TAG_PROMPT, TAG_CHECKER_PROMPT
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
        self._tag_batch_size = settings.tag_fetch_batch_size

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
    ) -> dict[str, list[Dict[str, int | str]]]:
        """
        Return merge directives (by id) and tag update suggestions for tags created in the window.
        """
        tags_in_range = self.fetch_tags_in_range(start_datetime, end_datetime)

        synonym_suggestions: list[Dict[str, list[str]]] = []
        candidate_map: Dict[int, list[Tag]] = {}
        all_tags: list[Tag] = []
        merge_plan: dict[str, list[Dict[str, int | str]]] = {"merges": [], "updates": []}
        try:
            candidate_map, all_tags = self._find_synonym_candidates(
                tags_in_range, top_k=10, batch_size=self._tag_batch_size
            )
            candidate_map = {
                tag_id: candidates for tag_id, candidates in candidate_map.items() if candidates
            }
            synonym_suggestions = self._score_candidates_with_llm(tags_in_range, candidate_map)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to build tag synonym suggestions", extra={"error": str(exc)})
            if not all_tags:
                all_tags = self._collect_all_tags()
            synonym_suggestions = self._build_fallback_suggestions(tags_in_range, candidate_map)

        merge_plan = self._build_merge_plan(synonym_suggestions, tags_in_range, all_tags)
        try:
            updates, extra_merges = self._build_tag_updates(
                synonym_suggestions,
                merge_plan.get("merges", []),
                tags_in_range,
                all_tags,
            )
            merge_plan["updates"] = updates
            if extra_merges:
                existing_pairs = {
                    (merge["source_tag_id"], merge["target_tag_id"])
                    for merge in merge_plan.get("merges", [])
                    if isinstance(merge, dict)
                }
                for merge in extra_merges:
                    pair = (merge.get("source_tag_id"), merge.get("target_tag_id"))
                    if pair in existing_pairs:
                        continue
                    merge_plan.setdefault("merges", []).append(merge)
                    existing_pairs.add(pair)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to build tag update suggestions", extra={"error": str(exc)})
            merge_plan["updates"] = []

        return merge_plan

    def _find_synonym_candidates(
        self,
        tags_in_range: list[Tag],
        top_k: int = 10,
        batch_size: int | None = None,
    ) -> Tuple[Dict[int, list[Tag]], list[Tag]]:
        """
        Use FAISS to find likely synonym candidates between range tags and the global set,
        fetching tags in batches to reduce memory pressure.

        Returns:
            (mapping of tag_id -> candidate tags, all indexed tags)
        """
        if not tags_in_range:
            return {}, []

        batch_size = batch_size or self._tag_batch_size
        range_tag_ids = {tag.id for tag in tags_in_range if tag.id is not None}

        index: faiss.IndexFlatIP | None = None
        indexed_tags: list[Tag] = []
        tag_index_by_id: Dict[int, int] = {}
        cached_range_embeddings: Dict[int, np.ndarray] = {}

        for batch in self._query_service.iter_all_tags(batch_size=batch_size):
            if not batch:
                continue

            tag_texts = [self._tag_to_text(tag) for tag in batch]
            embeddings = self._encode_tag_texts(tag_texts)
            if embeddings.size == 0:
                continue

            if index is None:
                index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            start_idx = len(indexed_tags)
            indexed_tags.extend(batch)

            for offset, tag in enumerate(batch):
                global_idx = start_idx + offset
                tag_index_by_id[tag.id] = global_idx
                if tag.id in range_tag_ids:
                    cached_range_embeddings[tag.id] = embeddings[offset : offset + 1]

        if index is None or not indexed_tags:
            return {}, indexed_tags

        # Ensure merge planning can see the queried tags even if they were filtered out of the stream.
        indexed_tag_ids = {tag.id for tag in indexed_tags}
        for tag in tags_in_range:
            if tag.id not in indexed_tag_ids:
                indexed_tags.append(tag)

        search_k = min(index.ntotal, top_k + 1)  # +1 to allow skipping self-match.
        if search_k == 0:
            return {}, indexed_tags
        candidates_by_tag: Dict[int, list[Tag]] = {}

        for source_tag in tags_in_range:
            query_vector = cached_range_embeddings.get(source_tag.id)
            if query_vector is None:
                query_vector = self._encode_tag_texts([self._tag_to_text(source_tag)])

            similarities, neighbor_indices = index.search(query_vector, search_k)

            seen_candidates: set[int] = set()
            for score, neighbor_idx in zip(similarities[0], neighbor_indices[0]):
                if neighbor_idx == -1:
                    continue
                candidate_tag = indexed_tags[neighbor_idx]
                if candidate_tag.id == source_tag.id:
                    continue
                if score < self._similarity_threshold:
                    continue
                if candidate_tag.id in seen_candidates:
                    continue

                seen_candidates.add(candidate_tag.id)
                candidates_by_tag.setdefault(source_tag.id, []).append(candidate_tag)

            candidates_by_tag.setdefault(source_tag.id, [])

        return candidates_by_tag, indexed_tags

    def _score_candidates_with_llm(
        self,
        tags_in_range: list[Tag],
        candidates_by_tag: Dict[int, list[Tag]],
    ) -> list[Dict[str, list[str]]]:
        """
        Run the merge prompt to pick exact synonym matches from the embedding candidates.
        """
        batch_size = 100
        prompt_payload: list[dict[str, list[str]]] = []
        skipped_keywords: list[str] = []

        for tag in tags_in_range:
            candidate_tags = [candidate.name for candidate in candidates_by_tag.get(tag.id, [])]
            if candidate_tags:
                prompt_payload.append({"keyword": tag.name, "candidate_tags": candidate_tags})
            else:
                skipped_keywords.append(tag.name)

        if not prompt_payload:
            return [{"keyword": keyword, "exact_synonyms": []} for keyword in skipped_keywords]

        chain = llm_model.build_chain(prompt=MERGE_TAG_PROMPT)
        results: list[Dict[str, list[str]]] = []

        for start_idx in range(0, len(prompt_payload), batch_size):
            batch = prompt_payload[start_idx : start_idx + batch_size]
            raw_response = chain.invoke(
                {"input_section": json.dumps(batch, ensure_ascii=False, indent=2)}
            )
            parsed_response = self._parse_llm_json_response(raw_response)
            results.extend(self._normalize_llm_output(batch, parsed_response))

        results_by_keyword = {entry["keyword"]: entry.get("exact_synonyms", []) for entry in results}

        return [
            {"keyword": tag.name, "exact_synonyms": results_by_keyword.get(tag.name, [])}
            for tag in tags_in_range
        ]

    @staticmethod
    def _parse_llm_json_response(raw_response: str) -> list | dict:
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
    ) -> dict[str, list[Dict[str, int]]]:
        """
        Build a list of merge directives (source -> target) using direct suggestions.
        """
        if not synonym_suggestions:
            return {"merges": [], "updates": []}

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
            return {"merges": [], "updates": []}

        merges: list[Dict[str, int]] = []
        for source_id, target_id in direct_mapping.items():
            if source_id == target_id:
                continue
            source_tag = tag_by_id.get(source_id)
            target_tag = tag_by_id.get(target_id)
            if source_tag and target_tag:
                merges.append({"source_tag_id": source_tag.id, "target_tag_id": target_tag.id})

        return {"merges": merges, "updates": []}

    def _build_tag_updates(
        self,
        synonym_suggestions: list[Dict[str, list[str]]],
        merges: list[Dict[str, int]],
        tags_in_range: list[Tag],
        all_tags: list[Tag],
    ) -> tuple[list[Dict[str, int | str]], list[Dict[str, int]]]:
        """
        Use TAG_CHECKER_PROMPT to vet tags lacking synonyms and merge targets in the window.
        """
        labels_to_check = self._collect_labels_for_tag_checker(
            synonym_suggestions, merges, tags_in_range
        )
        if not labels_to_check:
            return [], []

        tag_checker_results = self._run_tag_checker(labels_to_check)
        if not tag_checker_results:
            return [], []

        tags_by_name: Dict[str, Tag] = {tag.name: tag for tag in tags_in_range if tag.id is not None}
        all_tags_by_name: Dict[str, Tag] = {tag.name: tag for tag in all_tags if tag.id is not None}
        updates: list[Dict[str, int | str]] = []
        extra_merges: list[Dict[str, int]] = []
        seen_ids: set[int] = set()
        existing_merge_pairs = {
            (merge.get("source_tag_id"), merge.get("target_tag_id"))
            for merge in merges
            if isinstance(merge, dict)
        }

        for label in labels_to_check:
            result = tag_checker_results.get(label, {})
            if not isinstance(result, dict):
                continue
            if result.get("is_valid") is True:
                continue
            alternatives = result.get("suggested_alternatives", [])
            if not isinstance(alternatives, list) or not alternatives:
                continue
            new_tag = next((alt for alt in alternatives if isinstance(alt, str)), None)
            if not new_tag:
                continue
            if new_tag == label:
                continue
            tag = tags_by_name.get(label)
            if not tag or tag.id is None or tag.id in seen_ids:
                continue
            if new_tag in all_tags_by_name:
                target_tag = all_tags_by_name[new_tag]
                if target_tag.id is None:
                    continue
                pair = (tag.id, target_tag.id)
                if pair in existing_merge_pairs or tag.id == target_tag.id:
                    continue
                extra_merges.append({"source_tag_id": tag.id, "target_tag_id": target_tag.id})
                existing_merge_pairs.add(pair)
                continue

            seen_ids.add(tag.id)
            updates.append({"id": tag.id, "old_tag": label, "new_tag": new_tag})

        return updates, extra_merges

    @staticmethod
    def _collect_labels_for_tag_checker(
        synonym_suggestions: list[Dict[str, list[str]]],
        merges: list[Dict[str, int]],
        tags_in_range: list[Tag],
    ) -> list[str]:
        """Collect labels needing validation: empty-synonym keywords and in-range merge targets."""
        tag_by_id: Dict[int, Tag] = {tag.id: tag for tag in tags_in_range if tag.id is not None}
        labels: list[str] = []
        seen_labels: set[str] = set()

        for suggestion in synonym_suggestions:
            keyword = suggestion.get("keyword")
            synonyms = suggestion.get("exact_synonyms", [])
            if (
                keyword
                and isinstance(synonyms, list)
                and len(synonyms) == 0
                and keyword not in seen_labels
            ):
                seen_labels.add(keyword)
                labels.append(keyword)

        for merge in merges:
            target_id = merge.get("target_tag_id")
            if not isinstance(target_id, int):
                continue
            target_tag = tag_by_id.get(target_id)
            if not target_tag:
                continue
            if target_tag.name in seen_labels:
                continue
            seen_labels.add(target_tag.name)
            labels.append(target_tag.name)

        return labels

    @staticmethod
    def _run_tag_checker(labels: list[str]) -> Dict[str, Dict[str, object]]:
        """Run the TAG_CHECKER_PROMPT against a list of labels."""
        if not labels:
            return {}

        chain = llm_model.build_chain(prompt=TAG_CHECKER_PROMPT)
        raw_response = chain.invoke(
            {"labels_arr": json.dumps(labels, ensure_ascii=False, indent=2)}
        )
        parsed_response = TagService._parse_llm_json_response(raw_response)

        if isinstance(parsed_response, dict):
            return parsed_response

        logger.warning("Tag checker returned unexpected format", extra={"raw_response": raw_response})
        return {}

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

    def _encode_tag_texts(self, tag_texts: List[str]) -> np.ndarray:
        """Encode a batch of tag texts into normalized embeddings."""
        if not tag_texts:
            return np.asarray([], dtype="float32")

        embeddings = self._encoder.encode(
            tag_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _collect_all_tags(self) -> list[Tag]:
        """Collect all tags in batches to avoid a single large fetch."""
        collected: list[Tag] = []
        for batch in self._query_service.iter_all_tags(batch_size=self._tag_batch_size):
            collected.extend(batch)
        return collected
