import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from tqdm import tqdm

from agent.vibe_func.parallel_utils import parallel_process
from agent.vibe_func.token_estimator import create_token_estimator_function
from agent.vibe_func.utils import parse_json_dict
from llm_client.messages import MessageList, UserMessage

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    tokenizer_model: str = "Qwen/Qwen3-32B"
    prompt_reserved_tokens: int = 1000
    max_prompt_content_tokens: int = 15384
    max_output_tokens: int = 16384
    max_chunk_tokens: int = 4096
    min_chunk_tokens: int = 200
    judge_max_workers: int = 8
    use_llm_judge: bool = True
    use_llm_structure: bool = False


class MarkdownChunker:
    """
    Standalone chunker that mirrors the Extract QA chunker stage logic.
    """

    def __init__(self, llm_client=None, config: Optional[ChunkerConfig] = None):
        config = config or ChunkerConfig()
        self.llm_client = llm_client
        self.token_estimator = create_token_estimator_function(
            model_name=config.tokenizer_model
        )
        self.max_prompt_content_tokens = config.max_prompt_content_tokens
        self.max_output_tokens = config.max_output_tokens
        self.max_chunk_tokens = config.max_chunk_tokens
        self.min_chunk_tokens = config.min_chunk_tokens
        self.judge_max_workers = config.judge_max_workers
        self.use_llm_judge = config.use_llm_judge
        self.use_llm_structure = config.use_llm_structure

    def chunk(self, input_data: Union[str, Path, List[Document]]) -> List[Document]:
        docs = self._load_and_split(input_data)
        docs = self._split_oversized_docs(docs)
        if not any(self._has_header(d.metadata) for d in docs):
            logger.info(
                "No headers detected; returning fallback length-limited chunks: %s",
                len(docs),
            )
            docs = self._attach_token_counts(docs)
            docs = self._llm_filter_docs(docs)
            return self._filter_small_chunks(docs)
        meta_array = self._build_meta_array(docs)

        if self.use_llm_structure and self.llm_client:
            boundaries = self._identify_main_content_boundaries(meta_array)
        else:
            boundaries = {
                "first_chapter_index": 0,
                "end_marker_index": len(docs),
            }
        start_idx = int(boundaries.get("first_chapter_index", 0))
        end_idx = int(boundaries.get("end_marker_index", len(docs)))
        main_docs = docs[start_idx:end_idx]
        main_meta = meta_array[start_idx:end_idx]

        if self.use_llm_structure and self.llm_client:
            chapter_info = self._identify_chapters(main_meta)
            structure = self._assemble_structure(main_meta, chapter_info)
        else:
            structure = []
        merged = self._merge_documents(main_docs, structure)
        merged = self._split_oversized_docs(merged)
        merged = self._attach_token_counts(merged)
        merged = self._llm_filter_docs(merged)
        merged = self._filter_small_chunks(merged)
        logger.info("Chunking complete: %s → %s documents", len(docs), len(merged))
        return merged

    def estimate_tokens(self, text: str) -> int:
        return self._estimate_tokens(text)

    # --- Internal helpers mirrored from the original stage ---

    def _load_and_split(
        self, input_data: Union[str, Path, List[Document]]
    ) -> List[Document]:
        if isinstance(input_data, list) and all(isinstance(d, Document) for d in input_data):
            return input_data

        if isinstance(input_data, (str, Path)):
            try:
                path = Path(input_data) if isinstance(input_data, str) else input_data
                if path.exists():
                    if path.is_dir():
                        raise IsADirectoryError(
                            "Input path points to a directory, expected a file or text content"
                        )
                    if path.suffix.lower() in {".pkl", ".pickle"}:
                        with open(path, "rb") as f:
                            return pickle.load(f)
                    text = path.read_text(encoding="utf-8")
                else:
                    text = str(input_data)
            except OSError as e:
                if e.errno == 36:
                    text = str(input_data)
                else:
                    raise
        else:
            text = str(input_data)

        text = re.sub(r"<span id=\"page-\d+-\d+\"></span>", "", text)
        splitter = MarkdownHeaderTextSplitter(
            [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        return splitter.split_text(text)

    def _build_meta_array(self, docs: List[Document]) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        for d in docs:
            token_count = self._estimate_tokens(d.page_content)
            meta.append(
                {
                    "metadata": d.metadata,
                    "content_preview": d.page_content[:200],
                    "token_count": token_count,
                }
            )

        try:
            serialized = json.dumps(meta, ensure_ascii=False)
            serialized_tokens = self._estimate_tokens(serialized)
            if serialized_tokens > self.max_prompt_content_tokens:
                logger.warning(
                    "Metadata array (%s tokens) exceeds prompt limit (%s); truncating",
                    serialized_tokens,
                    self.max_prompt_content_tokens,
                )
        except Exception:
            pass
        return meta

    def _identify_main_content_boundaries(
        self, meta_array: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.llm_client:
            return {
                "first_chapter_index": 0,
                "end_marker_index": len(meta_array),
            }

        listing = []
        for i, m in enumerate(meta_array):
            listing.append(
                f"\n[Document {i}]\nMetadata: {m.get('metadata')}\nToken Count: {m.get('token_count')}\nContent Preview: {m.get('content_preview')}\n"
            )
        doc_list_str = self._truncate_to_limit("".join(listing))

        prompt = Prompts.IDENTIFY_BOUNDARIES.replace("${doc_list}", doc_list_str)

        try:
            resp = self.llm_client.execute(
                MessageList([UserMessage(content=prompt)]),
                max_tokens=self.max_output_tokens,
                temperature=0,
            )
            parsed = parse_json_dict(resp.get_text_content())
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.warning("Boundary detection failed: %s", e)
        return {
            "first_chapter_index": 0,
            "end_marker_index": len(meta_array),
        }

    def _identify_chapters(self, main_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.llm_client:
            return {"chapters": [["C1", "Main Content", [0, max(0, len(main_meta) - 1)]]]}

        lines = []
        for i, m in enumerate(main_meta):
            meta_str = str(m.get("metadata"))
            lines.append(
                f"[{i}] Metadata: {meta_str} | Tokens: {m.get('token_count')} | Preview: {m.get('content_preview')[:150]}..."
            )
        content_list = self._truncate_to_limit("\n".join(lines))

        prompt = Prompts.IDENTIFY_CHAPTERS.replace("${content_list}", content_list)

        try:
            resp = self.llm_client.execute(
                MessageList([UserMessage(content=prompt)]),
                max_tokens=self.max_output_tokens,
                temperature=0,
            )
            result = parse_json_dict(resp.get_text_content())
            starts = result.get("chapter_starts", [0]) if isinstance(result, dict) else [0]
        except Exception as e:
            logger.warning("Chapter detection failed: %s; using single chapter", e)
            starts = [0]

        chapters: List[List[Any]] = []
        for i, start in enumerate(starts):
            end = (starts[i + 1] - 1) if i < len(starts) - 1 else len(main_meta) - 1
            title = ""
            if 0 <= start < len(main_meta):
                md = main_meta[start].get("metadata", {})
                title = (
                    md.get("Header 1")
                    or md.get("Header 2")
                    or md.get("title")
                    or f"Chapter {i + 1}"
                )
            chapters.append([f"C{i+1}", title, [int(start), int(end)]])
        return {"chapters": chapters}

    def _identify_sections_in_chapter(
        self, chapter_info: List[Any], main_meta: List[Dict[str, Any]]
    ) -> List[List[Any]]:
        if not self.llm_client:
            return []
        chapter_id, chapter_title, (global_start, global_end) = (
            chapter_info[0],
            chapter_info[1],
            chapter_info[2],
        )
        chapter_meta = main_meta[global_start : global_end + 1]

        parts = []
        for i, m in enumerate(chapter_meta):
            idx = global_start + i
            md = m.get("metadata", {})
            header_str = md.get("Header 3") or md.get("Header 2") or md.get("Header 1") or str(md)
            parts.append(
                f"\n[Index {idx}]\nMetadata: {header_str}\nToken Count: {m.get('token_count')}\nContent Preview: {m.get('content_preview')}\n"
            )
        content_list = self._truncate_to_limit("".join(parts))

        ch_num = chapter_id[1:]
        prompt = (
            Prompts.IDENTIFY_SECTIONS.replace("${chapter_title}", str(chapter_title))
            .replace("${chapter_num}", str(ch_num))
            .replace("${content_list}", content_list)
        )

        try:
            resp = self.llm_client.execute(
                MessageList([UserMessage(content=prompt)]),
                max_tokens=self.max_output_tokens,
                temperature=0,
            )
            parsed = parse_json_dict(resp.get_text_content())
            return parsed.get("sections", []) if isinstance(parsed, dict) else []
        except Exception as e:
            logger.warning("Section detection failed for %s: %s", chapter_id, e)
            return []

    def _assemble_structure(
        self, main_meta: List[Dict[str, Any]], chapter_info: Dict[str, Any]
    ) -> List[List[Any]]:
        structure: List[List[Any]] = []
        for ch in chapter_info.get("chapters", []):
            structure.append(ch)
            ch_sections = self._identify_sections_in_chapter(ch, main_meta)
            for s in ch_sections:
                structure.append(s)
        return structure

    def _merge_documents(
        self, main_docs: List[Document], structure: List[List[Any]]
    ) -> List[Document]:
        merged: List[Document] = []
        processed = set()
        for item in structure:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                logger.warning("Skipping malformed structure item: %s", item)
                continue
            bounds = item[2]
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                logger.warning("Skipping structure item with invalid bounds: %s", item)
                continue
            item_id, item_title, (start_idx, end_idx) = item[0], item[1], bounds
            if isinstance(start_idx, list) or isinstance(end_idx, list):
                start_idx, end_idx = start_idx[0], end_idx[1] if isinstance(end_idx, list) else end_idx
            is_chapter = str(item_id).startswith("C")
            is_section = str(item_id).startswith("S")

            if is_chapter:
                ch_num = str(item_id)[1:]
                has_sections = any(str(s[0]).startswith(f"S{ch_num}.") for s in structure)
                if not has_sections:
                    for idx in range(int(start_idx), int(end_idx) + 1):
                        if 0 <= idx < len(main_docs):
                            merged.append(main_docs[idx])
                            processed.add(idx)
            elif is_section:
                buf: List[str] = []
                for idx in range(int(start_idx), int(end_idx) + 1):
                    if 0 <= idx < len(main_docs):
                        buf.append(main_docs[idx].page_content)
                        processed.add(idx)
                if buf:
                    ch_num = str(item_id).split(".")[0][1:]
                    ch_title = f"Chapter {ch_num}"
                    for ch in structure:
                        if str(ch[0]) == f"C{ch_num}":
                            ch_title = ch[1]
                            break
                    merged.append(
                        Document(
                            page_content="\n\n".join(buf),
                            metadata={
                                "Header 1": ch_title,
                                "Header 2": item_title,
                                "is_merged_section": True,
                            },
                        )
                    )

        for idx in range(len(main_docs)):
            if idx not in processed:
                merged.append(main_docs[idx])
        return merged

    def _estimate_tokens(self, text: str) -> int:
        try:
            return (
                int(self.token_estimator(text))
                if self.token_estimator
                else max(1, len(text) // 4)
            )
        except Exception:
            return max(1, len(text) // 4)

    def _truncate_to_limit(self, content: str) -> str:
        if not content:
            return content
        approx_chars = max(64, self.max_prompt_content_tokens * 4)
        if len(content) <= approx_chars:
            return content
        return content[:approx_chars]

    def _has_header(self, metadata: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(metadata, dict):
            return False
        return any(metadata.get(h) for h in ("Header 1", "Header 2", "Header 3"))

    def _split_oversized_docs(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        sharded: List[Document] = []
        for doc in docs:
            tokens = self._estimate_tokens(doc.page_content)
            if tokens <= self.max_chunk_tokens:
                sharded.append(doc)
                continue
            parts = self._split_doc_by_paragraph(doc, self.max_chunk_tokens)
            for part in parts:
                if self._estimate_tokens(part.page_content) <= self.max_chunk_tokens:
                    sharded.append(part)
                else:
                    sharded.extend(self._force_split(part, self.max_chunk_tokens))
        return sharded

    def _force_split(self, doc: Document, max_tokens: int) -> List[Document]:
        """Hard fallback for content with no paragraph/sentence boundaries (e.g. index pages)."""
        text = doc.page_content
        meta = dict(doc.metadata or {})
        separators = [
            (r'(?<=\))\s*[,;]\s*', ", "),
            (r',\s+', ", "),
            (r'\s+', " "),
        ]
        for pattern, join_str in separators:
            parts = re.split(pattern, text)
            if len(parts) < 2:
                continue
            shards: List[Document] = []
            buf: List[str] = []
            buf_tokens = 0
            for p in parts:
                t = self._estimate_tokens(p)
                if buf_tokens + t > max_tokens and buf:
                    shards.append(Document(page_content=join_str.join(buf), metadata=dict(meta)))
                    buf, buf_tokens = [], 0
                buf.append(p)
                buf_tokens += t
            if buf:
                shards.append(Document(page_content=join_str.join(buf), metadata=dict(meta)))
            if all(self._estimate_tokens(s.page_content) <= max_tokens for s in shards):
                return shards
        approx_chars = max(100, max_tokens * 2)
        while approx_chars > 50:
            shards = [
                Document(page_content=text[i:i + approx_chars], metadata=dict(meta))
                for i in range(0, len(text), approx_chars)
                if text[i:i + approx_chars].strip()
            ]
            if all(self._estimate_tokens(s.page_content) <= max_tokens for s in shards):
                return shards
            approx_chars = approx_chars * 2 // 3
        return [Document(page_content=text[i:i + 100], metadata=dict(meta))
                for i in range(0, len(text), 100) if text[i:i + 100].strip()]

    def _split_doc_by_paragraph(self, doc: Document, max_tokens: int) -> List[Document]:
        shards: List[Document] = []
        paras = doc.page_content.split("\n\n")
        buf: List[str] = []
        buf_tokens = 0
        for para in paras:
            t = self._estimate_tokens(para)
            if t > max_tokens:
                if buf:
                    shards.append(
                        Document(
                            page_content="\n\n".join(buf),
                            metadata=dict(doc.metadata or {}),
                        )
                    )
                    buf, buf_tokens = [], 0
                shards.extend(self._split_paragraph_by_sentence(doc, para, max_tokens))
            elif buf_tokens + t > max_tokens:
                if buf:
                    shards.append(
                        Document(
                            page_content="\n\n".join(buf),
                            metadata=dict(doc.metadata or {}),
                        )
                    )
                buf, buf_tokens = [para], t
            else:
                buf.append(para)
                buf_tokens += t
        if buf:
            shards.append(
                Document(
                    page_content="\n\n".join(buf),
                    metadata=dict(doc.metadata or {}),
                )
            )
        return shards

    def _split_paragraph_by_sentence(
        self, doc: Document, para: str, max_tokens: int
    ) -> List[Document]:
        sent_list = para.split(". ")
        shards: List[Document] = []
        buf: List[str] = []
        buf_tokens = 0
        for sent in sent_list:
            st = self._estimate_tokens(sent + ".")
            if buf_tokens + st > max_tokens:
                if buf:
                    shards.append(
                        Document(
                            page_content=". ".join(buf) + ".",
                            metadata=dict(doc.metadata or {}),
                        )
                    )
                    buf, buf_tokens = [], 0
            buf.append(sent)
            buf_tokens += st
        if buf:
            shards.append(
                Document(
                    page_content=". ".join(buf) + ".",
                    metadata=dict(doc.metadata or {}),
                )
            )
        return shards

    def _llm_filter_docs(self, docs: List[Document]) -> List[Document]:
        """Use the configured LLM to drop low-value or garbled chunks."""
        if not self.use_llm_judge or not self.llm_client or not docs:
            return docs

        kept_indices: List[int] = []
        filtered_reasons: Dict[int, str] = {}
        max_workers = max(1, self.judge_max_workers)

        def _judge_single_arg(args: tuple[int, Document]) -> tuple[int, bool, str]:
            idx, doc = args
            content = doc.page_content or ""
            approx_chars = min(
                len(content), max(512, self.max_prompt_content_tokens * 4)
            )
            if len(content) > approx_chars:
                content = content[:approx_chars]
            prompt = Prompts.JUDGE_CHUNK_QUALITY.replace("${content}", content)
            keep = True
            reason = ""
            try:
                resp = self.llm_client.execute(
                    MessageList([UserMessage(content=prompt)]),
                    max_tokens=min(512, self.max_output_tokens),
                    temperature=0,
                )
                parsed = parse_json_dict(resp.get_text_content())
                if isinstance(parsed, dict):
                    keep = bool(parsed.get("keep", True))
                    reason = (
                        str(parsed.get("reason", "")) if parsed.get("reason") else ""
                    )
            except Exception as e:
                logger.warning("LLM judge failed for chunk %s: %s", idx, e)
            finally:
                with progress_lock:
                    progress_bar.update(1)
            return idx, keep, reason

        thread_args = list(enumerate(docs))
        progress_bar = tqdm(
            total=len(thread_args),
            desc="LLM Judge",
            disable=not thread_args,
        )
        progress_lock = Lock()
        try:
            results = parallel_process(
                _judge_single_arg,
                thread_args,
                max_workers=max_workers,
                show_progress=False,
                desc="Judging document chunks",
            )
        finally:
            progress_bar.close()

        for result in results:
            if result:
                idx, keep, reason = result
                if keep:
                    kept_indices.append(idx)
                else:
                    filtered_reasons[idx] = reason

        if filtered_reasons:
            logger.info(
                "LLM judge removed %s chunk(s); remaining %s",
                len(filtered_reasons),
                len(kept_indices),
            )
        return [docs[i] for i in sorted(kept_indices)]

    def _attach_token_counts(self, docs: List[Document]) -> List[Document]:
        """Add token_count into each document's metadata for downstream inspection."""
        annotated: List[Document] = []
        for d in docs:
            md = dict(d.metadata or {})
            md["token_count"] = self._estimate_tokens(d.page_content)
            annotated.append(Document(page_content=d.page_content, metadata=md))
        return annotated

    def _filter_small_chunks(self, docs: List[Document]) -> List[Document]:
        """Drop chunks whose token_count falls below the configured minimum."""
        if not docs or self.min_chunk_tokens <= 0:
            return docs
        filtered: List[Document] = []
        for d in docs:
            tokens = d.metadata.get("token_count")
            tokens = tokens if isinstance(tokens, (int, float)) else self._estimate_tokens(d.page_content)
            if tokens >= self.min_chunk_tokens:
                filtered.append(d)
        if len(filtered) != len(docs):
            logger.info(
                "Filtered out %s chunks below %s tokens; remaining %s",
                len(docs) - len(filtered),
                self.min_chunk_tokens,
                len(filtered),
            )
        return filtered


class Prompts:
    IDENTIFY_BOUNDARIES = (
        "You are analyzing a list of document chunks from an educational or reference text. "
        "Your task is to:\n"
        "1. Identify the FIRST document block that marks the beginning of the main content\n"
        "2. Identify the FIRST document block that comes AFTER the main content ends\n\n"
        "Main content typically starts with numbered/labeled major divisions. "
        "Skip front matter (title, copyright, preface, TOC).\n"
        "Main content typically ends before appendices, references, index, glossary, answers, or additional resources.\n\n"
        "Document list:\n${doc_list}\n\n"
        "Return JSON:\n{\n  \"first_chapter_index\": <int>,\n  \"end_marker_index\": <int>\n}"
    )

    IDENTIFY_CHAPTERS = (
        "Identify the start indices of top-level divisions in this document.\n\n"
        "Content list:\n${content_list}\n\n"
        "Return JSON:\n{\"chapter_starts\": [0, 15, 28, ...]}"
    )

    IDENTIFY_SECTIONS = (
        "You are analyzing chunks from ${chapter_title}. Identify second-level divisions (sections) within this chapter "
        "and group consecutive chunks.\n\n"
        "Chapter content:\n${content_list}\n\n"
        "Return JSON:\n{\n  \"sections\": [\n    [\"S${chapter_num}.1\", \"First Section Title\", [start_idx, end_idx]],\n"
        "    [\"S${chapter_num}.2\", \"Second Section Title\", [start_idx, end_idx]]\n  ]\n}"
    )

    JUDGE_CHUNK_QUALITY = (
    "You are evaluating a text chunk to decide if it should be kept as training material "
    "for knowledge distillation.\n\n"
    "Goal: keep only chunks that contain self-contained, explanatory, professional-level content. "
    "If you are unsure, choose keep=false.\n\n"
    "KEEP (keep=true) only if:\n"
    "- The chunk has at least a few complete sentences that explain a concept, method, result, or definition.\n"
    "- A reader could learn something non-trivial from this chunk without seeing surrounding pages.\n\n"
    "DISCARD (keep=false) if ANY of the following are true:\n"
    "1) The chunk is mostly index-like or glossary-like: terms followed by page numbers, "
    "   cross-references, or markdown links like [282](#page-289-13). Example patterns:\n"
    "   - 'molecular vibrations, 101–103; wave functions, 14–26; ...'\n"
    "   - 'radial artery, **[890](#page-897-3)**, **[918](#page-925-17)** radial collateral ligament, ...'\n"
    "   Even if the chunk is very long, if it is mostly terms + page numbers/links, discard it.\n"
    "2) The chunk is mostly a table, character table, or matrix of symbols/numbers (including markdown tables "
    "   with many '|' characters) and there is no surrounding explanation of what the table means.\n"
    "3) The chunk lacks professional-level knowledge or contains subjective content (e.g. preface, foreword, "
    "   acknowledgements, structural headings like 'Chapter 3 – Results', or navigation text like "
    "   'Index', 'References', 'Table of Contents').\n"
    "4) The chunk is obviously garbled OCR, or mainly broken fragments that are hard to interpret.\n"
    "5) The chunk is mainly pointers to other material (e.g. 'see Figure 2', 'see Chapter 5') without explaining "
    "   the underlying ideas.\n"
    "6) The chunk is an answer key, solution manual, or list of short answers to review/practice questions "
    "   (e.g. '[1] B [2] D [3] C ...' or '[1] The kidneys. [2] X-rays.'). Even if some answers contain "
    "   brief explanations, answer keys are not self-contained teaching material.\n\n"
    "Borderline rule: if the chunk is a mix of noise and a tiny amount of real content, choose keep=false.\n\n"
    "Examples (for calibration only):\n"
    "Example KEEP:\n"
    "Chunk: 'Gradient descent updates parameters by moving opposite to the gradient. Given learning rate η, the update "
    "is θ_{t+1} = θ_t − η∇L(θ_t). This iteratively reduces the loss under mild smoothness assumptions.'\n"
    "Output: {\"keep\": true, \"reason\": \"Self-contained explanation of gradient descent and its update rule.\"}\n\n"
    "Example DISCARD (index-like):\n"
    "Chunk: 'Wave equation, 16; wave functions, 14–26; molecular orbitals, 117; selection rules, 414–415; "
    "semiconductors, 231–234; silicon, 231, 241, 250.'\n"
    "Output: {\"keep\": false, \"reason\": \"Index-style terms with page numbers, no explanatory sentences.\"}\n\n"
    "Now evaluate the following chunk.\n\n"
    "Chunk:\n${content}\n\n"
    "MUST TO RETURN ONLY JSON WITH FORMAT:\n"
    "{\"keep\": true|false, \"reason\": \"short explanation (max 20 words)\"}\n"
)
