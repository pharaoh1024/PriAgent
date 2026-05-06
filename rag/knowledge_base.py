"""
RAG knowledge base for Android API semantics.

Improvements over the paper:
1. Hierarchical retrieval: embed both a short "index" doc and a full doc per API,
   then return the full doc after candidate selection (parent-child pattern).
2. Relevance score filtering: discard retrieved docs below a similarity threshold
   to avoid injecting irrelevant context.
3. Lazy index build: build once, persist to disk, reload on subsequent runs.
4. Dual-source loading: merges the hand-annotated Python list (12 entries with
   detailed FP indicators) and the JSON knowledge base (27 class entries with
   broader API coverage including camera, WiFi, biometrics, etc.).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag.android_api_kb import ANDROID_API_ENTRIES
from config import EMBEDDING_MODEL, FAISS_INDEX_DIR, RAG_TOP_K, DATA_DIR


def _entry_to_documents(entry: dict) -> Tuple[Document, Document]:
    """
    Returns two documents per entry:
    - child: short summary for embedding / candidate selection
    - parent: full rich content returned after retrieval
    """
    api_name = entry["api_name"]
    category = entry["category"]
    risk = entry["risk_level"]

    # Child document — compact, optimised for semantic similarity search
    child_text = (
        f"API: {api_name}\n"
        f"Category: {category}\n"
        f"Risk: {risk}\n"
        f"Purpose: {entry['official_purpose']}"
    )

    # Parent document — full context injected into the agent prompt
    fp_bullets = "\n".join(f"  - {fp}" for fp in entry.get("fp_indicators", []))
    use_bullets = "\n".join(f"  - {u}" for u in entry.get("typical_use_patterns", []))
    perms = ", ".join(entry.get("permissions_required", [])) or "none"
    obsolete = entry.get("obsolete_since") or "still active"

    parent_text = (
        f"### {api_name}\n"
        f"**Category**: {category} | **Risk Level**: {risk}\n"
        f"**Required Permissions**: {perms}\n"
        f"**Obsolete Since**: {obsolete}\n\n"
        f"**Official Purpose**:\n{entry['official_purpose']}\n\n"
        f"**Typical Use Patterns**:\n{use_bullets}\n\n"
        f"**False Positive Indicators** (signs the flow may be benign):\n{fp_bullets}"
    )

    metadata = {"api_name": api_name, "category": category, "risk_level": risk}
    child_doc = Document(page_content=child_text, metadata={**metadata, "doc_type": "child"})
    parent_doc = Document(page_content=parent_text, metadata={**metadata, "doc_type": "parent"})
    return child_doc, parent_doc


def _json_class_to_documents(entry: dict) -> Tuple[Document, Document]:
    """
    Convert a JSON-format class entry (from android_privacy_knowledge.json) to
    child+parent documents.  These supplement the hand-annotated Python list with
    broader API coverage (camera, WiFi, biometrics, ConnectivityManager, etc.).
    """
    full_name = entry.get("full_name", entry.get("name", "Unknown"))
    package = entry.get("package", "")
    description = entry.get("description", "")
    permissions = ", ".join(entry.get("required_permissions", [])) or "none required"
    since = entry.get("since_version", "1")
    privacy = entry.get("privacy_impact", {})
    sensitivity = privacy.get("sensitivity", "unknown").upper()
    data_types = ", ".join(privacy.get("data_types_accessed", []))

    method_lines = []
    for m in entry.get("methods", []):
        dep_note = " [DEPRECATED]" if m.get("deprecated") else ""
        method_lines.append(
            f"  - `{m.get('signature', m['name'])}`{dep_note}: {m.get('description', '')}"
        )
    methods_text = "\n".join(method_lines) if method_lines else "  (see class-level documentation)"

    parent_text = (
        f"### {full_name}\n"
        f"**Package**: {package} | **Risk Level**: {sensitivity}\n"
        f"**Required Permissions**: {permissions}\n"
        f"**Available since API {since}** | **Data accessed**: {data_types}\n\n"
        f"**Official Description**: {description}\n\n"
        f"**Privacy Impact**: {privacy.get('description', '')}\n\n"
        f"**Key Methods**:\n{methods_text}"
    )

    child_text = (
        f"API: {full_name}\n"
        f"Package: {package}\n"
        f"Risk: {sensitivity}\n"
        f"Purpose: {description}"
    )

    metadata = {"api_name": full_name, "category": package, "risk_level": sensitivity}
    child_doc = Document(page_content=child_text, metadata={**metadata, "doc_type": "child"})
    parent_doc = Document(page_content=parent_text, metadata={**metadata, "doc_type": "parent"})
    return child_doc, parent_doc


class AndroidAPIKnowledgeBase:
    """
    Vector store over Android sensitive API documentation.

    Uses a parent-child retrieval strategy:
    1. Search the child (compact) documents for the best candidates.
    2. Swap each candidate for its richer parent document before returning to agents.

    This keeps embedding quality high (short, focused texts) while injecting
    full context into the LLM prompt.
    """

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._child_store: FAISS | None = None
        self._parent_lookup: dict[str, str] = {}  # api_name -> parent text
        self._load_or_build()

    # ── index management ────────────────────────────────────────────────

    def _load_or_build(self) -> None:
        index_path = FAISS_INDEX_DIR
        if index_path.exists():
            try:
                self._child_store = FAISS.load_local(
                    str(index_path), self._embeddings,
                    allow_dangerous_deserialization=True
                )
                lookup_file = index_path / "parent_lookup.json"
                if lookup_file.exists():
                    self._parent_lookup = json.loads(lookup_file.read_text())
                    return
            except Exception:
                pass  # corrupted index — rebuild
        self._build()

    def _build(self) -> None:
        child_docs: List[Document] = []

        # Primary source: hand-annotated entries with detailed FP indicators
        for entry in ANDROID_API_ENTRIES:
            child, parent = _entry_to_documents(entry)
            child_docs.append(child)
            self._parent_lookup[entry["api_name"]] = parent.page_content

        # Supplementary source: JSON knowledge base (broader API coverage)
        json_path = DATA_DIR / "android_privacy_knowledge (RAG).json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for entry in data.get("apis", []):
                if entry.get("type") != "class":
                    continue  # skip permission entries — not actionable for verification
                full_name = entry.get("full_name", entry.get("name", ""))
                if full_name in self._parent_lookup:
                    continue  # primary source takes precedence for overlapping APIs
                child, parent = _json_class_to_documents(entry)
                child_docs.append(child)
                self._parent_lookup[full_name] = parent.page_content

        self._child_store = FAISS.from_documents(child_docs, self._embeddings)

        # Persist
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._child_store.save_local(str(FAISS_INDEX_DIR))
        (FAISS_INDEX_DIR / "parent_lookup.json").write_text(
            json.dumps(self._parent_lookup, ensure_ascii=False, indent=2)
        )

    # ── public API ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = RAG_TOP_K,
        score_threshold: float = 0.35,
    ) -> List[str]:
        """
        Return up to *k* parent document texts whose child embeddings are
        semantically similar to *query*.  Documents below *score_threshold*
        are discarded to prevent irrelevant context injection.
        """
        if self._child_store is None:
            return []

        results_with_scores: List[Tuple[Document, float]] = (
            self._child_store.similarity_search_with_relevance_scores(query, k=k)
        )

        parent_texts: List[str] = []
        for doc, score in results_with_scores:
            if score < score_threshold:
                continue
            api_name = doc.metadata.get("api_name", "")
            parent_text = self._parent_lookup.get(api_name, doc.page_content)
            parent_texts.append(parent_text)

        return parent_texts

    def format_context(self, query: str, k: int = RAG_TOP_K) -> str:
        """Return retrieved docs formatted as a single context block for prompt injection."""
        docs = self.retrieve(query, k=k)
        if not docs:
            return "No relevant Android API documentation found."
        return "\n\n---\n\n".join(docs)


# Module-level singleton — built once per process
_kb_instance: AndroidAPIKnowledgeBase | None = None


def get_knowledge_base() -> AndroidAPIKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = AndroidAPIKnowledgeBase()
    return _kb_instance
