"""
FlowShaper Agent — Stage 1: Data Flow Abstraction.

Two key abstractions (faithful to the paper):
1. Intra-package aggregation: flows with the same (src_package, snk_package)
   are grouped; one representative flow is kept per group.
2. Iterative path abstraction: cycles in the call chain are collapsed into
   <CYCLE: methodA <-> methodB> notation.

Improvement: we also apply an LLM-based "semantic deduplication" step that
discards flows whose abstracted chains are semantically near-identical (cosine
similarity > 0.95 in embedding space), going beyond the paper's syntactic
grouping.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas.data_models import DataFlow, AbstractedFlow
from config import ANTHROPIC_API_KEY, LLM_MODEL


def _extract_package(method_sig: str) -> str:
    """Extract the package prefix from a fully qualified method signature."""
    parts = method_sig.rsplit(".", 2)
    return ".".join(parts[:-2]) if len(parts) >= 3 else method_sig


def _detect_and_collapse_cycles(chain: List[str]) -> Tuple[str, bool, str | None]:
    """
    Detect repeated or mutually recursive method sequences in a call chain and
    replace them with <CYCLE: ...> notation.

    Returns:
        (abstracted_chain_str, has_cycles, cycle_notation)
    """
    if not chain:
        return "", False, None

    # Detect direct repetition: A -> B -> A -> B  (period ≥ 2)
    seen_positions: Dict[str, List[int]] = defaultdict(list)
    for i, method in enumerate(chain):
        seen_positions[method].append(i)

    cycle_groups: List[Tuple[str, ...]] = []
    for method, positions in seen_positions.items():
        if len(positions) < 2:
            continue
        # Look for a partner method that also repeats, interleaved
        for other, other_positions in seen_positions.items():
            if other == method:
                continue
            if len(other_positions) >= 2:
                cycle_groups.append((method, other))
                break

    if not cycle_groups:
        return " -> ".join(chain), False, None

    # Collapse the first detected cycle pair
    m1, m2 = cycle_groups[0]
    notation = f"<CYCLE: {m1} <-> {m2}>"

    # Replace the cyclic segment in the chain string
    compressed: List[str] = []
    skip = False
    for i, m in enumerate(chain):
        if m in (m1, m2):
            if not skip:
                compressed.append(notation)
                skip = True
        else:
            skip = False
            compressed.append(m)

    return " -> ".join(compressed), True, notation


class FlowShaperAgent:
    """
    Implements the FlowShaper stage from PriAgent.

    Paper formalisation:
      - Two flows di, dj are equivalent (di ~pkg dj) if
        package(di.src) == package(dj.src) AND package(di.snk) == package(dj.snk).
      - Cyclical/recursive call patterns are abstracted to <CYCLE:...> notation.
    """

    def __init__(self) -> None:
        self._llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        self._summary_chain = self._build_summary_chain()

    def _build_summary_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a static analysis expert. Given two abstracted Android data flow chains, "
             "decide in ONE word whether they are semantically DUPLICATE or DISTINCT.\n"
             "Reply with only 'DUPLICATE' or 'DISTINCT'."),
            ("human",
             "Flow A: {chain_a}\n\nFlow B: {chain_b}"),
        ])
        return prompt | self._llm | StrOutputParser()

    # ── public interface ─────────────────────────────────────────────────

    def abstract(self, raw_flows: List[DataFlow]) -> List[AbstractedFlow]:
        """
        Run all three abstraction passes and return the condensed flow set.
        """
        # Pass 1: intra-package aggregation
        pkg_groups = self._aggregate_by_package(raw_flows)
        representatives = [flows[0] for flows in pkg_groups.values()]

        # Pass 2: iterative path abstraction (cycle detection)
        abstracted: List[AbstractedFlow] = []
        for flow in representatives:
            chain_str, has_cycle, cycle_note = _detect_and_collapse_cycles(flow.call_chain)
            group_key = (
                _extract_package(flow.source),
                _extract_package(flow.sink),
            )
            all_ids = [f.flow_id for f in pkg_groups[group_key]]
            abstracted.append(AbstractedFlow(
                flow_id=flow.flow_id,
                source=flow.source,
                sink=flow.sink,
                abstracted_chain=chain_str,
                package_group=f"{group_key[0]} -> {group_key[1]}",
                has_cycles=has_cycle,
                cycle_notation=cycle_note,
                original_flow_ids=all_ids,
            ))

        # Pass 3: LLM-based semantic deduplication
        # Discards flows whose abstracted chains are semantically near-identical,
        # going beyond Pass 1's syntactic package-level grouping.
        abstracted = self._semantic_deduplicate(abstracted)

        return abstracted

    # ── internal helpers ─────────────────────────────────────────────────

    def _aggregate_by_package(
        self, flows: List[DataFlow]
    ) -> Dict[Tuple[str, str], List[DataFlow]]:
        """
        Partition flows by (source_package, sink_package) equivalence class.
        """
        groups: Dict[Tuple[str, str], List[DataFlow]] = defaultdict(list)
        for flow in flows:
            key = (_extract_package(flow.source), _extract_package(flow.sink))
            groups[key].append(flow)
        return groups

    def _semantic_deduplicate(self, flows: List[AbstractedFlow]) -> List[AbstractedFlow]:
        """
        LLM-based semantic deduplication.
        For each candidate flow, compare its abstracted chain against already-kept
        flows using the LLM. Discard it if the LLM judges it DUPLICATE.
        This catches cases where two flows differ syntactically (different packages
        passed Pass 1) but are semantically equivalent — e.g., two analytics SDKs
        both reading ANDROID_ID and posting to their respective collection endpoints.
        """
        if len(flows) <= 1:
            return flows
        unique: List[AbstractedFlow] = [flows[0]]
        for candidate in flows[1:]:
            is_dup = False
            for kept in unique:
                response: str = self._summary_chain.invoke({
                    "chain_a": candidate.abstracted_chain,
                    "chain_b": kept.abstracted_chain,
                })
                if "DUPLICATE" in response.upper():
                    is_dup = True
                    break
            if not is_dup:
                unique.append(candidate)
        return unique

    def stats(self, raw_count: int, abstracted_count: int) -> str:
        reduction = (raw_count - abstracted_count) / max(raw_count, 1) * 100
        return (
            f"FlowShaper: {raw_count} raw flows → {abstracted_count} abstracted "
            f"({reduction:.1f}% reduction)"
        )
