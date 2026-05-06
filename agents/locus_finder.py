"""
LocusFinder Agent — Stage 2: Semantic Locus Identification.

Key improvement over the paper:
  - Dynamic τ_loci: instead of a fixed cap of 5, we compute
      τ_loci = clamp(chain_length // 5, TAU_LOCI_MIN, TAU_LOCI_MAX)
    so very long chains get proportionally more loci while short chains
    are not over-analysed.
  - The CoT prompt uses structured output (Pydantic) rather than free text,
    making downstream parsing robust.
"""

from __future__ import annotations
import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from schemas.data_models import (
    AbstractedFlow, LocusAssignment, SemanticLocus, LocusDimension
)
from config import ANTHROPIC_API_KEY, LLM_MODEL, TAU_LOCI_MIN, TAU_LOCI_MAX


# ── Structured output schema for the LLM ────────────────────────────────

class _LocusItem(BaseModel):
    method_name: str = Field(description="Exact method name from the chain")
    dimension: str = Field(
        description=(
            "One of: data_transformation, conditional_gating, "
            "proximity_to_source, proximity_to_sink, api_specificity, data_colocation"
        )
    )
    rationale: str = Field(description="1-2 sentence CoT explanation for selection")
    priority: int = Field(description="Priority rank: 1 = most important", ge=1)


class _LocusFinderOutput(BaseModel):
    loci: List[_LocusItem] = Field(description="Selected semantic loci, ordered by priority")


# ── Prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior Android security engineer performing privacy compliance analysis.
Your task is to identify the SEMANTIC LOCI — the critical methods in a data flow
call chain whose logic truly determines whether the flow is a genuine privacy
violation or a false positive.

Evaluate each method against these five dimensions:
1. **data_transformation** — Can this method alter, sanitise, hash, or aggregate data,
   making it less sensitive by the time it reaches the sink?
2. **conditional_gating** — Does this method contain permission checks, consent flags,
   or boolean gates that could make the flow infeasible at runtime?
3. **proximity_to_source** — Methods immediately after the source reveal the raw data type.
4. **proximity_to_sink** — Methods just before the sink reveal what actually gets transmitted.
5. **api_specificity** — Is this a suspicious, purpose-built class (e.g., data exfil helper)
   rather than a generic utility?
6. **data_colocation** — Does this method aggregate multiple sensitive fields together
   (strong indicator of user profiling)?

Select at most {tau_loci} loci. Return them as valid JSON matching the schema.
Think step-by-step before selecting each locus.
"""

_HUMAN_PROMPT = """\
Data flow:
  Source : {source}
  Sink   : {sink}
  Chain  : {abstracted_chain}

Methods in chain (numbered):
{numbered_chain}

Select the {tau_loci} most semantically significant loci. Return JSON only.
"""


def _dynamic_tau(chain_length: int) -> int:
    """Compute adaptive τ_loci based on call chain length."""
    raw = max(chain_length // 5, TAU_LOCI_MIN)
    return min(raw, TAU_LOCI_MAX)


def _parse_chain_methods(flow: AbstractedFlow) -> List[str]:
    """Extract individual method names from the abstracted chain string."""
    # Split on ' -> ', strip cycle markers but keep the cycle label
    parts = flow.abstracted_chain.split(" -> ")
    return [p.strip() for p in parts if p.strip()]


class LocusFinderAgent:
    """
    Identifies the minimal set of semantically critical methods (loci) within
    each abstracted data flow chain.
    """

    def __init__(self) -> None:
        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        # Use structured output — eliminates JSON parsing fragility
        self._structured_llm = llm.with_structured_output(_LocusFinderOutput)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ])
        self._chain = self._prompt | self._structured_llm

    def identify_loci(self, flow: AbstractedFlow) -> LocusAssignment:
        """Run LocusFinder on a single abstracted flow."""
        methods = _parse_chain_methods(flow)
        tau = _dynamic_tau(len(methods))

        numbered = "\n".join(f"  {i+1:2d}. {m}" for i, m in enumerate(methods))

        raw: _LocusFinderOutput = self._chain.invoke({
            "source": flow.source,
            "sink": flow.sink,
            "abstracted_chain": flow.abstracted_chain,
            "numbered_chain": numbered,
            "tau_loci": tau,
        })

        loci = [
            SemanticLocus(
                method_name=item.method_name,
                dimension=LocusDimension(item.dimension),
                rationale=item.rationale,
                priority=item.priority,
            )
            for item in sorted(raw.loci, key=lambda x: x.priority)[:tau]
        ]

        return LocusAssignment(
            flow_id=flow.flow_id,
            loci=loci,
            tau_loci_used=tau,
        )

    def identify_batch(self, flows: List[AbstractedFlow]) -> List[LocusAssignment]:
        """Process a batch of flows (sequential — parallelism handled by LangGraph)."""
        return [self.identify_loci(f) for f in flows]
