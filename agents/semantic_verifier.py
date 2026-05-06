"""
SemanticVerifier Agent — Stage 3: RAG-Powered Semantic Verification.

This is the analytical core of PriAgent.  Key features:
1. Tool-augmented agent: uses @tool-decorated functions to retrieve code on demand.
2. RAG context: Android API knowledge is injected from the FAISS vector store.
3. Few-shot learning: three exemplar patterns teach the LLM to spot common FP types.
4. Structured output: verdict is a Pydantic model — no brittle JSON parsing.

Improvement over the paper — SELF-REFLECTION LOOP:
  If the verdict confidence < REFLECTION_CONFIDENCE_THRESHOLD, a second LLM call
  re-examines the reasoning from a "devil's advocate" perspective and may revise
  the judgment.  This reduces both false-positive over-correction and missed TPs.

Improvement — CodeSummarizer integration:
  Code blocks > TAU_SIZE_TOKENS are compressed before injection,
  preventing context-window overflow.
"""

from __future__ import annotations
import json
import tiktoken
from typing import List, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from schemas.data_models import (
    AbstractedFlow, LocusAssignment, Verdict, VerdictType, FalsePositiveCategory
)
from rag.knowledge_base import get_knowledge_base
from rag.android_api_kb import ANDROID_API_ENTRIES
from agents.code_summarizer import CodeSummarizerAgent
from tools.code_tools import register_decompiled_code, get_method_source
from config import (
    ANTHROPIC_API_KEY, LLM_MODEL, TAU_SIZE_TOKENS,
    REFLECTION_CONFIDENCE_THRESHOLD
)


# ── Structured output schemas ────────────────────────────────────────────

class _VerdictOutput(BaseModel):
    judgment: str = Field(description="'True Positive' or 'False Positive'")
    confidence: float = Field(description="0.0 to 1.0 confidence score", ge=0.0, le=1.0)
    explanation: str = Field(description="Step-by-step Chain-of-Thought reasoning")
    fp_category: str = Field(
        default="",
        description=(
            "If False Positive: one of Data Transformation, API Semantic Misinterpretation, "
            "Condition-Dependent Infeasible Path, Simple Reflection Misresolution, "
            "Core Application Functionality, User Consent Gated. Empty string for True Positive."
        )
    )
    rag_apis_consulted: List[str] = Field(
        default_factory=list,
        description="API names whose documentation was consulted from the knowledge base"
    )


# ── Few-shot exemplars (from paper Section 4, RAG-Powered Verification) ──

FEW_SHOT_EXEMPLARS = """
--- EXEMPLAR 1: User-Initiated Flow (False Positive — Core Application Functionality) ---
Flow: getLastKnownLocation() -> ... -> HttpClient.execute("https://api.weather.com/v1/forecast")
Analysis: The flow is gated by User.onClick() at the entry. The sink URL is api.weather.com,
a well-known weather service. The privacy policy states "we use your location to provide
weather forecasts." The location data is sent only when the user actively requests a forecast.
Verdict: FALSE POSITIVE — Core Application Functionality (confidence: 0.95)

--- EXEMPLAR 2: Deprecated API Returns Null (False Positive — API Semantic Misinterpretation) ---
Flow: TelephonyManager.getDeviceId() -> ... -> HttpClient.execute("https://analytics.example.com")
Analysis: The app targets Android API 29+. getDeviceId() returns null for non-privileged apps
on Android 10 and later. The null value is passed down the chain and the HTTP request body
contains a null IMEI field — no actual device identifier is leaked.
Verdict: FALSE POSITIVE — API Semantic Misinterpretation (confidence: 0.90)

--- EXEMPLAR 3: Undisclosed PII Exfiltration (True Positive) ---
Flow: ContactsContract.query() -> UserProfileHelper.buildPayload() -> HttpClient.execute("https://thirdparty-data.io/ingest")
Analysis: The method UserProfileHelper.buildPayload() aggregates contact names, phone numbers,
and device ID into a JSON payload. The sink URL "thirdparty-data.io" is a third-party data
broker domain. The privacy policy does NOT mention sharing contacts with third parties.
Verdict: TRUE POSITIVE (confidence: 0.92)
"""


# ── Prompts ───────────────────────────────────────────────────────────────

_VERIFIER_SYSTEM = """\
You are an expert Android privacy security analyst performing compliance auditing.
Your task is to determine whether a flagged data flow is a True Positive (genuine privacy
violation) or a False Positive (benign, intended functionality).

You have access to:
1. The abstracted data flow path.
2. Source code or summaries of the most semantically critical methods (semantic loci).
3. Official Android API documentation retrieved from the knowledge base.
4. Few-shot exemplars of common false positive patterns.
5. Historical audit patterns from prior sessions.

Perform Chain-of-Thought reasoning: first analyse each piece of evidence, then synthesise
a verdict with a confidence score.

{few_shot_examples}

--- ANDROID API KNOWLEDGE (RAG-retrieved) ---
{rag_context}

--- HISTORICAL AUDIT PATTERNS ---
{pattern_hint}
"""

_VERIFIER_HUMAN = """\
=== DATA FLOW TO ANALYSE ===
Flow ID : {flow_id}
Source  : {source}
Sink    : {sink}
Chain   : {abstracted_chain}

=== SEMANTIC LOCI (critical methods) ===
{loci_with_code}

Now perform your analysis and return your verdict as JSON.
"""

_REFLECTION_SYSTEM = """\
You are a critical reviewer of security analysis verdicts.
A colleague has produced the following verdict on an Android data flow.
Your job: play devil's advocate. Challenge the reasoning, consider alternative
interpretations, and either confirm or revise the verdict.
Be especially sceptical if the original confidence is low (below 0.65).
Return the revised verdict as JSON using the same schema.
"""

_REFLECTION_HUMAN = """\
Original Verdict:
{original_verdict_json}

Data Flow:
{flow_summary}

Challenge the reasoning and produce a revised verdict.
"""


class SemanticVerifierAgent:
    """
    Verifies each abstracted data flow against code evidence, RAG-retrieved API
    knowledge, and few-shot exemplars.  Optionally applies a self-reflection pass
    for low-confidence verdicts.
    """

    def __init__(self, decompiled_code: Dict[str, str]) -> None:
        register_decompiled_code(decompiled_code)
        self._kb = get_knowledge_base()
        self._summarizer = CodeSummarizerAgent()
        self._enc = tiktoken.get_encoding("cl100k_base")

        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        self._structured_llm = llm.with_structured_output(_VerdictOutput)
        self._verifier_prompt = ChatPromptTemplate.from_messages([
            ("system", _VERIFIER_SYSTEM),
            ("human", _VERIFIER_HUMAN),
        ])
        self._reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", _REFLECTION_SYSTEM),
            ("human", _REFLECTION_HUMAN),
        ])
        self._chain = self._verifier_prompt | self._structured_llm
        self._reflect_chain = self._reflection_prompt | self._structured_llm

    # ── public interface ─────────────────────────────────────────────────

    def verify(
        self,
        flow: AbstractedFlow,
        assignment: LocusAssignment,
        pattern_hint: str = "",
    ) -> Verdict:
        # 1. Gather evidence for each locus
        loci_with_code = self._gather_locus_evidence(assignment)

        # 2. Build RAG context from source + sink API names
        rag_query = f"{flow.source} {flow.sink} Android privacy data flow"
        rag_context = self._kb.format_context(rag_query)

        # 3. Determine which APIs were consulted
        consulted = [
            entry["api_name"]
            for entry in ANDROID_API_ENTRIES
            if entry["api_name"] in rag_context
        ]

        # 4. First-pass verdict
        raw: _VerdictOutput = self._chain.invoke({
            "flow_id": flow.flow_id,
            "source": flow.source,
            "sink": flow.sink,
            "abstracted_chain": flow.abstracted_chain,
            "loci_with_code": loci_with_code,
            "few_shot_examples": FEW_SHOT_EXEMPLARS,
            "rag_context": rag_context,
            "pattern_hint": pattern_hint or "No prior patterns available.",
        })

        reflected = False

        # 5. Self-reflection for low-confidence verdicts
        if raw.confidence < REFLECTION_CONFIDENCE_THRESHOLD:
            flow_summary = (
                f"Source: {flow.source}\n"
                f"Sink: {flow.sink}\n"
                f"Chain: {flow.abstracted_chain}"
            )
            raw = self._reflect_chain.invoke({
                "original_verdict_json": json.dumps(raw.model_dump(), indent=2),
                "flow_summary": flow_summary,
            })
            reflected = True

        # 6. Map to domain model
        fp_cat = None
        if raw.fp_category:
            try:
                fp_cat = FalsePositiveCategory(raw.fp_category)
            except ValueError:
                fp_cat = FalsePositiveCategory.DATA_TRANSFORMATION  # safe default

        return Verdict(
            flow_id=flow.flow_id,
            judgment=VerdictType(raw.judgment),
            explanation=raw.explanation,
            confidence=raw.confidence,
            fp_category=fp_cat,
            reflected=reflected,
            rag_apis_consulted=consulted or raw.rag_apis_consulted,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _gather_locus_evidence(self, assignment: LocusAssignment) -> str:
        """Retrieve and optionally summarise source code for each locus."""
        parts = []
        for locus in assignment.loci:
            code = get_method_source.invoke(locus.method_name)
            if code.startswith("[NOT FOUND]"):
                evidence = f"[Source unavailable]\nReason selected: {locus.rationale}"
            else:
                token_count = len(self._enc.encode(code))
                if token_count > TAU_SIZE_TOKENS:
                    code = self._summarizer.summarize(locus.method_name, code)
                    evidence = f"[SUMMARIZED — {token_count} tokens compressed]\n{code}"
                else:
                    evidence = code

            parts.append(
                f"**{locus.method_name}** (dimension: {locus.dimension.value}, "
                f"priority: {locus.priority})\n"
                f"Reason selected: {locus.rationale}\n"
                f"```java\n{evidence}\n```"
            )
        return "\n\n".join(parts)
