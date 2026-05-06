from __future__ import annotations
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# ── Input layer ─────────────────────────────────────────────────────────

class DataFlow(BaseModel):
    """A single taint flow produced by a static analysis tool."""
    flow_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str           # e.g. "android.telephony.TelephonyManager.getDeviceId()"
    sink: str             # e.g. "okhttp3.OkHttpClient.newCall()"
    call_chain: List[str] # ordered list of intermediate method signatures
    app_package: str = ""
    app_name: str = ""

    @property
    def chain_length(self) -> int:
        return len(self.call_chain)


# ── Stage 1: FlowShaper output ───────────────────────────────────────────

class AbstractedFlow(BaseModel):
    """Data flow after cycle detection and package-level aggregation."""
    flow_id: str
    source: str
    sink: str
    abstracted_chain: str        # human-readable, with <CYCLE:...> markers
    package_group: str = ""      # "com.example.app.network"
    has_cycles: bool = False
    cycle_notation: Optional[str] = None
    original_flow_ids: List[str] = Field(default_factory=list)


# ── Stage 2: LocusFinder output ──────────────────────────────────────────

class LocusDimension(str, Enum):
    DATA_TRANSFORM = "data_transformation"
    CONDITIONAL_GATE = "conditional_gating"
    SOURCE_PROXIMITY = "proximity_to_source"
    SINK_PROXIMITY = "proximity_to_sink"
    API_SPECIFICITY = "api_specificity"
    DATA_COLOCATION = "data_colocation"


class SemanticLocus(BaseModel):
    """A critical method selected as a semantic locus for deep inspection."""
    method_name: str
    dimension: LocusDimension
    rationale: str    # agent's CoT reasoning for selection
    priority: int     # 1 = highest priority


class LocusAssignment(BaseModel):
    flow_id: str
    loci: List[SemanticLocus]
    tau_loci_used: int   # actual cap applied (dynamic, not always 5)


# ── Stage 3: SemanticVerifier output ────────────────────────────────────

class VerdictType(str, Enum):
    TRUE_POSITIVE = "True Positive"
    FALSE_POSITIVE = "False Positive"


class FalsePositiveCategory(str, Enum):
    DATA_TRANSFORMATION = "Data Transformation"
    API_MISINTERPRETATION = "API Semantic Misinterpretation"
    INFEASIBLE_PATH = "Condition-Dependent Infeasible Path"
    SIMPLE_REFLECTION = "Simple Reflection Misresolution"
    CORE_FUNCTIONALITY = "Core Application Functionality"
    USER_CONSENT = "User Consent Gated"


class Verdict(BaseModel):
    flow_id: str
    judgment: VerdictType
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    fp_category: Optional[FalsePositiveCategory] = None  # set when FP
    reflected: bool = False    # True if self-reflection was triggered
    rag_apis_consulted: List[str] = Field(default_factory=list)


# ── Stage 4a: PolicyScanner output ──────────────────────────────────────

class PolicyClaim(BaseModel):
    data_type: str           # e.g. "device_identifier", "location"
    collection_stated: bool
    usage_purpose: Optional[str] = None
    third_party_sharing: bool = False
    user_control_mentioned: bool = False
    relevant_excerpt: str    # verbatim policy text supporting the claim


# ── Stage 4b: ComplianceArbiter output (final) ───────────────────────────

class RiskCategory(str, Enum):
    HIGH_RISK = "High-Risk Violation"
    UNDECLARED = "Undeclared Behavior"
    DECLARED = "Declared Behavior"


class AuditReport(BaseModel):
    flow_id: str
    risk_category: RiskCategory
    analyst_briefing: str       # LLM narrative explanation
    code_evidence: str          # key code snippets cited
    policy_evidence: Optional[str] = None   # matching/contradicting policy text
    recommendation: str         # actionable remediation suggestion
    data_types_involved: List[str] = Field(default_factory=list)


# ── LangGraph state ──────────────────────────────────────────────────────
# Using TypedDict is idiomatic for LangGraph, but we expose Pydantic helpers
# for external interfaces.

class AuditSummary(BaseModel):
    """Top-level result returned to callers."""
    app_name: str
    session_id: str
    total_raw_flows: int
    false_positives_eliminated: int
    fp_reduction_rate: float
    confirmed_violations: List[AuditReport]
    declared_behaviors: List[AuditReport]
    processing_cost_estimate: Dict[str, Any] = Field(default_factory=dict)
