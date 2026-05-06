"""
ComplianceArbiter Agent — Stage 4b: Holistic Compliance Synthesis.

Cross-examines True Positive verdicts against policy claims and produces the
final audit report with three risk tiers:
  1. High-Risk Violation — flow undisclosed or contradicts policy.
  2. Undeclared Behavior — flow not mentioned in policy (ambiguous).
  3. Declared Behavior — flow accurately covered by policy (benign).

Improvement: the arbiter also generates a concrete remediation recommendation,
giving developers actionable guidance — something the original paper omits.
"""

from __future__ import annotations
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from schemas.data_models import (
    Verdict, PolicyClaim, AuditReport, RiskCategory, VerdictType
)
from config import ANTHROPIC_API_KEY, LLM_MODEL


class _ArbiterOutput(BaseModel):
    risk_category: str = Field(
        description="One of: 'High-Risk Violation', 'Undeclared Behavior', 'Declared Behavior'"
    )
    analyst_briefing: str = Field(
        description="Narrative explanation correlating code evidence with policy coverage"
    )
    code_evidence: str = Field(description="Key code-level facts cited")
    policy_evidence: str = Field(default="", description="Matching or contradicting policy text")
    recommendation: str = Field(description="Concrete remediation action for the developer")
    data_types_involved: List[str] = Field(
        default_factory=list,
        description="Sensitive data types involved (e.g. device_identifier, location)"
    )


_SYSTEM = """\
You are a privacy compliance officer writing a formal audit report.
Given a verified True Positive data flow and the app's privacy policy claims,
produce a structured audit finding.

Risk categories:
- **High-Risk Violation**: The flow collects/transmits sensitive data that is explicitly
  CONTRADICTED by the policy, or the data type is entirely ABSENT from the policy.
- **Undeclared Behavior**: The data is not mentioned in the policy but the transmission
  could be benign (ambiguous — needs developer clarification).
- **Declared Behavior**: The flow is accurately described and justified in the policy
  (this should rarely appear, as only True Positives reach this stage).

Always include a concrete RECOMMENDATION for the developer.
"""

_HUMAN = """\
=== VERIFIED DATA FLOW (True Positive) ===
Flow ID   : {flow_id}
Source    : {source}
Sink      : {sink}
LLM Verdict Explanation:
{verdict_explanation}

=== PRIVACY POLICY CLAIMS ===
{policy_claims_text}

Produce the compliance audit report as JSON.
"""


def _format_claims(claims: List[PolicyClaim]) -> str:
    if not claims:
        return "No structured claims extracted from policy."
    lines = []
    for c in claims:
        lines.append(
            f"- Data type: {c.data_type}\n"
            f"  Collection stated: {c.collection_stated}\n"
            f"  Purpose: {c.usage_purpose or 'not stated'}\n"
            f"  Third-party sharing: {c.third_party_sharing}\n"
            f"  Excerpt: \"{c.relevant_excerpt}\""
        )
    return "\n".join(lines)


class ComplianceArbiterAgent:
    def __init__(self) -> None:
        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        self._structured_llm = llm.with_structured_output(_ArbiterOutput)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self._chain = self._prompt | self._structured_llm

    def synthesise(
        self,
        verdict: Verdict,
        flow_source: str,
        flow_sink: str,
        policy_claims: List[PolicyClaim],
    ) -> AuditReport:
        raw: _ArbiterOutput = self._chain.invoke({
            "flow_id": verdict.flow_id,
            "source": flow_source,
            "sink": flow_sink,
            "verdict_explanation": verdict.explanation,
            "policy_claims_text": _format_claims(policy_claims),
        })

        try:
            risk = RiskCategory(raw.risk_category)
        except ValueError:
            risk = RiskCategory.UNDECLARED

        return AuditReport(
            flow_id=verdict.flow_id,
            risk_category=risk,
            analyst_briefing=raw.analyst_briefing,
            code_evidence=raw.code_evidence,
            policy_evidence=raw.policy_evidence or None,
            recommendation=raw.recommendation,
            data_types_involved=raw.data_types_involved,
        )

    def synthesise_batch(
        self,
        true_positive_verdicts: List[Verdict],
        flow_lookup: dict,  # flow_id -> AbstractedFlow
        policy_claims: List[PolicyClaim],
    ) -> List[AuditReport]:
        reports = []
        for verdict in true_positive_verdicts:
            flow = flow_lookup.get(verdict.flow_id)
            if flow is None:
                continue
            report = self.synthesise(verdict, flow.source, flow.sink, policy_claims)
            reports.append(report)
        return reports
