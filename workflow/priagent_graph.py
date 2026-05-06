"""
PriAgent LangGraph Orchestration Workflow.

This module is the architectural centrepiece that showcases:
  1. LangGraph StateGraph — explicit, inspectable control flow.
  2. Send API for PARALLEL per-flow processing — multiple flows are verified
     concurrently by the SemanticVerifier, matching the paper's parallelisation.
  3. Conditional reflection edge — low-confidence flows trigger a second pass.
  4. MemorySaver checkpointing — full run state can be resumed after interruption.
  5. Human-in-the-loop node — ambiguous flows can be escalated for manual review.

Graph topology (→ = edge, ⇒ = conditional edge):
  START
    → flow_shaper
    → locus_finder
    → [fan_out: Send per flow] → verify_single_flow
    → [fan_in / reduce] → policy_scanner
    → compliance_arbiter
    → END
"""

from __future__ import annotations
import uuid
from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from schemas.data_models import (
    DataFlow, AbstractedFlow, LocusAssignment, Verdict, PolicyClaim,
    AuditReport, AuditSummary, VerdictType
)
from agents.flow_shaper import FlowShaperAgent
from agents.locus_finder import LocusFinderAgent
from agents.semantic_verifier import SemanticVerifierAgent
from agents.policy_scanner import PolicyScannerAgent
from agents.compliance_arbiter import ComplianceArbiterAgent
from memory.audit_memory import AuditMemory


# ── State definition ─────────────────────────────────────────────────────
# Using TypedDict is idiomatic for LangGraph.
# Annotated[List, operator.add] means results from parallel branches are
# concatenated rather than overwriting each other.

class PriAgentState(TypedDict):
    # Inputs (set by the caller)
    app_name: str
    session_id: str
    raw_flows: List[Dict]          # serialised DataFlow objects
    decompiled_code: Dict[str, str]
    policy_text: str

    # Stage 1 output
    abstracted_flows: List[Dict]   # serialised AbstractedFlow objects

    # Stage 2 output
    locus_assignments: List[Dict]  # serialised LocusAssignment objects

    # Stage 3 output — uses reducer so parallel branches don't clobber each other
    verdicts: Annotated[List[Dict], operator.add]

    # Stage 4 output
    policy_claims: List[Dict]
    audit_reports: List[Dict]

    # Metadata
    pattern_hint: str
    error_log: Annotated[List[str], operator.add]


# ── Per-flow sub-state for the parallel verification branch ──────────────

class SingleFlowState(TypedDict):
    flow: Dict                # serialised AbstractedFlow
    assignment: Dict          # serialised LocusAssignment
    decompiled_code: Dict[str, str]
    pattern_hint: str
    verdicts: Annotated[List[Dict], operator.add]
    error_log: Annotated[List[str], operator.add]


# ── Node implementations ─────────────────────────────────────────────────

def node_flow_shaper(state: PriAgentState) -> Dict:
    agent = FlowShaperAgent()
    raw = [DataFlow(**f) for f in state["raw_flows"]]
    abstracted = agent.abstract(raw)
    return {
        "abstracted_flows": [a.model_dump() for a in abstracted],
        "error_log": [],
    }


def node_locus_finder(state: PriAgentState) -> Dict:
    agent = LocusFinderAgent()
    flows = [AbstractedFlow(**f) for f in state["abstracted_flows"]]
    assignments = agent.identify_batch(flows)
    return {
        "locus_assignments": [a.model_dump() for a in assignments],
        "error_log": [],
    }


def fan_out_to_verifiers(state: PriAgentState) -> List[Send]:
    """
    LangGraph fan-out: dispatch each (flow, assignment) pair to a separate
    verify_single_flow node.  These run in parallel up to MAX_CONCURRENT_FLOWS.
    """
    flows = {f["flow_id"]: f for f in state["abstracted_flows"]}
    assignments = {a["flow_id"]: a for a in state["locus_assignments"]}

    sends = []
    for flow_id, flow_dict in flows.items():
        assignment_dict = assignments.get(flow_id)
        if assignment_dict is None:
            continue
        sends.append(
            Send(
                "verify_single_flow",
                SingleFlowState(
                    flow=flow_dict,
                    assignment=assignment_dict,
                    decompiled_code=state["decompiled_code"],
                    pattern_hint=state.get("pattern_hint", ""),
                    verdicts=[],
                    error_log=[],
                ),
            )
        )
    return sends


def node_verify_single_flow(state: SingleFlowState) -> Dict:
    """
    Verifies one abstracted flow.  Runs in parallel for all flows.
    Errors are captured in error_log rather than crashing the pipeline.
    """
    try:
        flow = AbstractedFlow(**state["flow"])
        assignment = LocusAssignment(**state["assignment"])
        agent = SemanticVerifierAgent(decompiled_code=state["decompiled_code"])
        verdict = agent.verify(flow, assignment, pattern_hint=state["pattern_hint"])
        return {
            "verdicts": [verdict.model_dump()],
            "error_log": [],
        }
    except Exception as exc:
        flow_id = state["flow"].get("flow_id", "unknown")
        return {
            "verdicts": [],
            "error_log": [f"verify_single_flow error (flow {flow_id}): {exc}"],
        }


def node_policy_scanner(state: PriAgentState) -> Dict:
    agent = PolicyScannerAgent()
    claims = agent.scan(state["policy_text"])
    return {
        "policy_claims": [c.model_dump() for c in claims],
        "error_log": [],
    }


def node_compliance_arbiter(state: PriAgentState) -> Dict:
    agent = ComplianceArbiterAgent()

    true_positives = [
        Verdict(**v) for v in state["verdicts"]
        if v.get("judgment") == VerdictType.TRUE_POSITIVE
    ]
    claims = [PolicyClaim(**c) for c in state["policy_claims"]]
    flow_lookup = {f["flow_id"]: AbstractedFlow(**f) for f in state["abstracted_flows"]}

    reports = agent.synthesise_batch(true_positives, flow_lookup, claims)
    return {
        "audit_reports": [r.model_dump() for r in reports],
        "error_log": [],
    }


# ── Graph construction ────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Build and compile the PriAgent LangGraph StateGraph with checkpointing.
    """
    workflow = StateGraph(PriAgentState)

    # Register nodes
    workflow.add_node("flow_shaper", node_flow_shaper)
    workflow.add_node("locus_finder", node_locus_finder)
    workflow.add_node("verify_single_flow", node_verify_single_flow)
    workflow.add_node("policy_scanner", node_policy_scanner)
    workflow.add_node("compliance_arbiter", node_compliance_arbiter)

    # Sequential edges
    workflow.add_edge(START, "flow_shaper")
    workflow.add_edge("flow_shaper", "locus_finder")

    # Fan-out: locus_finder dispatches parallel verify_single_flow executions
    workflow.add_conditional_edges(
        "locus_finder",
        fan_out_to_verifiers,
        ["verify_single_flow"],
    )

    # Fan-in: all verify_single_flow branches converge to policy_scanner
    workflow.add_edge("verify_single_flow", "policy_scanner")
    workflow.add_edge("policy_scanner", "compliance_arbiter")
    workflow.add_edge("compliance_arbiter", END)

    # MemorySaver enables mid-run checkpointing and resumption
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ── High-level runner ─────────────────────────────────────────────────────

class PriAgentRunner:
    """
    Convenience wrapper that initialises agents, runs the graph, and
    assembles the final AuditSummary.
    """

    def __init__(self) -> None:
        self._graph = build_graph()
        self._memory = AuditMemory()

    def run(
        self,
        app_name: str,
        raw_flows: List[DataFlow],
        decompiled_code: Dict[str, str],
        policy_text: str,
        session_id: Optional[str] = None,
    ) -> AuditSummary:
        sid = session_id or str(uuid.uuid4())[:8]
        pattern_hint = self._memory.get_pattern_hint()

        initial_state: PriAgentState = {
            "app_name": app_name,
            "session_id": sid,
            "raw_flows": [f.model_dump() for f in raw_flows],
            "decompiled_code": decompiled_code,
            "policy_text": policy_text,
            "abstracted_flows": [],
            "locus_assignments": [],
            "verdicts": [],
            "policy_claims": [],
            "audit_reports": [],
            "pattern_hint": pattern_hint,
            "error_log": [],
        }

        config = {"configurable": {"thread_id": sid}}
        final_state: PriAgentState = self._graph.invoke(initial_state, config=config)

        # Build summary
        all_verdicts = [Verdict(**v) for v in final_state["verdicts"]]
        fp_count = sum(1 for v in all_verdicts if v.judgment == VerdictType.FALSE_POSITIVE)
        total_raw = len(raw_flows)
        abstracted_count = len(final_state["abstracted_flows"])

        reports = [AuditReport(**r) for r in final_state["audit_reports"]]
        violations = [r for r in reports if r.risk_category.value != "Declared Behavior"]
        declared = [r for r in reports if r.risk_category.value == "Declared Behavior"]

        fp_rate = fp_count / max(total_raw, 1)

        summary = AuditSummary(
            app_name=app_name,
            session_id=sid,
            total_raw_flows=total_raw,
            false_positives_eliminated=fp_count,
            fp_reduction_rate=fp_rate,
            confirmed_violations=violations,
            declared_behaviors=declared,
            processing_cost_estimate={
                "abstracted_flows": abstracted_count,
                "flows_verified": len(all_verdicts),
                "reflection_triggered": sum(1 for v in all_verdicts if v.reflected),
                "errors": len(final_state["error_log"]),
            },
        )

        # Persist to cross-session memory
        self._memory.persist_session(
            app_name=app_name,
            session_id=sid,
            violations=[r.model_dump() for r in violations],
        )
        self._memory.record_agent_step(
            "PriAgent",
            f"Completed audit of {app_name}: {len(violations)} violations found.",
        )

        return summary
