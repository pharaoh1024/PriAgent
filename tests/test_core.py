"""
Quick smoke-test: verifies the project imports and core logic work correctly
without making any LLM API calls.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.data_models import (
    DataFlow, AbstractedFlow, Verdict, VerdictType, AuditReport, RiskCategory
)
from agents.flow_shaper import FlowShaperAgent, _detect_and_collapse_cycles, _extract_package
from agents.locus_finder import _dynamic_tau


# ── test cycle detection ─────────────────────────────────────────────────

def test_cycle_detection():
    chain = ["A()", "B()", "C()", "B()", "C()", "D()"]
    text, has_cycle, notation = _detect_and_collapse_cycles(chain)
    assert has_cycle, "Should detect a cycle"
    assert "CYCLE" in text
    print("PASS  test_cycle_detection")


def test_no_cycle():
    chain = ["A()", "B()", "C()", "D()"]
    text, has_cycle, notation = _detect_and_collapse_cycles(chain)
    assert not has_cycle
    assert text == "A() -> B() -> C() -> D()"
    print("PASS  test_no_cycle")


def test_package_extraction():
    sig = "com.example.app.network.NetworkHelper.sendData(String)"
    pkg = _extract_package(sig)
    assert pkg == "com.example.app.network", f"Got: {pkg}"
    print("PASS  test_package_extraction")


def test_dynamic_tau():
    assert _dynamic_tau(0) == 3    # clamp to min
    assert _dynamic_tau(25) == 5   # 25 // 5
    assert _dynamic_tau(100) == 8  # clamp to max
    print("PASS  test_dynamic_tau")


def test_package_aggregation_no_llm():
    flows = [
        DataFlow(
            flow_id=f"f{i}",
            source="android.telephony.TelephonyManager.getDeviceId()",
            sink="okhttp3.OkHttpClient.newCall()",
            call_chain=["A()", "B()"],
            app_package="com.example.app",
        )
        for i in range(5)
    ]
    agent = FlowShaperAgent()
    groups = agent._aggregate_by_package(flows)
    # All 5 flows have the same src/snk package → 1 group
    assert len(groups) == 1
    key = list(groups.keys())[0]
    assert len(groups[key]) == 5
    print("PASS  test_package_aggregation_no_llm")


def test_data_model_serialisation():
    flow = DataFlow(
        flow_id="test",
        source="src.Method()",
        sink="snk.Method()",
        call_chain=["A()", "B()"],
    )
    d = flow.model_dump()
    restored = DataFlow(**d)
    assert restored.flow_id == "test"
    print("PASS  test_data_model_serialisation")


if __name__ == "__main__":
    test_cycle_detection()
    test_no_cycle()
    test_package_extraction()
    test_dynamic_tau()
    test_package_aggregation_no_llm()
    test_data_model_serialisation()
    print("\nAll unit tests passed.")
