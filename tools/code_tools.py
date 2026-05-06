"""
LangChain tools used by the SemanticVerifier agent.

Defining tools with the @tool decorator enables function-calling / tool-use
— one of the explicit technical requirements in the job description.

The SemanticVerifier receives these tools and decides autonomously when to
invoke them, mimicking the behaviour of an expert analyst who looks up code
on demand rather than scanning everything upfront.
"""

from __future__ import annotations
import re
from typing import Dict

import tiktoken
from langchain_core.tools import tool

from config import TAU_SIZE_TOKENS


# ── code retrieval ───────────────────────────────────────────────────────

_DECOMPILED_CODE_REGISTRY: Dict[str, str] = {}


def register_decompiled_code(code_map: Dict[str, str]) -> None:
    """Called by the orchestrator before a verification run."""
    _DECOMPILED_CODE_REGISTRY.clear()
    _DECOMPILED_CODE_REGISTRY.update(code_map)


@tool
def get_method_source(method_signature: str) -> str:
    """
    Retrieve the decompiled source code for a given Android method signature.
    Use this tool when you need to inspect the implementation of a specific
    method in the data flow call chain.

    Args:
        method_signature: Fully qualified method signature, e.g.
            "com.example.app.network.NetworkHelper.sendData(String, String)"

    Returns:
        The decompiled source code, or a NOT_FOUND message if unavailable.
    """
    # Exact match first
    if method_signature in _DECOMPILED_CODE_REGISTRY:
        return _DECOMPILED_CODE_REGISTRY[method_signature]

    # Fuzzy: match on the simple method name portion (after last dot/paren)
    simple_name = re.split(r"[.(]", method_signature)[-2] if "." in method_signature else method_signature
    for sig, code in _DECOMPILED_CODE_REGISTRY.items():
        if simple_name in sig:
            return code

    return f"[NOT FOUND] Source code for '{method_signature}' is unavailable in the decompiled corpus."


@tool
def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using the cl100k_base tokenizer.
    Useful for deciding whether a code block needs summarisation before analysis.

    Args:
        text: The text to measure.

    Returns:
        Integer token count.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


@tool
def check_needs_summarization(method_signature: str) -> str:
    """
    Check whether the source code of a method exceeds the summarisation
    threshold (TAU_SIZE_TOKENS).  Returns 'SUMMARIZE' or 'FULL'.

    Args:
        method_signature: Method to check.

    Returns:
        'SUMMARIZE' if the code exceeds the token threshold, else 'FULL'.
    """
    code = get_method_source.invoke(method_signature)  # type: ignore[attr-defined]
    if code.startswith("[NOT FOUND]"):
        return "NOT_FOUND"
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(code))
    return "SUMMARIZE" if tokens > TAU_SIZE_TOKENS else "FULL"


@tool
def extract_permission_checks(method_signature: str) -> str:
    """
    Scan the source code of a method for Android permission check patterns
    (e.g., checkSelfPermission, enforceCallingPermission).  Returns a
    summary of any permission gates found.

    Args:
        method_signature: The method to inspect.

    Returns:
        A plain-text summary of permission checks found, or 'NONE_FOUND'.
    """
    code = get_method_source.invoke(method_signature)  # type: ignore[attr-defined]
    if code.startswith("[NOT FOUND]"):
        return "NONE_FOUND — source unavailable."

    patterns = [
        r"checkSelfPermission",
        r"checkCallingPermission",
        r"enforceCallingOrSelfPermission",
        r"requestPermissions",
        r"shouldShowRequestPermissionRationale",
        r"PackageManager\.PERMISSION_GRANTED",
    ]
    found = []
    for pat in patterns:
        if re.search(pat, code):
            found.append(pat.replace(r"\.", "."))

    if not found:
        return "NONE_FOUND — no Android permission checks detected in this method."
    return "Permission checks found: " + ", ".join(found)


# Exported tool list for agent binding
SEMANTIC_VERIFIER_TOOLS = [
    get_method_source,
    count_tokens,
    check_needs_summarization,
    extract_permission_checks,
]
