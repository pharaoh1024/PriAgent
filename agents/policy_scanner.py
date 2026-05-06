"""
PolicyScanner Agent — Stage 4a: Privacy Policy Parsing.

Parses unstructured privacy policy text and extracts structured claims about
data collection, usage, and sharing for each sensitive data type.
"""

from __future__ import annotations
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from schemas.data_models import PolicyClaim
from config import ANTHROPIC_API_KEY, LLM_MODEL


class _PolicyScanOutput(BaseModel):
    claims: List[PolicyClaim] = Field(
        description="Structured list of data collection/usage/sharing claims found in the policy"
    )


_SYSTEM = """\
You are a legal-technical analyst specialising in mobile app privacy policies.
Parse the provided privacy policy text and extract structured claims for each
data type that is mentioned (e.g., device identifier, location, contacts, microphone, etc.).

For each data type, determine:
- Whether collection is explicitly stated.
- The stated purpose of use (if given).
- Whether third-party sharing is mentioned.
- Whether user control (opt-out, consent) is mentioned.
- The most relevant verbatim excerpt (≤ 2 sentences).

Return a JSON array of claims.
"""

_HUMAN = """\
Privacy Policy Text:
\"\"\"
{policy_text}
\"\"\"

Extract all data claims. Return JSON only.
"""


class PolicyScannerAgent:
    def __init__(self) -> None:
        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        self._structured_llm = llm.with_structured_output(_PolicyScanOutput)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self._chain = self._prompt | self._structured_llm

    def scan(self, policy_text: str) -> List[PolicyClaim]:
        result: _PolicyScanOutput = self._chain.invoke({"policy_text": policy_text})
        return result.claims
