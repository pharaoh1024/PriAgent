"""
CodeSummarizer — helper agent used by SemanticVerifier.

When a method's source code exceeds TAU_SIZE_TOKENS, the summariser produces
a structured, high-fidelity abstraction that preserves:
- Core data-handling logic
- I/O operations
- Control flow predicates (if/else, try/catch)
- Any explicit PII references or API calls

This is NOT a naive truncation — it is a semantics-preserving compression.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import ANTHROPIC_API_KEY, LLM_MODEL

_SYSTEM = """\
You are an expert Android reverse engineer. Produce a STRUCTURED SUMMARY of the
decompiled Java/Smali method below.  Your summary MUST preserve:
- All sensitive API calls (location, contacts, device ID, network I/O, SMS, etc.)
- All conditional branches that gate data access (permission checks, null checks, boolean flags)
- All data transformation operations (hash, encrypt, truncate, anonymise)
- All outbound network calls with their URL/endpoint if determinable
- Parameter names and return type

Format your summary as compact pseudocode with inline comments.
Do NOT include generic boilerplate, imports, or logging.
"""

_HUMAN = """\
Method: {method_signature}
Source Code:
```java
{source_code}
```
Produce the structured summary now.
"""


class CodeSummarizerAgent:
    def __init__(self) -> None:
        llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", _HUMAN),
        ])
        self._chain = prompt | llm | StrOutputParser()

    def summarize(self, method_signature: str, source_code: str) -> str:
        return self._chain.invoke({
            "method_signature": method_signature,
            "source_code": source_code,
        })
