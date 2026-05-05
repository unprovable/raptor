"""Hypothesis-driven, tool-grounded vulnerability validation.

The LLM forms hypotheses about security weaknesses ("input X flows unchecked
to sink Y"); deterministic tools (Semgrep, Coccinelle, CodeQL, SMT) test
those hypotheses; the LLM never directly classifies code as vulnerable.

Research basis: KNighter (SOSP 2025, 92 kernel bugs), SAILOR
(arXiv:2604.06506, 379 vulns vs 12 pure-agentic), IRIS (ICLR 2025, 2x
CodeQL recall). Pure self-critique without tool grounding actively
degrades quality (IEEE-ISTAS 2025: 37.6% more critical vulns after 5
iterations) — this package exists to ground LLM reasoning in mechanical
evidence.

Public API:
    from packages.hypothesis_validation import (
        Hypothesis, ValidationResult, ToolAdapter, ToolCapability,
    )
    from packages.hypothesis_validation.adapters import (
        CoccinelleAdapter, SemgrepAdapter,
    )
    from packages.hypothesis_validation.runner import validate

    h = Hypothesis(
        claim="parse_input return value used as array index without check",
        target=Path("src/parser.c"),
        target_function="dispatch",
        cwe="CWE-129",
    )
    result = validate(h, [CoccinelleAdapter(), SemgrepAdapter()], llm_client)
    if result.verdict == "confirmed":
        for ev in result.evidence:
            print(f"{ev.tool}: {ev.summary}")
"""

from .hypothesis import Hypothesis
from .result import Evidence, ValidationResult
from .adapters.base import ToolAdapter, ToolCapability, ToolInvocation, ToolEvidence

__all__ = [
    "Hypothesis",
    "ValidationResult",
    "Evidence",
    "ToolAdapter",
    "ToolCapability",
    "ToolInvocation",
    "ToolEvidence",
]
