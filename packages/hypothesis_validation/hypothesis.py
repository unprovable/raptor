"""Hypothesis dataclass — what the LLM thinks might be wrong."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Hypothesis:
    """A claim about a potential vulnerability that mechanical tools can test.

    The LLM produces hypotheses by reasoning about a function's assumptions
    ("this trusts X to be NULL-checked, what if it isn't?"). The runner
    then asks the LLM to translate the hypothesis into a tool invocation.

    Attributes:
        claim: Free-text description of the suspected weakness. Should be
            specific enough that a tool rule can be generated from it.
        target: File or directory to test against.
        target_function: Optional specific function within the target.
            Empty string when the hypothesis applies to the whole file.
        cwe: Optional CWE-NNN tag. Used for selecting exemplars and
            for routing the prompt template.
        suggested_tools: Optional ordered list of adapter names the LLM
            proposed for testing. The runner can override by selecting
            a different adapter from those available.
        context: Optional additional context to inject into the prompt
            (callers, callees, related annotations).
    """

    claim: str
    target: Path
    target_function: str = ""
    cwe: str = ""
    suggested_tools: List[str] = field(default_factory=list)
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "target": str(self.target),
            "target_function": self.target_function,
            "cwe": self.cwe,
            "suggested_tools": list(self.suggested_tools),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        if not d or not isinstance(d, dict):
            return cls(claim="", target=Path("."))
        return cls(
            claim=d.get("claim", ""),
            target=Path(d.get("target", ".")),
            target_function=d.get("target_function", ""),
            cwe=d.get("cwe", ""),
            suggested_tools=list(d.get("suggested_tools") or []),
            context=d.get("context", ""),
        )
