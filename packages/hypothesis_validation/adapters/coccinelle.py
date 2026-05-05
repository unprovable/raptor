"""Coccinelle adapter — hypothesis validation via SmPL semantic patches.

Coccinelle is the right tool when the hypothesis is about C-level patterns,
inconsistencies across callers, or control-flow shape (lock balance, NULL
checks, error paths). The LLM-generated rule is written in SmPL.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

from packages import coccinelle as coccinelle_pkg

from .base import ToolAdapter, ToolCapability, ToolEvidence


_SYNTAX_EXAMPLE = """\
// Find malloc() return values used without a NULL check.
@unchecked@
expression E;
position p;
identifier fld;
@@

* E@p = malloc(...);
... when != \\(E == NULL\\|!E\\|IS_ERR(E)\\)
* E->fld

@script:python@
p << unchecked.p;
E << unchecked.E;
@@
import json, sys
for _p in p:
    sys.stderr.write("COCCIRESULT:" + json.dumps({
        "file": _p.file, "line": int(_p.line), "col": int(_p.column),
        "rule": "unchecked",
        "message": "%s used without NULL check" % E,
    }) + "\\n")
"""


class CoccinelleAdapter(ToolAdapter):
    """Adapter wrapping packages/coccinelle/ for hypothesis validation."""

    @property
    def name(self) -> str:
        return "coccinelle"

    def is_available(self) -> bool:
        return coccinelle_pkg.is_available()

    def describe(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            good_for=[
                "Inconsistency detection across callers (e.g. 'find callers that don't check the return of foo')",
                "Lock/unlock symmetry, refcount balance, error-path cleanup",
                "NULL-check enforcement after allocation",
                "Pattern matching with control-flow awareness via the ... operator",
                "C and C++ source",
            ],
            bad_for=[
                "Inter-procedural dataflow tracking — use codeql instead",
                "Path satisfiability / concrete value reasoning — use smt instead",
                "Languages other than C/C++",
                "Pure regex matching with no semantic content — use semgrep",
            ],
            syntax_example=_SYNTAX_EXAMPLE,
            languages=["c", "cpp"],
        )

    def run(
        self,
        rule: str,
        target: Path,
        *,
        timeout: int = 300,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolEvidence:
        if not self.is_available():
            return ToolEvidence(
                tool=self.name, rule=rule, success=False,
                error="spatch is not installed",
            )

        if not rule or not rule.strip():
            return ToolEvidence(
                tool=self.name, rule=rule, success=False,
                error="empty rule",
            )

        # spatch needs the rule as a file. Write to temp then run.
        rule_file: Optional[Path] = None
        try:
            tmp = NamedTemporaryFile(
                prefix="cocci_hv_", suffix=".cocci",
                mode="w", delete=False,
            )
            tmp.write(rule)
            tmp.close()
            rule_file = Path(tmp.name)

            result = coccinelle_pkg.run_rule(
                target=target,
                rule=rule_file,
                timeout=timeout,
                env=env,
            )
        except OSError as e:
            return ToolEvidence(
                tool=self.name, rule=rule, success=False,
                error=f"failed to invoke spatch: {e}",
            )
        finally:
            if rule_file is not None:
                try:
                    rule_file.unlink()
                except OSError:
                    pass

        if not result.ok:
            return ToolEvidence(
                tool=self.name, rule=rule, success=False,
                error="; ".join(result.errors) or f"spatch returned {result.returncode}",
            )

        matches = [m.to_dict() for m in result.matches]
        n = len(matches)
        files = sorted({m["file"] for m in matches if m.get("file")})
        if n:
            summary = f"{n} match{'es' if n != 1 else ''} in {len(files)} file{'s' if len(files) != 1 else ''}"
        else:
            summary = "no matches"

        return ToolEvidence(
            tool=self.name,
            rule=rule,
            success=True,
            matches=matches,
            summary=summary,
        )
