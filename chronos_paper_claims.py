"""chronos_paper_claims.py - rule-based paper claim audit (no LLM).

Reads a plain-text, Markdown, or TeX manuscript and extracts scientific claims
by rules only. It emits a PaperClaim ledger with ClaimRecord-style honesty
fields: supports, does_not_support, evidence_level, next_gate, and
claim_boundary.

Usage:
  python chronos_paper_claims.py Chronos-K1.txt

Outputs:
  artifacts/paper_claims/Chronos-K1.claims.jsonl
  artifacts/paper_claims/Chronos-K1.claims.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


PAPER_CLAIM_TYPES = frozenset(
    {
        "theoretical_claim",
        "conditional_claim",
        "conjecture",
        "definition",
        "empirical_evidence",
        "boundary_note",
    }
)
PAPER_EVIDENCE_LEVELS = frozenset(
    {
        "theoretical_argument",
        "conditional_argument",
        "conjecture",
        "definitional",
        "numerical_experiment",
        "stated_limitation",
    }
)
PAPER_ALLOWED_ACTIONS = frozenset({"record"})
FORBIDDEN_ACTIONS = frozenset({"certified", "proved", "validated", "promote"})

DEFAULT_MANUSCRIPT_PATHS = (
    "Chronos-K1.txt",
    os.path.join("papers", "Chronos-K1.txt"),
    os.path.join("docs", "manuscripts", "Chronos-K1.txt"),
)
OUT_DIR = os.path.join("artifacts", "paper_claims")

_LATEX_ENV_BEGIN = re.compile(
    r"\\begin\{(theorem|lemma|proposition|corollary|definition|conjecture)\}(?:\[([^\]]*)\])?",
    re.I,
)
_THEORETICAL_CUE = re.compile(r"\b(theorem|lemma|proposition|corollary)\b", re.I)
_DEFINITION_CUE = re.compile(r"\bdefinition\b", re.I)
_CONJECTURE_CUE = re.compile(r"\bconjecture\b", re.I)
_EMPIRICAL_CUE = re.compile(
    r"\b(numerical(ly)?|simulation(s)?|simulated|measured|empirical(ly)?|experiment(s|al)?|"
    r"verified|confirmed|reproduce(s|d)?)\b",
    re.I,
)
_CONDITIONAL_CUE = re.compile(
    r"(conditionally\s+recover|conditional(ly)?|given (the )?(two )?external input|"
    r"requires? (the )?(two )?external input|under the (field-level )?ansatz|"
    r"field-level ansatz|candidate ansatz|\bansatz\b)",
    re.I,
)
_BOUNDARY_CUE = re.compile(
    r"(not derived|not a derivation|requires? external input|left open|open problem|"
    r"left as an open problem|left to future work|we do not claim|does not imply|"
    r"do not imply|not establish|not validated|not certified|not a claim|limitation(s)?|"
    r"caveat|beyond the scope|not a full reconstruction|structural rather than fully reconstructive|"
    r"neither perspective derives|not derived from|future work)",
    re.I,
)
_OPEN_PROBLEM_CUE = re.compile(r"(open problem|left open|left as an open problem|left to future work)", re.I)
_STRUCTURAL_LATEX = re.compile(
    r"\\(documentclass|usepackage|setlength|usetikzlibrary|hypersetup|newtheorem|newcommand|"
    r"renewcommand|title|author|date|maketitle|section|subsection|subsubsection|label|ref|cite)"
)

_DEFAULT_DOES_NOT_SUPPORT = [
    "proof verification by this audit",
    "peer-review replacement",
    "experimental validation",
    "universal physics AI",
]
_DEFAULT_BOUNDARY = "rule-based paper audit; valid only as extraction under manuscript wording"

_SAMPLE = """# Chronos-K1 sample
Theorem 1. K=1 induces a Lorentzian signature under the stated assumptions.
Definition. A leaf is an invariant set of the null flow.
We conjecture that the same structure extends to a larger class.
Numerically verified: the recovery time scales with slope -2.000.
The thermodynamic perspective conditionally recovers Einstein equations given two external inputs.
The field-level ansatz is motivated by but not derived from the point-level condition.
This identification is left as an open problem.
"""


@dataclass
class PaperClaim:
    claim_id: str
    source_file: str
    line_start: int
    line_end: int
    claim_text_full: str
    claim_text_preview: str
    claim_type: str
    evidence_level: str
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    next_gate: str = ""
    claim_boundary: str = ""
    allowed_action: str = "record"

    def __post_init__(self) -> None:
        if self.claim_type not in PAPER_CLAIM_TYPES:
            raise ValueError(f"unknown claim_type {self.claim_type!r}")
        if self.evidence_level not in PAPER_EVIDENCE_LEVELS:
            raise ValueError(f"unknown evidence_level {self.evidence_level!r}")
        if self.allowed_action in FORBIDDEN_ACTIONS or self.allowed_action not in PAPER_ALLOWED_ACTIONS:
            raise ValueError(f"allowed_action {self.allowed_action!r} is forbidden")
        if self.evidence_level == "conjecture" and self.claim_type == "theoretical_claim":
            raise ValueError("a conjecture must not be emitted as theorem/proof")
        if self.evidence_level == "numerical_experiment" and self.claim_type == "theoretical_claim":
            raise ValueError("numerical evidence must not be emitted as proof")
        if self.claim_type == "conditional_claim" and self.evidence_level != "conditional_argument":
            raise ValueError("conditional claims must use conditional_argument evidence")
        if self.claim_type == "conditional_claim":
            joined = " ".join(self.does_not_support).lower()
            if not any(cue in joined for cue in ("conditional", "external input", "ansatz", "not derived")):
                raise ValueError("conditional claims must record unproven input boundaries")
        if self.claim_type in ("theoretical_claim", "conditional_claim"):
            supports_joined = " ".join(self.supports).lower()
            if "universal physics ai" in supports_joined:
                raise ValueError("paper claims must not support universal physics AI")
        if not self.does_not_support:
            raise ValueError("every paper claim must have does_not_support")
        if not self.claim_boundary.strip():
            raise ValueError("every paper claim must have claim_boundary")
        if not self.next_gate.strip():
            raise ValueError("every paper claim must have next_gate")

    def to_dict(self) -> dict:
        return asdict(self)


def _slug(text: str, maxlen: int = 54) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug[:maxlen] or "claim").strip("_")


def _preview(text: str, maxlen: int = 240) -> str:
    return text if len(text) <= maxlen else text[: maxlen - 3].rstrip() + "..."


def _strip_tex(text: str) -> str:
    text = text.replace("~", " ")
    text = re.sub(r"\\(textit|emph|textbf)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\[a-zA-Z]+", lambda m: m.group(0).replace("\\", ""), text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _classify_line(line: str) -> Optional[tuple[str, str]]:
    env = _LATEX_ENV_BEGIN.search(line)
    if _BOUNDARY_CUE.search(line):
        return "boundary_note", "stated_limitation"
    if _CONDITIONAL_CUE.search(line):
        return "conditional_claim", "conditional_argument"
    if env:
        name = env.group(1).lower()
        if name == "definition":
            return "definition", "definitional"
        if name == "conjecture":
            return "conjecture", "conjecture"
        return "theoretical_claim", "theoretical_argument"
    if _CONJECTURE_CUE.search(line):
        return "conjecture", "conjecture"
    if _DEFINITION_CUE.search(line):
        return "definition", "definitional"
    if _EMPIRICAL_CUE.search(line):
        return "empirical_evidence", "numerical_experiment"
    if _THEORETICAL_CUE.search(line):
        return "theoretical_claim", "theoretical_argument"
    return None


def _supports_for(claim_type: str) -> list[str]:
    if claim_type == "theoretical_claim":
        return ["paper states a theoretical result under manuscript assumptions"]
    if claim_type == "conditional_claim":
        return ["paper states a conditional result given explicit inputs or ansatz"]
    if claim_type == "conjecture":
        return ["paper states a conjecture without treating it as proof"]
    if claim_type == "definition":
        return ["paper introduces terminology or a formal definition"]
    if claim_type == "empirical_evidence":
        return ["paper reports numerical, simulation, or experimental evidence"]
    return ["paper records a limitation, scope boundary, or non-claim"]


def _does_not_support_for(claim_type: str, text: str) -> list[str]:
    out = list(_DEFAULT_DOES_NOT_SUPPORT)
    if claim_type == "theoretical_claim":
        out.append("machine-checked proof")
    elif claim_type == "conditional_claim":
        out.append("unconditional result; depends on stated external input or ansatz not derived here")
    elif claim_type == "conjecture":
        out.append("proved theorem")
    elif claim_type == "definition":
        out.append("empirical or theoretical validation by itself")
    elif claim_type == "empirical_evidence":
        out.append("formal proof or certified mechanism")
    else:
        out.append("positive scientific claim beyond the recorded boundary")
    if _BOUNDARY_CUE.search(text):
        out.append("claim limited by explicit manuscript boundary wording")
    return out


def _next_gate_for(claim_type: str, text: str) -> str:
    if _OPEN_PROBLEM_CUE.search(text):
        return "resolve the explicitly stated open problem"
    if claim_type == "theoretical_claim":
        return "verify proof details and connect to reproducible checks"
    if claim_type == "conditional_claim":
        return "discharge or independently justify the external inputs or ansatz"
    if claim_type == "conjecture":
        return "attempt proof, counterexample, or scoped evidence"
    if claim_type == "definition":
        return "use the definition in downstream claims"
    if claim_type == "empirical_evidence":
        return "replicate evidence and test robustness against confounds"
    return "record boundary and prevent overclaiming"


def _boundary_for(claim_type: str) -> str:
    return f"{_DEFAULT_BOUNDARY}; extracted as {claim_type}, not independently verified"


def _should_skip(line: str) -> bool:
    if not line.strip():
        return True
    if line.strip().startswith("%"):
        return True
    if line.strip().startswith("```"):
        return True
    if _LATEX_ENV_BEGIN.search(line):
        return False
    if _STRUCTURAL_LATEX.match(line.strip()):
        return True
    return False


def find_manuscript(cli_path: Optional[str] = None) -> Optional[str]:
    candidates: list[str] = []
    if cli_path:
        candidates.append(cli_path)
    candidates.extend(DEFAULT_MANUSCRIPT_PATHS)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.readlines()


def extract_claims(path: str) -> list[PaperClaim]:
    lines = _load_lines(path)
    source_file = os.path.basename(path)
    claims: list[PaperClaim] = []
    seen_ids: set[str] = set()

    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if _should_skip(line):
            continue
        classification = _classify_line(line)
        if classification is None:
            continue
        claim_type, evidence_level = classification
        text = _strip_tex(line)
        if not text:
            continue

        claim_id_base = f"paper_{idx:04d}_{_slug(text)}"
        claim_id = claim_id_base
        suffix = 2
        while claim_id in seen_ids:
            claim_id = f"{claim_id_base}_{suffix}"
            suffix += 1
        seen_ids.add(claim_id)

        claims.append(
            PaperClaim(
                claim_id=claim_id,
                source_file=source_file,
                line_start=idx,
                line_end=idx,
                claim_text_full=text,
                claim_text_preview=_preview(text),
                claim_type=claim_type,
                evidence_level=evidence_level,
                supports=_supports_for(claim_type),
                does_not_support=_does_not_support_for(claim_type, text),
                next_gate=_next_gate_for(claim_type, text),
                claim_boundary=_boundary_for(claim_type),
                allowed_action="record",
            )
        )
    return claims


def _count_by(claims: list[PaperClaim], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        key = getattr(claim, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_claims(claims: list[PaperClaim]) -> dict:
    return {
        "count_total": len(claims),
        "by_claim_type": _count_by(claims, "claim_type"),
        "by_evidence_level": _count_by(claims, "evidence_level"),
    }


def output_paths(source_path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return (
        os.path.join(OUT_DIR, f"{stem}.claims.jsonl"),
        os.path.join(OUT_DIR, f"{stem}.claims.md"),
    )


def write_jsonl(claims: list[PaperClaim], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
    return path


def write_markdown(claims: list[PaperClaim], summary: dict, source_path: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        f"# Paper Claim Audit: {os.path.basename(source_path)}",
        "",
        "Rule-based extraction, no LLM, no external API.",
        "",
        "This audit does not verify proofs or replace peer review. It separates theoretical claims,",
        "definitions, conditional claims, empirical evidence, and boundary notes while recording",
        "`supports`, `does_not_support`, `evidence_level`, `next_gate`, and `claim_boundary`.",
        "",
        f"- total claims: {summary['count_total']}",
        f"- by claim_type: {summary['by_claim_type']}",
        f"- by evidence_level: {summary['by_evidence_level']}",
        "",
    ]
    for claim in claims:
        lines.extend(
            [
                f"## {claim.claim_id}",
                "",
                f"- source: `{claim.source_file}:{claim.line_start}`",
                f"- claim_type: `{claim.claim_type}`",
                f"- evidence_level: `{claim.evidence_level}`",
                f"- allowed_action: `{claim.allowed_action}`",
                f"- claim_text_preview: {claim.claim_text_preview}",
                f"- supports: {'; '.join(claim.supports)}",
                f"- does_not_support: {'; '.join(claim.does_not_support)}",
                f"- next_gate: {claim.next_gate}",
                f"- claim_boundary: {claim.claim_boundary}",
                "",
            ]
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def run_anti_cheat_tests() -> int:
    checks = 0

    def expect_error(fn) -> None:
        nonlocal checks
        try:
            fn()
        except ValueError:
            checks += 1
            return
        raise AssertionError("expected anti-cheat ValueError")

    base = {
        "claim_id": "x",
        "source_file": "paper.txt",
        "line_start": 1,
        "line_end": 1,
        "claim_text_full": "claim",
        "claim_text_preview": "claim",
        "supports": ["paper states a scoped claim"],
        "does_not_support": list(_DEFAULT_DOES_NOT_SUPPORT),
        "next_gate": "next",
        "claim_boundary": _DEFAULT_BOUNDARY,
    }

    expect_error(lambda: PaperClaim(**base, claim_type="theoretical_claim", evidence_level="conjecture"))
    expect_error(lambda: PaperClaim(**base, claim_type="theoretical_claim", evidence_level="numerical_experiment"))
    expect_error(
        lambda: PaperClaim(
            **base,
            claim_type="conditional_claim",
            evidence_level="conditional_argument",
            allowed_action="certified",
        )
    )
    expect_error(
        lambda: PaperClaim(
            **{**base, "does_not_support": list(_DEFAULT_DOES_NOT_SUPPORT)},
            claim_type="conditional_claim",
            evidence_level="conditional_argument",
        )
    )
    expect_error(lambda: PaperClaim(**{**base, "does_not_support": []}, claim_type="definition", evidence_level="definitional"))
    expect_error(lambda: PaperClaim(**{**base, "claim_boundary": ""}, claim_type="definition", evidence_level="definitional"))
    expect_error(lambda: PaperClaim(**{**base, "next_gate": ""}, claim_type="definition", evidence_level="definitional"))
    for action in ("proved", "validated", "promote"):
        expect_error(lambda action=action: PaperClaim(**base, claim_type="definition", evidence_level="definitional", allowed_action=action))

    assert _classify_line("Conjecture. This should hold.") == ("conjecture", "conjecture")
    checks += 1
    assert _classify_line("Numerically verified by simulation.") == ("empirical_evidence", "numerical_experiment")
    checks += 1
    assert _classify_line("The field-level ansatz conditionally recovers the result.") == (
        "conditional_claim",
        "conditional_argument",
    )
    checks += 1
    assert _classify_line("This is not derived and is left open.") == ("boundary_note", "stated_limitation")
    checks += 1

    valid = PaperClaim(
        **{
            **base,
            "claim_type": "conditional_claim",
            "evidence_level": "conditional_argument",
            "does_not_support": list(_DEFAULT_DOES_NOT_SUPPORT) + ["ansatz not derived"],
        }
    )
    assert valid.allowed_action == "record"
    checks += 1
    return checks


def _write_sample() -> str:
    path = "Chronos-K1.sample.txt"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_SAMPLE)
    return path


def main(argv: list[str]) -> int:
    cli_path = argv[1] if len(argv) > 1 else None
    anti_cheat_count = run_anti_cheat_tests()
    path = find_manuscript(cli_path)
    if path is None:
        path = _write_sample()
        print(f"using built-in sample: {path}")
    else:
        print(f"using manuscript: {path}")

    claims = extract_claims(path)
    summary = summarize_claims(claims)
    jsonl_path, md_path = output_paths(path)
    write_jsonl(claims, jsonl_path)
    write_markdown(claims, summary, path, md_path)

    print(f"extracted {summary['count_total']} paper claims from {os.path.basename(path)}")
    print(f"wrote {jsonl_path}")
    print(f"wrote {md_path}")
    print(f"by_claim_type: {summary['by_claim_type']}")
    print(f"by_evidence_level: {summary['by_evidence_level']}")
    print(f"anti-cheat assertions passed ({anti_cheat_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
