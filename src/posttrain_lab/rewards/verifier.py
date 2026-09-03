"""Canonical exact/symbolic verifier for mathematical model completions.

The extraction policy is deliberately narrower than Math-Verify's default
best-effort prediction parser.  A prediction must expose a terminal answer via
``\\boxed{...}``, ``<answer>...</answer>``, an explicit answer label, or be a
single direct mathematical surface.  The last explicit surface wins, including
when it is malformed, so an earlier correct answer cannot rescue a later wrong
or broken final answer.

Math-Verify remains the pinned symbolic backend.  This module owns the stable
project contract around it: role-specific parsing, bounded inputs, structured
failure reasons, strict reference handling, and exact binary rewards.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from importlib.metadata import version
from typing import Any

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    LatexNormalizationConfig,
)
from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_compare
from math_verify.errors import TimeoutException
from sympy import FiniteSet, Interval, Symbol, Tuple
from sympy.core.relational import Equality, Relational
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrixbase import MatrixBase
from sympy.sets.sets import Set

_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BOX_COMMAND = re.compile(r"\\(?:boxed|fbox)(?![A-Za-z])")
_ANSWER_TAG_OPEN = re.compile(r"<answer\s*>", re.IGNORECASE)
_ANSWER_TAG_CLOSE = re.compile(r"</answer\s*>", re.IGNORECASE)
_TEXT_ANSWER_MARKER = re.compile(
    r"(?<![\w\\])(?:final\s+answer|answer|最终答案|答案)\s*"
    r"(?:is|equals|是|=|:|\uff1a)\s*",
    re.IGNORECASE,
)
_NEGATED_ANSWER_PREFIX = re.compile(
    r"(?:\b(?:not|no|never|wrong|incorrect|false|fake|example|previous|old|rejected|"
    r"candidate|tentative|possible|possibly|discarded|ignore|ignored)\b|n't\b|"
    r"\bnon[\s-]*(?:final\b|$)|(?:不是|不要|错误|不正确|虚假|示例|之前|旧|拒绝|候选|可能|忽略))",
    re.IGNORECASE,
)
_BACKTICK_RUN = re.compile(r"`+")
_MARKDOWN_FENCE_LINE = re.compile(r"(?m)^[ ]{0,3}(`{3,}|~{3,})([^\n]*)$")
_ASCII_WORD = re.compile(r"[A-Za-z]+")
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_TEXT_LIKE_COMMAND = re.compile(
    r"\\(?:text(?:normal|bf|it|rm)?|math(?:rm|it|bf)|mbox)(?![A-Za-z])",
    re.IGNORECASE,
)
_TWO_ARGUMENT_PRIMARY_COMMAND = re.compile(
    r"\\(?:[dtc]?frac|binom)(?![A-Za-z])",
    re.IGNORECASE,
)
_SQRT_COMMAND = re.compile(r"\\sqrt(?![A-Za-z])", re.IGNORECASE)
_BEGIN_COMMAND = re.compile(r"\\begin(?![A-Za-z])", re.IGNORECASE)
_FUNCTION_PRIMARY_COMMAND = re.compile(
    r"\\(?:max|min|sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|log|ln|exp|gcd|lcm|floor|"
    r"ceil|abs|det|lim|sup|inf|argmax|argmin|operatorname)(?![A-Za-z])",
    re.IGNORECASE,
)
_JUXTAPOSITION_PRIMARY_COMMAND = re.compile(
    r"\\(?:[dtc]?frac|binom|sqrt|m?left|max|min|sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|"
    r"log|ln|exp|gcd|lcm|floor|ceil|abs|det|lim|sup|inf|argmax|argmin|operatorname)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_IGNORABLE_MATH_SPACING = re.compile(r"(?:\s+|~|\\[,;:!]|\\\s+)")
_DISALLOWED_CANDIDATE_COMMAND = re.compile(
    r"\\(?:because|therefore|since|hence|quad|qquad|enspace|enskip|thinspace|medspace|"
    r"thickspace|negthinspace|negmedspace|negthickspace|hspace|vspace|hskip|vskip|kern|"
    r"mkern|hfil|hfill|newline|"
    r"linebreak|phantom|href|url|include|input|write|read|land|wedge|lor|vee|implies|"
    r"iff|Longrightarrow|Longleftrightarrow|displaystyle|ldots)(?![A-Za-z])",
    re.IGNORECASE,
)
_DISALLOWED_REMOVED_CHARACTER = re.compile(r"[\"']|\\\$")
_LATEX_ROW_BREAK = re.compile(r"\\\\(?![A-Za-z])")
_LOGICAL_CONNECTOR_CHARACTER = re.compile("[&\N{LOGICAL AND}\N{LOGICAL OR}]")
_EMPTY_OR_SPACING_LATEX = re.compile(
    r"\\(?:[,;:!])|\\(?:text|mathrm)\s*\{\s*\}|\\\s+|~|\{\s*\}",
    re.IGNORECASE,
)
_ADJACENT_NUMBERS_WITHOUT_OPERATOR = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)\s+[+-]?(?:\d|\.\d)"
)
_NUMERIC_SUBSCRIPT = re.compile(r"(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)\s*_")
_TRANSPOSE_SUPERSCRIPT = re.compile(
    r"\^\s*(?:T(?![A-Za-z])|\\top(?![A-Za-z])|\{\s*(?:T|\\top)\s*\})"
)
_DISALLOWED_CANDIDATE_WORDS = frozenset(
    {
        "actually",
        "and",
        "answer",
        "because",
        "but",
        "candidate",
        "conclusion",
        "else",
        "example",
        "fake",
        "final",
        "false",
        "hence",
        "however",
        "ignore",
        "incorrect",
        "instead",
        "instruction",
        "maybe",
        "never",
        "no",
        "not",
        "old",
        "or",
        "otherwise",
        "possible",
        "possibly",
        "previous",
        "rejected",
        "return",
        "since",
        "system",
        "then",
        "tentative",
        "therefore",
        "thus",
        "wrong",
    }
)
_DIRECT_ALLOWED_WORDS = frozenset(
    {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "abs",
        "cos",
        "exp",
        "inf",
        "infty",
        "ln",
        "log",
        "max",
        "min",
        "mod",
        "pi",
        "sin",
        "sqrt",
        "tan",
        # Unit words intentionally stay narrow. Narrative connectors such as
        # "because", "then", "and", and "or" remain invalid.
        "cent",
        "cents",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
        "cm",
        "day",
        "days",
        "degree",
        "degrees",
        "dollar",
        "dollars",
        "feet",
        "foot",
        "gram",
        "grams",
        "hour",
        "hours",
        "inch",
        "inches",
        "kg",
        "kilogram",
        "kilograms",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
        "km",
        "liter",
        "liters",
        "litre",
        "litres",
        "meter",
        "meters",
        "metre",
        "metres",
        "mg",
        "mile",
        "miles",
        "milligram",
        "milligrams",
        "millimeter",
        "millimeters",
        "millimetre",
        "millimetres",
        "minute",
        "minutes",
        "ml",
        "mm",
        "radian",
        "radians",
        "second",
        "seconds",
        "square",
        "unit",
        "units",
        "yard",
        "yards",
    }
)
_ALLOWED_UNIT_WORDS = frozenset(
    {
        "cent",
        "cents",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
        "cm",
        "day",
        "days",
        "degree",
        "degrees",
        "dollar",
        "dollars",
        "feet",
        "foot",
        "gram",
        "grams",
        "hour",
        "hours",
        "inch",
        "inches",
        "kg",
        "kilogram",
        "kilograms",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
        "km",
        "liter",
        "liters",
        "litre",
        "litres",
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
        "mg",
        "mile",
        "miles",
        "milligram",
        "milligrams",
        "millimeter",
        "millimeters",
        "millimetre",
        "millimetres",
        "minute",
        "minutes",
        "ml",
        "mm",
        "radian",
        "radians",
        "second",
        "seconds",
        "square",
        "unit",
        "units",
        "yard",
        "yards",
    }
)
_MATH_ENVIRONMENT_PAIRS = (
    ("$$", "$$"),
    ("$", "$"),
    (r"\(", r"\)"),
    (r"\[", r"\]"),
)


class ExtractionStatus(StrEnum):
    """Outcome of locating one canonical answer surface."""

    EXTRACTED = "extracted"
    EMPTY_INPUT = "empty_input"
    INPUT_TOO_LONG = "input_too_long"
    UNSAFE_CONTROL_CHARACTER = "unsafe_control_character"
    NO_FINAL_ANSWER = "no_final_answer"
    MALFORMED_FINAL_ANSWER = "malformed_final_answer"
    CANDIDATE_TOO_LONG = "candidate_too_long"
    NOT_EVALUATED = "not_evaluated"


class AnswerSource(StrEnum):
    """Surface that supplied the candidate answer."""

    REFERENCE = "reference"
    BOXED = "boxed"
    ANSWER_TAG = "answer_tag"
    TEXT_MARKER = "text_marker"
    DIRECT = "direct"


class ParseStatus(StrEnum):
    """Outcome of converting an extracted surface into symbolic objects."""

    PARSED = "parsed"
    NOT_EXTRACTED = "not_extracted"
    UNPARSEABLE = "unparseable"
    TIMEOUT = "timeout"
    BACKEND_ERROR = "backend_error"
    SKIPPED = "skipped"


class VerificationStatus(StrEnum):
    """Canonical reason attached to one verification decision."""

    MATCH = "match"
    MISMATCH = "mismatch"
    PREDICTION_NOT_EXTRACTED = "prediction_not_extracted"
    PREDICTION_UNPARSEABLE = "prediction_unparseable"
    PREDICTION_TIMEOUT = "prediction_timeout"
    REFERENCE_INVALID = "reference_invalid"
    VERIFICATION_TIMEOUT = "verification_timeout"
    BACKEND_ERROR = "backend_error"


class VerifierInfrastructureError(RuntimeError):
    """Raised when an exact reward cannot be safely assigned."""


@dataclass(frozen=True, slots=True)
class VerifierPolicy:
    """Frozen parsing and comparison limits for D05."""

    schema_version: str = "d05-verifier-v1"
    max_input_chars: int = 32_768
    max_candidate_chars: int = 4_096
    direct_answer_max_chars: int = 512
    parsing_timeout_seconds: int = 5
    verification_timeout_seconds: int = 5
    float_rounding: int = 12
    numeric_precision: int = 30
    strict_symbol_matching: bool = True
    allow_set_relation_comparison: bool = False
    reference_cache_size: int = 4_096

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        for name in (
            "max_input_chars",
            "max_candidate_chars",
            "direct_answer_max_chars",
            "parsing_timeout_seconds",
            "verification_timeout_seconds",
            "numeric_precision",
            "reference_cache_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_candidate_chars > self.max_input_chars:
            raise ValueError("max_candidate_chars cannot exceed max_input_chars")
        if self.direct_answer_max_chars > self.max_candidate_chars:
            raise ValueError("direct_answer_max_chars cannot exceed max_candidate_chars")
        if isinstance(self.float_rounding, bool) or not isinstance(self.float_rounding, int):
            raise ValueError("float_rounding must be an integer")
        if self.float_rounding < 0 or self.float_rounding > self.numeric_precision:
            raise ValueError("float_rounding must be in [0, numeric_precision]")
        if not isinstance(self.strict_symbol_matching, bool):
            raise ValueError("strict_symbol_matching must be Boolean")
        if not isinstance(self.allow_set_relation_comparison, bool):
            raise ValueError("allow_set_relation_comparison must be Boolean")

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Bounded, serializable answer extraction record."""

    status: ExtractionStatus
    source: AnswerSource | None = None
    candidate: str | None = None
    marker_count: int = 0
    span_start: int | None = None
    span_end: int | None = None
    reason: str | None = None

    @property
    def extracted(self) -> bool:
        return self.status is ExtractionStatus.EXTRACTED

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "source": self.source.value if self.source is not None else None,
            "candidate": self.candidate,
            "marker_count": self.marker_count,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    """Extraction plus backend parse outcome.

    ``values`` intentionally stays out of serialized records; it may contain
    SymPy objects and is only passed to the pinned verifier backend.
    """

    status: ParseStatus
    extraction: ExtractionResult
    values: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    value_types: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None

    @property
    def parsed(self) -> bool:
        return self.status is ParseStatus.PARSED

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "extraction": self.extraction.to_record(),
            "value_count": len(self.values),
            "value_types": list(self.value_types),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Auditable binary reward decision for one reference/prediction pair."""

    status: VerificationStatus
    reward: float | None
    matched: bool | None
    reference: ParsedAnswer
    prediction: ParsedAnswer
    policy_digest: str
    backend_versions: dict[str, str]
    reason: str | None = None

    @property
    def assignable(self) -> bool:
        return self.reward is not None

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reward": self.reward,
            "matched": self.matched,
            "reference": self.reference.to_record(),
            "prediction": self.prediction.to_record(),
            "policy_digest": self.policy_digest,
            "backend_versions": dict(self.backend_versions),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Binary rewards and full decisions for a logical reward batch."""

    rewards: tuple[float, ...]
    results: tuple[VerificationResult, ...]

    @property
    def correct_count(self) -> int:
        return sum(reward == 1.0 for reward in self.rewards)

    @property
    def incorrect_count(self) -> int:
        return len(self.rewards) - self.correct_count


@dataclass(frozen=True, slots=True)
class _AnswerSurface:
    start: int
    end: int
    source: AnswerSource
    candidate: str | None
    reason: str | None = None


def verifier_backend_versions() -> dict[str, str]:
    """Return exact symbolic-backend package versions for provenance."""

    packages = (
        "math-verify",
        "latex2sympy2-extended",
        "antlr4-python3-runtime",
        "sympy",
    )
    return {package: version(package) for package in packages}


def _is_escaped(text: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def _inside_markdown_code(text: str, index: int) -> bool:
    """Conservatively track fenced and inline Markdown code spans."""

    active_fence: tuple[str, int, int] | None = None
    fenced_ranges: list[tuple[int, int]] = []
    for match in _MARKDOWN_FENCE_LINE.finditer(text):
        if match.start() >= index:
            break
        delimiter = match.group(1)
        delimiter_character = delimiter[0]
        trailing_text = match.group(2)
        if active_fence is None:
            if delimiter_character == "`" and "`" in trailing_text:
                continue
            active_fence = (delimiter_character, len(delimiter), match.start())
            continue
        active_character, active_length, opening_start = active_fence
        if (
            delimiter_character == active_character
            and len(delimiter) >= active_length
            and not trailing_text.strip()
        ):
            fenced_ranges.append((opening_start, match.end()))
            active_fence = None
    if active_fence is not None:
        return True

    active_inline_delimiter_length: int | None = None
    for match in _BACKTICK_RUN.finditer(text, 0, index):
        if any(start <= match.start() <= end for start, end in fenced_ranges):
            continue
        delimiter_length = match.end() - match.start()
        if active_inline_delimiter_length is None:
            active_inline_delimiter_length = delimiter_length
        elif delimiter_length == active_inline_delimiter_length:
            active_inline_delimiter_length = None
    return active_inline_delimiter_length is not None


def _balanced_box_surfaces(text: str) -> list[_AnswerSurface]:
    surfaces: list[_AnswerSurface] = []
    for match in _BOX_COMMAND.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if _inside_markdown_code(text, match.start()):
            surfaces.append(
                _AnswerSurface(
                    match.start(),
                    match.end(),
                    AnswerSource.BOXED,
                    None,
                    "answer marker is inside a fenced code block",
                )
            )
            continue
        if cursor >= len(text) or text[cursor] != "{":
            surfaces.append(
                _AnswerSurface(
                    match.start(),
                    match.end(),
                    AnswerSource.BOXED,
                    None,
                    "boxed answer is missing an opening brace",
                )
            )
            continue

        depth = 0
        close_index: int | None = None
        for index in range(cursor, len(text)):
            character = text[index]
            if character == "{" and not _is_escaped(text, index):
                depth += 1
            elif character == "}" and not _is_escaped(text, index):
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
                if depth < 0:
                    break
        if close_index is None:
            surfaces.append(
                _AnswerSurface(
                    match.start(),
                    len(text),
                    AnswerSource.BOXED,
                    None,
                    "boxed answer has unbalanced braces",
                )
            )
            continue
        candidate = text[cursor + 1 : close_index].strip()
        surfaces.append(
            _AnswerSurface(
                match.start(),
                close_index + 1,
                AnswerSource.BOXED,
                candidate or None,
                None if candidate else "boxed answer is empty",
            )
        )
    return surfaces


def _answer_tag_surfaces(text: str) -> list[_AnswerSurface]:
    surfaces: list[_AnswerSurface] = []
    openings = list(_ANSWER_TAG_OPEN.finditer(text))
    closings = list(_ANSWER_TAG_CLOSE.finditer(text))
    tag_tokens = sorted(
        [
            *((match.start(), True) for match in openings),
            *((match.start(), False) for match in closings),
        ]
    )
    stack: list[int] = []
    nested_openings: set[int] = set()
    for position, is_opening in tag_tokens:
        if is_opening:
            if stack:
                nested_openings.add(position)
                nested_openings.update(stack)
            stack.append(position)
        elif stack:
            stack.pop()

    for opening in openings:
        if _inside_markdown_code(text, opening.start()):
            surfaces.append(
                _AnswerSurface(
                    opening.start(),
                    opening.end(),
                    AnswerSource.ANSWER_TAG,
                    None,
                    "answer tag is inside a fenced code block",
                )
            )
            continue
        if opening.start() in nested_openings:
            surfaces.append(
                _AnswerSurface(
                    opening.start(),
                    len(text),
                    AnswerSource.ANSWER_TAG,
                    None,
                    "nested answer tags are not a valid terminal answer surface",
                )
            )
            continue
        closing = _ANSWER_TAG_CLOSE.search(text, opening.end())
        next_opening = _ANSWER_TAG_OPEN.search(text, opening.end())
        if closing is None or (
            next_opening is not None and next_opening.start() < closing.start()
        ):
            surfaces.append(
                _AnswerSurface(
                    opening.start(),
                    len(text),
                    AnswerSource.ANSWER_TAG,
                    None,
                    "answer tag is missing a non-nested closing tag",
                )
            )
            continue
        candidate = text[opening.end() : closing.start()].strip()
        surfaces.append(
            _AnswerSurface(
                opening.start(),
                closing.end(),
                AnswerSource.ANSWER_TAG,
                candidate or None,
                None if candidate else "answer tag is empty",
            )
        )
    return surfaces


def _top_level_container_surfaces(text: str) -> list[_AnswerSurface]:
    containers: list[_AnswerSurface] = []
    for surface in sorted(
        [*_balanced_box_surfaces(text), *_answer_tag_surfaces(text)],
        key=lambda item: (item.start, -item.end),
    ):
        if any(container.start < surface.start < container.end for container in containers):
            continue
        containers.append(surface)
    return containers


def _clean_text_marker_candidate(candidate: str) -> str:
    candidate = candidate.strip()
    candidate = re.sub(r"^(?:\*\*|__)\s*", "", candidate)
    candidate = re.sub(r"\s*(?:\*\*|__)$", "", candidate)
    if candidate.endswith(".") and not candidate.endswith(".."):
        candidate = candidate[:-1]
    if candidate.endswith("\N{IDEOGRAPHIC FULL STOP}"):
        candidate = candidate[:-1]
    return candidate.strip()


def _text_marker_surfaces(text: str) -> list[_AnswerSurface]:
    surfaces: list[_AnswerSurface] = []
    for marker in _TEXT_ANSWER_MARKER.finditer(text):
        if _inside_markdown_code(text, marker.start()):
            surfaces.append(
                _AnswerSurface(
                    marker.start(),
                    marker.end(),
                    AnswerSource.TEXT_MARKER,
                    None,
                    "answer marker is inside a fenced code block",
                )
            )
            continue
        line_start = text.rfind("\n", 0, marker.start()) + 1
        marker_prefix = text[line_start : marker.start()].rstrip()
        marker_clause_prefix = re.split(r"[.;!?\u3002\uff1b\uff01\uff1f]", marker_prefix)[-1]
        if _NEGATED_ANSWER_PREFIX.search(marker_clause_prefix):
            continue
        line_end = text.find("\n", marker.end())
        if line_end < 0:
            line_end = len(text)
        candidate = _clean_text_marker_candidate(text[marker.end() : line_end])
        surfaces.append(
            _AnswerSurface(
                marker.start(),
                line_end,
                AnswerSource.TEXT_MARKER,
                candidate or None,
                None if candidate else "text answer marker has no value",
            )
        )
    return surfaces


def _looks_like_direct_answer(text: str, *, limit: int) -> bool:
    if not text or len(text) > limit or "`" in text:
        return False
    if len([line for line in text.splitlines() if line.strip()]) != 1:
        return False
    if _CJK_CHARACTER.search(text):
        return False
    without_commands = _LATEX_COMMAND.sub("", text)
    words = _ASCII_WORD.findall(without_commands)
    return all(len(word) == 1 or word.lower() in _DIRECT_ALLOWED_WORDS for word in words)


def _has_one_or_no_outer_math_environment(candidate: str) -> bool:
    """Reject mixed or repeated math delimiters that enable rightmost-match hijacking."""

    dollar_positions = [
        index
        for index, character in enumerate(candidate)
        if character == "$" and not _is_escaped(candidate, index)
    ]
    latex_pairs = ((r"\(", r"\)"), (r"\[", r"\]"))
    present_latex_pairs = [
        (opening, closing)
        for opening, closing in latex_pairs
        if opening in candidate or closing in candidate
    ]
    if dollar_positions and present_latex_pairs:
        return False
    if dollar_positions:
        single_dollar = dollar_positions == [0, len(candidate) - 1] and len(candidate) > 2
        double_dollar = (
            dollar_positions == [0, 1, len(candidate) - 2, len(candidate) - 1]
            and len(candidate) > 4
        )
        return single_dollar or double_dollar
    if present_latex_pairs:
        if len(present_latex_pairs) != 1:
            return False
        opening, closing = present_latex_pairs[0]
        return (
            candidate.startswith(opening)
            and candidate.endswith(closing)
            and candidate.count(opening) == 1
            and candidate.count(closing) == 1
            and len(candidate) > len(opening) + len(closing)
        )
    return True


def _outer_math_parts(candidate: str) -> tuple[str, str, str]:
    """Split one already-validated outer math environment from its body."""

    stripped = candidate.strip()
    for opening, closing in _MATH_ENVIRONMENT_PAIRS:
        if (
            stripped.startswith(opening)
            and stripped.endswith(closing)
            and len(stripped) > len(opening) + len(closing)
        ):
            return opening, stripped[len(opening) : -len(closing)], closing
    return "", stripped, ""


def _matching_delimiter_index(
    text: str,
    opening_index: int,
    *,
    opening: str,
    closing: str,
) -> int | None:
    """Find a balanced delimiter while respecting escaped delimiter characters."""

    if opening_index >= len(text) or text[opening_index] != opening:
        return None
    depth = 0
    for index in range(opening_index, len(text)):
        character = text[index]
        if character == opening and not _is_escaped(text, index):
            depth += 1
        elif character == closing and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _text_like_commands_are_allowed_units(candidate: str) -> bool:
    """Allow only one trailing text-like command containing frozen unit words.

    Math-Verify's unit normalization removes arbitrary ``text``/``mbox``
    payloads.  Validating the complete argument before invoking the backend
    prevents negation, prose, digits, Unicode, or nested commands from being
    silently discarded.
    """

    _, body, _ = _outer_math_parts(candidate)
    matches = list(_TEXT_LIKE_COMMAND.finditer(body))
    if not matches:
        return True
    if len(matches) != 1:
        return False
    match = matches[0]
    if _is_escaped(body, match.start()):
        return False
    opening_index = match.end()
    while opening_index < len(body) and body[opening_index].isspace():
        opening_index += 1
    if opening_index >= len(body) or body[opening_index] != "{":
        return False
    closing_index = _matching_delimiter_index(
        body,
        opening_index,
        opening="{",
        closing="}",
    )
    if closing_index is None or body[closing_index + 1 :].strip():
        return False
    if not body[: match.start()].strip():
        return False

    content = body[opening_index + 1 : closing_index].strip()
    if re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+)*", content) is None:
        return False
    return all(word.lower() in _ALLOWED_UNIT_WORDS for word in content.split())


def _skip_ignorable_math_spacing(text: str, index: int) -> int:
    cursor = index
    while cursor < len(text):
        match = _IGNORABLE_MATH_SPACING.match(text, cursor)
        if match is None:
            break
        cursor = match.end()
    return cursor


def _protected_command_argument_boundaries(text: str) -> set[tuple[int, int]]:
    """Return delimiter transitions that are command syntax, not products."""

    protected: set[tuple[int, int]] = set()
    for match in _TWO_ARGUMENT_PRIMARY_COMMAND.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        first_open = _skip_ignorable_math_spacing(text, match.end())
        first_close = _matching_delimiter_index(
            text,
            first_open,
            opening="{",
            closing="}",
        )
        if first_close is None:
            continue
        second_open = _skip_ignorable_math_spacing(text, first_close + 1)
        if second_open < len(text) and text[second_open] == "{":
            protected.add((first_close, second_open))

    for match in _SQRT_COMMAND.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        optional_open = _skip_ignorable_math_spacing(text, match.end())
        if optional_open >= len(text) or text[optional_open] != "[":
            continue
        optional_close = _matching_delimiter_index(
            text,
            optional_open,
            opening="[",
            closing="]",
        )
        if optional_close is None:
            continue
        required_open = _skip_ignorable_math_spacing(text, optional_close + 1)
        if required_open < len(text) and text[required_open] == "{":
            protected.add((optional_close, required_open))

    for match in _BEGIN_COMMAND.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        environment_open = _skip_ignorable_math_spacing(text, match.end())
        environment_close = _matching_delimiter_index(
            text,
            environment_open,
            opening="{",
            closing="}",
        )
        if environment_close is None:
            continue
        content_start = _skip_ignorable_math_spacing(text, environment_close + 1)
        if content_start < len(text):
            protected.add((environment_close, content_start))

    for match in _FUNCTION_PRIMARY_COMMAND.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        cursor = _skip_ignorable_math_spacing(text, match.end())
        left_end: int | None = None
        if match.group(0).lower() == r"\operatorname":
            operator_name_close = _matching_delimiter_index(
                text,
                cursor,
                opening="{",
                closing="}",
            )
            if operator_name_close is None:
                continue
            left_end = operator_name_close
            cursor = operator_name_close + 1

        while True:
            script_marker = _skip_ignorable_math_spacing(text, cursor)
            if script_marker >= len(text) or text[script_marker] not in "_^":
                cursor = script_marker
                break
            script_start = _skip_ignorable_math_spacing(text, script_marker + 1)
            if script_start >= len(text):
                break
            if text[script_start] == "{":
                script_end = _matching_delimiter_index(
                    text,
                    script_start,
                    opening="{",
                    closing="}",
                )
                if script_end is None:
                    break
            elif text[script_start] == "\\":
                command = _LATEX_COMMAND.match(text, script_start)
                script_end = command.end() - 1 if command is not None else script_start
            else:
                script_end = script_start
            left_end = script_end
            cursor = script_end + 1

        argument_start = _skip_ignorable_math_spacing(text, cursor)
        if (
            left_end is not None
            and _is_juxtaposition_primary_start(text, argument_start)
        ):
            protected.add((left_end, argument_start))
    return protected


def _is_juxtaposition_primary_start(text: str, index: int) -> bool:
    if index >= len(text):
        return False
    character = text[index]
    if character in "([{":
        return not _is_escaped(text, index)
    if character.isdigit() or (
        character == "." and index + 1 < len(text) and text[index + 1].isdigit()
    ):
        return True
    match = _JUXTAPOSITION_PRIMARY_COMMAND.match(text, index)
    return match is not None and not _is_escaped(text, index)


def _is_primary_closing_delimiter(text: str, index: int) -> bool:
    if text[index] not in ")]}" or not _is_escaped(text, index):
        return text[index] in ")]}" and not _is_escaped(text, index)
    prefix = text[:index]
    return re.search(r"\\right\s*\\?$", prefix, re.IGNORECASE) is not None


def _explicitize_ambiguous_juxtaposition(candidate: str) -> str:
    """Insert multiplication at primary boundaries for a semantic cross-check.

    The pinned parser intentionally accepts mixed-number syntax, but the same
    path can turn ``1(41)`` into ``42``.  The original parse is accepted only
    when making such juxtaposition explicit leaves the symbolic value intact.
    """

    protected = _protected_command_argument_boundaries(candidate)
    insertion_points: set[int] = set()
    for left_index, character in enumerate(candidate):
        left_is_number = character.isdigit()
        left_is_closing = _is_primary_closing_delimiter(candidate, left_index)
        if not left_is_number and not left_is_closing:
            continue
        right_index = _skip_ignorable_math_spacing(candidate, left_index + 1)
        if (
            right_index == left_index + 1
            and left_is_number
            and right_index < len(candidate)
            and candidate[right_index] in ".0123456789"
        ):
            continue
        if (left_index, right_index) in protected:
            continue
        if _is_juxtaposition_primary_start(candidate, right_index):
            insertion_points.add(right_index)

    if not insertion_points:
        return candidate
    return "".join(
        (r"\cdot " if index in insertion_points else "") + character
        for index, character in enumerate(candidate)
    )


def _looks_like_single_answer_candidate(candidate: str, *, limit: int) -> bool:
    """Accept one compact math surface, never a prose/multi-surface payload."""

    if not _has_one_or_no_outer_math_environment(candidate):
        return False
    if not _text_like_commands_are_allowed_units(candidate):
        return False
    if any(_is_escaped(candidate, match.start()) for match in _BOX_COMMAND.finditer(candidate)):
        return False
    if (
        _DISALLOWED_CANDIDATE_COMMAND.search(candidate)
        or _DISALLOWED_REMOVED_CHARACTER.search(candidate)
        or _LATEX_ROW_BREAK.search(candidate)
        or _LOGICAL_CONNECTOR_CHARACTER.search(candidate)
        or _NUMERIC_SUBSCRIPT.search(candidate)
        or _TRANSPOSE_SUPERSCRIPT.search(candidate)
        or candidate.count("=") > 1
    ):
        return False
    if any(_is_escaped(candidate, match.start()) for match in _LATEX_COMMAND.finditer(candidate)):
        return False
    if len(candidate) > limit or "`" in candidate:
        return False
    if len([line for line in candidate.splitlines() if line.strip()]) != 1:
        return False
    gap_normalized = _EMPTY_OR_SPACING_LATEX.sub(" ", candidate)
    if _ADJACENT_NUMBERS_WITHOUT_OPERATOR.search(gap_normalized):
        return False
    without_commands = _LATEX_COMMAND.sub("", candidate)
    words = {word.lower() for word in _ASCII_WORD.findall(without_commands)}
    return words.isdisjoint(_DISALLOWED_CANDIDATE_WORDS)


def _canonical_prediction_candidate(candidate: str) -> str | None:
    """Unwrap only exact nested containers; reject any container with side material."""

    current = candidate.strip()
    for _ in range(4):
        containers = _top_level_container_surfaces(current)
        if not containers:
            return current
        if len(containers) != 1:
            return None
        container = containers[0]
        if (
            container.start != 0
            or container.end != len(current)
            or container.candidate is None
        ):
            return None
        current = container.candidate.strip()
        if not current:
            return None
    return None if _top_level_container_surfaces(current) else current


def _bounded_input(text: str, *, policy: VerifierPolicy) -> ExtractionResult | None:
    if not isinstance(text, str):
        raise TypeError("answer text must be a string")
    if not text.strip():
        return ExtractionResult(status=ExtractionStatus.EMPTY_INPUT, reason="answer is empty")
    if len(text) > policy.max_input_chars:
        return ExtractionResult(
            status=ExtractionStatus.INPUT_TOO_LONG,
            reason=f"answer exceeds {policy.max_input_chars} characters",
        )
    if _CONTROL_CHARACTER.search(text):
        return ExtractionResult(
            status=ExtractionStatus.UNSAFE_CONTROL_CHARACTER,
            reason="answer contains a disallowed control character",
        )
    return None


def extract_reference_answer(
    reference: str,
    *,
    policy: VerifierPolicy | None = None,
) -> ExtractionResult:
    """Treat a trusted dataset reference field as one bounded answer surface."""

    resolved_policy = policy or VerifierPolicy()
    invalid = _bounded_input(reference, policy=resolved_policy)
    if invalid is not None:
        return invalid
    candidate = reference.strip()
    if len(candidate) > resolved_policy.max_candidate_chars:
        return ExtractionResult(
            status=ExtractionStatus.CANDIDATE_TOO_LONG,
            source=AnswerSource.REFERENCE,
            reason=f"reference exceeds {resolved_policy.max_candidate_chars} characters",
        )
    return ExtractionResult(
        status=ExtractionStatus.EXTRACTED,
        source=AnswerSource.REFERENCE,
        candidate=candidate,
        marker_count=1,
        span_start=0,
        span_end=len(reference),
    )


def extract_prediction_answer(
    prediction: str,
    *,
    policy: VerifierPolicy | None = None,
) -> ExtractionResult:
    """Extract the last explicit terminal answer, or one concise direct answer."""

    resolved_policy = policy or VerifierPolicy()
    invalid = _bounded_input(prediction, policy=resolved_policy)
    if invalid is not None:
        return invalid

    # A malformed container owns the remainder of its span too.  Otherwise an
    # inner text marker could rescue an unclosed final box/tag and bypass the
    # fail-closed policy.
    containers = _top_level_container_surfaces(prediction)
    text_surfaces = [
        surface
        for surface in _text_marker_surfaces(prediction)
        if not any(
            container.start <= surface.start < container.end for container in containers
        )
    ]
    surfaces = [*containers, *text_surfaces]
    if surfaces:
        selected = max(surfaces, key=lambda surface: (surface.start, surface.end))
        if selected.candidate is None:
            return ExtractionResult(
                status=ExtractionStatus.MALFORMED_FINAL_ANSWER,
                source=selected.source,
                marker_count=len(surfaces),
                span_start=selected.start,
                span_end=selected.end,
                reason=selected.reason,
            )
        if len(selected.candidate) > resolved_policy.max_candidate_chars:
            return ExtractionResult(
                status=ExtractionStatus.CANDIDATE_TOO_LONG,
                source=selected.source,
                marker_count=len(surfaces),
                span_start=selected.start,
                span_end=selected.end,
                reason=f"candidate exceeds {resolved_policy.max_candidate_chars} characters",
            )
        return ExtractionResult(
            status=ExtractionStatus.EXTRACTED,
            source=selected.source,
            candidate=selected.candidate,
            marker_count=len(surfaces),
            span_start=selected.start,
            span_end=selected.end,
        )

    candidate = prediction.strip()
    if _looks_like_direct_answer(candidate, limit=resolved_policy.direct_answer_max_chars):
        return ExtractionResult(
            status=ExtractionStatus.EXTRACTED,
            source=AnswerSource.DIRECT,
            candidate=candidate,
            marker_count=0,
            span_start=prediction.find(candidate),
            span_end=prediction.find(candidate) + len(candidate),
        )
    return ExtractionResult(
        status=ExtractionStatus.NO_FINAL_ANSWER,
        marker_count=0,
        reason="prediction has no explicit terminal answer or concise direct math surface",
    )


def _already_in_math_environment(candidate: str) -> bool:
    stripped = candidate.strip()
    return any(
        stripped.startswith(opening)
        and stripped.endswith(closing)
        and len(stripped) >= len(opening) + len(closing)
        for opening, closing in _MATH_ENVIRONMENT_PAIRS
    )


def _prepare_candidate_for_backend(
    extraction: ExtractionResult,
    *,
    candidate_override: str | None = None,
) -> str:
    assert extraction.candidate is not None
    candidate = (
        candidate_override.strip()
        if candidate_override is not None
        else extraction.candidate.strip()
    )
    if extraction.source is AnswerSource.REFERENCE and not _looks_like_direct_answer(
        candidate,
        limit=len(candidate),
    ):
        return candidate
    if _already_in_math_environment(candidate):
        return candidate
    return f"${candidate}$"


def _prediction_extraction_config() -> list[LatexExtractionConfig | ExprExtractionConfig]:
    return [
        LatexExtractionConfig(
            try_extract_without_anchor=True,
            boxed_match_priority=50,
            normalization_config=LatexNormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                boxed="last",
                equations=False,
            ),
        ),
        ExprExtractionConfig(try_extract_without_anchor=True),
    ]


def _reference_extraction_config() -> list[LatexExtractionConfig | ExprExtractionConfig]:
    return [
        LatexExtractionConfig(
            try_extract_without_anchor=True,
            boxed_match_priority=50,
            normalization_config=LatexNormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=True,
                nits=True,
                boxed="last",
                equations=False,
            ),
        ),
        ExprExtractionConfig(try_extract_without_anchor=True),
    ]


def _structural_family(value: Any) -> str:
    if isinstance(value, FiniteSet):
        return "finite_set"
    if isinstance(value, Interval):
        return "interval"
    if isinstance(value, Tuple):
        return "tuple"
    if isinstance(value, Relational):
        return "relation"
    if isinstance(value, BooleanFunction):
        return "logical_relation"
    if isinstance(value, MatrixBase):
        return "matrix"
    if isinstance(value, Set):
        return "set"
    return "scalar"


def _simple_assignment_parts(value: Any) -> tuple[Symbol, Any] | None:
    if not isinstance(value, Equality):
        return None
    if (
        isinstance(value.lhs, Symbol)
        and value.lhs not in value.rhs.free_symbols
        and _structural_family(value.rhs) == "scalar"
    ):
        return value.lhs, value.rhs
    return None


def _reversible_assignment_parts(value: Any) -> tuple[Symbol, Any] | None:
    direct = _simple_assignment_parts(value)
    if direct is not None:
        return direct
    if not isinstance(value, Equality):
        return None
    if (
        isinstance(value.rhs, Symbol)
        and value.rhs not in value.lhs.free_symbols
        and _structural_family(value.lhs) == "scalar"
    ):
        return value.rhs, value.lhs
    return None


def _compatible_comparison_pair(
    reference: Any,
    prediction: Any,
    *,
    strict_symbol_matching: bool,
    allow_set_relation_comparison: bool,
) -> tuple[Any, Any] | None:
    """Return a safe backend pair, normalizing only simple assignments."""

    reference_family = _structural_family(reference)
    prediction_family = _structural_family(prediction)
    reference_assignment = _simple_assignment_parts(reference)
    prediction_assignment = _simple_assignment_parts(prediction)

    if reference_family == "relation" and prediction_family == "relation":
        reference_assignment = _reversible_assignment_parts(reference)
        prediction_assignment = _reversible_assignment_parts(prediction)
        if reference_assignment is not None or prediction_assignment is not None:
            if reference_assignment is None or prediction_assignment is None:
                return None
            reference_symbol, reference_value = reference_assignment
            prediction_symbol, prediction_value = prediction_assignment
            if strict_symbol_matching and reference_symbol != prediction_symbol:
                return None
            return reference_value, prediction_value
        if isinstance(reference, Equality) or isinstance(prediction, Equality):
            return (reference, prediction) if reference == prediction else None
        canonical_reference = reference.canonical
        canonical_prediction = prediction.canonical
        if type(canonical_reference) is not type(canonical_prediction):
            return None
        if (
            strict_symbol_matching
            and canonical_reference.free_symbols != canonical_prediction.free_symbols
        ):
            return None
        if not canonical_reference.free_symbols and canonical_reference != canonical_prediction:
            return None
        return canonical_reference, canonical_prediction

    if reference_family == "logical_relation" and prediction_family == "logical_relation":
        return (reference, prediction) if reference == prediction else None

    if reference_family == "relation" and prediction_family == "scalar":
        if reference_assignment is None:
            return None
        return reference_assignment[1], prediction
    if reference_family == "scalar" and prediction_family == "relation":
        if prediction_assignment is None:
            return None
        return reference, prediction_assignment[1]

    if reference_family == prediction_family:
        return reference, prediction
    if {reference_family, prediction_family} <= {"finite_set", "set"}:
        return reference, prediction
    if allow_set_relation_comparison and (
        "relation" in {reference_family, prediction_family}
        and {reference_family, prediction_family} & {"finite_set", "interval", "set"}
    ):
        return reference, prediction
    return None


def _compatible_comparison_pairs(
    reference_values: tuple[Any, ...],
    prediction_values: tuple[Any, ...],
    *,
    strict_symbol_matching: bool,
    allow_set_relation_comparison: bool,
) -> tuple[tuple[Any, Any], ...]:
    pairs: list[tuple[Any, Any]] = []
    for reference in reference_values:
        for prediction in prediction_values:
            pair = _compatible_comparison_pair(
                reference,
                prediction,
                strict_symbol_matching=strict_symbol_matching,
                allow_set_relation_comparison=allow_set_relation_comparison,
            )
            if pair is not None:
                pairs.append(pair)
    return tuple(pairs)


def _parsed_values_are_identical(first: tuple[Any, ...], second: tuple[Any, ...]) -> bool:
    if len(first) != len(second):
        return False
    try:
        return all(
            type(left) is type(right) and bool(left == right)
            for left, right in zip(first, second, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _parse_extraction(
    extraction: ExtractionResult,
    *,
    role: AnswerSource,
    policy: VerifierPolicy,
) -> ParsedAnswer:
    if not extraction.extracted:
        return ParsedAnswer(
            status=ParseStatus.NOT_EXTRACTED,
            extraction=extraction,
            reason=extraction.reason,
        )
    if threading.current_thread() is not threading.main_thread():
        return ParsedAnswer(
            status=ParseStatus.BACKEND_ERROR,
            extraction=extraction,
            reason="bounded Math-Verify parsing must run in a process main thread",
        )

    assert extraction.candidate is not None
    canonical_candidate: str | None = None
    if role is not AnswerSource.REFERENCE:
        canonical_candidate = _canonical_prediction_candidate(extraction.candidate)
        if canonical_candidate is None or not _looks_like_single_answer_candidate(
            canonical_candidate,
            limit=policy.max_candidate_chars,
        ):
            return ParsedAnswer(
                status=ParseStatus.UNPARSEABLE,
                extraction=extraction,
                reason="prediction candidate is not one unambiguous mathematical surface",
            )
    semantic_candidate = canonical_candidate or extraction.candidate
    if not _text_like_commands_are_allowed_units(semantic_candidate):
        return ParsedAnswer(
            status=ParseStatus.UNPARSEABLE,
            extraction=extraction,
            reason="text-like command is not one frozen trailing unit phrase",
        )

    backend_text = _prepare_candidate_for_backend(
        extraction,
        candidate_override=canonical_candidate,
    )
    explicit_juxtaposition_candidate = _explicitize_ambiguous_juxtaposition(
        semantic_candidate
    )
    config = (
        _reference_extraction_config()
        if role is AnswerSource.REFERENCE
        else _prediction_extraction_config()
    )
    try:
        values = math_verify_parse(
            backend_text,
            extraction_config=config,
            fallback_mode="no_fallback",
            extraction_mode="any_match" if role is AnswerSource.REFERENCE else "first_match",
            parsing_timeout=policy.parsing_timeout_seconds,
            raise_on_error=True,
        )
    except TimeoutException:
        return ParsedAnswer(
            status=ParseStatus.TIMEOUT,
            extraction=extraction,
            reason="symbolic parsing exceeded the configured timeout",
        )
    except Exception as error:  # backend versions are pinned; surface only the exception type
        return ParsedAnswer(
            status=ParseStatus.BACKEND_ERROR,
            extraction=extraction,
            reason=f"symbolic parser raised {type(error).__name__}",
        )
    if not values:
        return ParsedAnswer(
            status=ParseStatus.UNPARSEABLE,
            extraction=extraction,
            reason="symbolic parser produced no value",
        )
    value_tuple = tuple(values)
    if explicit_juxtaposition_candidate != semantic_candidate:
        explicit_backend_text = _prepare_candidate_for_backend(
            extraction,
            candidate_override=explicit_juxtaposition_candidate,
        )
        try:
            explicit_values = math_verify_parse(
                explicit_backend_text,
                extraction_config=config,
                fallback_mode="no_fallback",
                extraction_mode=(
                    "any_match" if role is AnswerSource.REFERENCE else "first_match"
                ),
                parsing_timeout=policy.parsing_timeout_seconds,
                raise_on_error=True,
            )
        except TimeoutException:
            return ParsedAnswer(
                status=ParseStatus.TIMEOUT,
                extraction=extraction,
                reason="juxtaposition cross-check exceeded the configured timeout",
            )
        except Exception as error:
            return ParsedAnswer(
                status=ParseStatus.BACKEND_ERROR,
                extraction=extraction,
                reason=f"juxtaposition cross-check raised {type(error).__name__}",
            )
        explicit_value_tuple = tuple(explicit_values)
        if not explicit_value_tuple or not _parsed_values_are_identical(
            value_tuple,
            explicit_value_tuple,
        ):
            return ParsedAnswer(
                status=ParseStatus.UNPARSEABLE,
                extraction=extraction,
                reason=(
                    "implicit primary juxtaposition changes meaning when "
                    "made explicit"
                ),
            )
    return ParsedAnswer(
        status=ParseStatus.PARSED,
        extraction=extraction,
        values=value_tuple,
        value_types=tuple(type(value).__name__ for value in value_tuple),
    )


class ExactMathVerifier:
    """Stateful verifier with bounded reference parsing cache."""

    def __init__(self, policy: VerifierPolicy | None = None) -> None:
        self.policy = policy or VerifierPolicy()
        self._versions = verifier_backend_versions()
        self._reference_cache: OrderedDict[str, ParsedAnswer] = OrderedDict()

    @property
    def backend_versions(self) -> dict[str, str]:
        return dict(self._versions)

    @property
    def policy_digest(self) -> str:
        return self.policy.digest()

    def clear_reference_cache(self) -> None:
        self._reference_cache.clear()

    def parse_reference(self, reference: str) -> ParsedAnswer:
        if threading.current_thread() is not threading.main_thread():
            extraction = extract_reference_answer(reference, policy=self.policy)
            return _parse_extraction(
                extraction,
                role=AnswerSource.REFERENCE,
                policy=self.policy,
            )
        cached = self._reference_cache.get(reference)
        if cached is not None:
            self._reference_cache.move_to_end(reference)
            return cached

        extraction = extract_reference_answer(reference, policy=self.policy)
        parsed = _parse_extraction(
            extraction,
            role=AnswerSource.REFERENCE,
            policy=self.policy,
        )
        # A transient timeout/backend failure must not poison future batches,
        # and invalid golds can be repaired only by changing the source data.
        if parsed.parsed:
            self._reference_cache[reference] = parsed
            if len(self._reference_cache) > self.policy.reference_cache_size:
                self._reference_cache.popitem(last=False)
        return parsed

    def parse_prediction(self, prediction: str) -> ParsedAnswer:
        extraction = extract_prediction_answer(prediction, policy=self.policy)
        return _parse_extraction(
            extraction,
            role=extraction.source or AnswerSource.DIRECT,
            policy=self.policy,
        )

    def _skipped_prediction(self, reason: str) -> ParsedAnswer:
        extraction = ExtractionResult(
            status=ExtractionStatus.NOT_EVALUATED,
            reason=reason,
        )
        return ParsedAnswer(
            status=ParseStatus.SKIPPED,
            extraction=extraction,
            reason=reason,
        )

    @staticmethod
    def _reference_failure_status(parsed_reference: ParsedAnswer) -> VerificationStatus:
        if parsed_reference.status in {ParseStatus.TIMEOUT, ParseStatus.BACKEND_ERROR}:
            return VerificationStatus.BACKEND_ERROR
        return VerificationStatus.REFERENCE_INVALID

    def _reference_failure_result(
        self,
        parsed_reference: ParsedAnswer,
    ) -> VerificationResult:
        reason = parsed_reference.reason or "reference could not be parsed"
        return VerificationResult(
            status=self._reference_failure_status(parsed_reference),
            reward=None,
            matched=None,
            reference=parsed_reference,
            prediction=self._skipped_prediction(
                "prediction was not evaluated because reference validation failed"
            ),
            policy_digest=self.policy_digest,
            backend_versions=self.backend_versions,
            reason=reason,
        )

    def _verify_parsed_reference(
        self,
        parsed_reference: ParsedAnswer,
        prediction: str,
    ) -> VerificationResult:
        if not parsed_reference.parsed:
            return self._reference_failure_result(parsed_reference)

        parsed_prediction = self.parse_prediction(prediction)
        common = {
            "reference": parsed_reference,
            "prediction": parsed_prediction,
            "policy_digest": self.policy_digest,
            "backend_versions": self.backend_versions,
        }
        if parsed_prediction.status is ParseStatus.NOT_EXTRACTED:
            return VerificationResult(
                status=VerificationStatus.PREDICTION_NOT_EXTRACTED,
                reward=0.0,
                matched=False,
                reason=parsed_prediction.reason,
                **common,
            )
        if parsed_prediction.status is ParseStatus.UNPARSEABLE:
            return VerificationResult(
                status=VerificationStatus.PREDICTION_UNPARSEABLE,
                reward=0.0,
                matched=False,
                reason=parsed_prediction.reason,
                **common,
            )
        if parsed_prediction.status is ParseStatus.TIMEOUT:
            return VerificationResult(
                status=VerificationStatus.PREDICTION_TIMEOUT,
                reward=0.0,
                matched=False,
                reason=parsed_prediction.reason,
                **common,
            )
        if parsed_prediction.status is ParseStatus.BACKEND_ERROR:
            return VerificationResult(
                status=VerificationStatus.BACKEND_ERROR,
                reward=None,
                matched=None,
                reason=parsed_prediction.reason,
                **common,
            )

        comparison_pairs = _compatible_comparison_pairs(
            parsed_reference.values,
            parsed_prediction.values,
            strict_symbol_matching=self.policy.strict_symbol_matching,
            allow_set_relation_comparison=self.policy.allow_set_relation_comparison,
        )
        if not comparison_pairs:
            return VerificationResult(
                status=VerificationStatus.MISMATCH,
                reward=0.0,
                matched=False,
                reason="reference and prediction have incompatible structural answer types",
                **common,
            )

        try:
            matched = any(
                math_verify_compare(
                    [reference_value],
                    [prediction_value],
                    float_rounding=self.policy.float_rounding,
                    numeric_precision=self.policy.numeric_precision,
                    strict=self.policy.strict_symbol_matching,
                    allow_set_relation_comp=self.policy.allow_set_relation_comparison,
                    timeout_seconds=self.policy.verification_timeout_seconds,
                    raise_on_error=True,
                )
                for reference_value, prediction_value in comparison_pairs
            )
        except TimeoutException:
            return VerificationResult(
                status=VerificationStatus.VERIFICATION_TIMEOUT,
                reward=0.0,
                matched=False,
                reason="symbolic comparison exceeded the configured timeout",
                **common,
            )
        except Exception as error:
            return VerificationResult(
                status=VerificationStatus.BACKEND_ERROR,
                reward=None,
                matched=None,
                reason=f"symbolic verifier raised {type(error).__name__}",
                **common,
            )

        return VerificationResult(
            status=VerificationStatus.MATCH if matched else VerificationStatus.MISMATCH,
            reward=1.0 if matched else 0.0,
            matched=matched,
            **common,
        )

    def verify(self, reference: str, prediction: str) -> VerificationResult:
        """Compare ``prediction`` against dataset ``reference`` in that order."""

        parsed_reference = self.parse_reference(reference)
        return self._verify_parsed_reference(parsed_reference, prediction)

    def exact_reward(self, reference: str, prediction: str) -> float:
        """Return exactly 0/1, failing on invalid gold or backend errors."""

        result = self.verify(reference, prediction)
        if result.reward is None:
            raise VerifierInfrastructureError(
                f"cannot assign exact reward: {result.status.value}: {result.reason}"
            )
        return result.reward

    def score_batch(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> RewardBatch:
        """Score one logical batch without silently truncating either sequence."""

        if isinstance(references, (str, bytes)) or isinstance(predictions, (str, bytes)):
            raise TypeError("references and predictions must be sequences of strings")
        if len(references) != len(predictions):
            raise ValueError("references and predictions must have the same length")
        if not references:
            raise ValueError("reward batch must be non-empty")
        # Validate every gold before parsing any model output.  A corrupt gold
        # invalidates the logical batch and must never be mislabeled as a model
        # error or allow later samples to consume verifier work.
        parsed_references: list[ParsedAnswer] = []
        for index, reference in enumerate(references):
            parsed_reference = self.parse_reference(reference)
            if not parsed_reference.parsed:
                status = self._reference_failure_status(parsed_reference)
                raise VerifierInfrastructureError(
                    f"cannot assign batch rewards at index {index}: "
                    f"{status.value}: {parsed_reference.reason}"
                )
            parsed_references.append(parsed_reference)

        results: list[VerificationResult] = []
        rewards: list[float] = []
        for index, (parsed_reference, prediction) in enumerate(
            zip(parsed_references, predictions, strict=True)
        ):
            result = self._verify_parsed_reference(parsed_reference, prediction)
            if result.reward is None:
                raise VerifierInfrastructureError(
                    f"cannot assign batch rewards at index {index}: "
                    f"{result.status.value}: {result.reason}"
                )
            results.append(result)
            rewards.append(result.reward)
        return RewardBatch(rewards=tuple(rewards), results=tuple(results))


DEFAULT_VERIFIER_POLICY = VerifierPolicy()


def verify_math_answer(
    reference: str,
    prediction: str,
    *,
    policy: VerifierPolicy | None = None,
) -> VerificationResult:
    """Stateless convenience entrypoint with explicit gold/prediction names."""

    return ExactMathVerifier(policy).verify(reference, prediction)


def exact_math_reward(
    reference: str,
    prediction: str,
    *,
    policy: VerifierPolicy | None = None,
) -> float:
    """Stateless exact binary reward entrypoint."""

    return ExactMathVerifier(policy).exact_reward(reference, prediction)
