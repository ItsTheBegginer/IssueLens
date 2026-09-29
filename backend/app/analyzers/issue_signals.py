"""Deterministic signal extractor for GitHub issues.

Extracts structured signals from issue text (title, body, comments) using
regex and string processing only.  No AI, no network calls.

Public API
----------
``extract_signals(issue) -> IssueSignals``
"""
from __future__ import annotations

import re
from typing import Optional

from app.models.github import IssueDetail
from app.models.issue_signals import IssueSignals

__all__ = ["extract_signals"]


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# File paths: optional leading backtick, then path-like token with extension
# Matches: `src/foo/bar.py`, routes/login.js, config.yaml, ./setup.cfg
_FILE_PATH_RE = re.compile(
    r"`?(?:\.{0,2}/)?"                     # optional backtick + leading ./ or ../
    r"(?:[A-Za-z0-9_.-]+/)*"               # zero or more directory segments
    r"[A-Za-z0-9_.-]+"                     # filename stem
    r"\."                                  # required dot
    r"(?:py|js|ts|jsx|tsx|java|kt|go|rb|rs|cpp|c|h|cs|php|swift|"
    r"yaml|yml|json|toml|cfg|ini|env|sh|bash|md|txt|xml|html|css)"
    r"(?:`|\b)",                           # trailing backtick or word boundary
    re.IGNORECASE,
)

# CamelCase class names: two or more capital-led words, not all-caps acronyms
_CAMEL_CLASS_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")

# snake_case function/method names (2+ segments, not pure numbers)
_SNAKE_FUNC_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+)\b")

# lowerCamelCase identifiers: start lowercase, contain at least one uppercase
# e.g. useDataFetcher, parseConfig, handleRequest
_LOWER_CAMEL_RE = re.compile(r"\b([a-z][a-z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)\b")

# Function call pattern: identifier immediately followed by (
_FUNC_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# Code fence blocks: ```...``` or `...`
_FENCE_BLOCK_RE = re.compile(r"```[\w]*\n?([\s\S]*?)```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

# Error patterns inside text (outside or inside fences)
# Covers: SomeError, SomeException, FAILED, error:, exception:, traceback lines
_ERROR_LINE_RE = re.compile(
    r"(?:"
    r"[A-Z][a-zA-Z]*(?:Error|Exception|Failure|Warning|Fault)\b"  # PythonError
    r"|(?:Error|Exception|Traceback|FAILED|FATAL|CRITICAL)"        # bare keywords
    r"(?:\s*[:（(][^\n]*)?"                                         # optional detail
    r"|File\s+\"[^\"]+\",\s+line\s+\d+"                           # Python traceback frame
    r"|at\s+[A-Za-z_$][A-Za-z0-9_$.]+\([^)]*\)"                  # Java/JS stack frame
    r")",
    re.MULTILINE,
)

# dotted.module.paths and ALL_CAPS constants (keywords)
_DOTTED_NAME_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]+)+\b")
_ALL_CAPS_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")

# Sentence splitter (rough)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Behavioral signal lead phrases
_EXPECTED_PHRASES = re.compile(
    r"(?:^|\.\s+|\n)"                            # start of sentence
    r"([^.\n]*?"
    r"(?:expected|expect|should(?:\s+(?:be|return|work|have))?|"
    r"want(?:ed)?|suppose[d]?\s+to|intend(?:ed)?)"
    r"[^.\n]*\.?)",
    re.IGNORECASE | re.MULTILINE,
)

_OBSERVED_PHRASES = re.compile(
    r"(?:^|\.\s+|\n)"
    r"([^.\n]*?"
    r"(?:actual(?:ly)?|instead|however|but\s+(?:it|I|the)|"
    r"(?:it\s+)?(?:throws|raises|returns|crashes|fails|fetches|errors?)|"
    r"(?:got|getting|seeing|see)\s+(?:an?\s+)?(?:error|exception|traceback|failure))"
    r"[^.\n]*\.?)",
    re.IGNORECASE | re.MULTILINE,
)

# Markdown section headings that signal behavioral content
_OBSERVED_HEADING_RE = re.compile(
    r"##\s*(?:actual\s*(?:behavior)?|observed(?:\s*behavior)?|what\s*(?:actually\s*)?happens?)"
    r"\s*\n+([^\n#][^\n]*(?:\n(?!#)[^\n]+)*)",
    re.IGNORECASE,
)
_EXPECTED_HEADING_RE = re.compile(
    r"##\s*(?:expected(?:\s*behavior)?|what\s*(?:I\s*)?expected?)"
    r"\s*\n+([^\n#][^\n]*(?:\n(?!#)[^\n]+)*)",
    re.IGNORECASE,
)

# Noise words to exclude from keyword lists
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "not", "in", "on", "at", "to", "of",
    "for", "with", "is", "it", "as", "by", "be", "are", "was", "were",
    "this", "that", "which", "when", "if", "from", "have", "has", "do",
    "does", "can", "will", "would", "should", "could", "may", "might",
    "i", "my", "me", "we", "our", "you", "your", "he", "she", "they",
    "his", "her", "its", "their", "any", "all", "no", "so", "but", "get",
    "set", "use", "used", "using", "also", "just", "up", "out", "into",
    "about", "after", "before", "than", "then", "there", "here", "what",
    "how", "why", "true", "false", "null", "none", "new", "type", "value",
    "return", "returns", "pass", "via", "per", "etc", "e", "g", "s",
    "import", "class", "def", "self", "super", "object",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_text(issue: IssueDetail) -> tuple[str, str]:
    """Return (title, full_body) where full_body includes all comments."""
    parts: list[str] = []
    if issue.body:
        parts.append(issue.body)
    for comment in issue.comment_list:
        if comment.body:
            parts.append(comment.body)
    return issue.title, "\n\n".join(parts)


def _strip_fences(text: str) -> tuple[list[str], str]:
    """Extract code-fence contents; return (fence_texts, text_without_fences)."""
    fences: list[str] = []

    def _grab(m: re.Match) -> str:  # type: ignore[type-arg]
        fences.append(m.group(1))
        return " "

    clean = _FENCE_BLOCK_RE.sub(_grab, text)
    return fences, clean


def _dedupe_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _is_noise(token: str) -> bool:
    return (
        len(token) < 2
        or token.lower() in _STOPWORDS
        or token.isdigit()
    )


# ---------------------------------------------------------------------------
# Individual extractors
# ---------------------------------------------------------------------------

_TRACEBACK_FRAME_RE = re.compile(
    r'File\s+["\']([^"\']+\.py)["\'],\s+line\s+(\d+),\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)'
)
_CLASS_DEF_RE = re.compile(r"\bclass\s+([A-Z][a-zA-Z0-9_]*)")
_QUALIFIED_CLASS_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]*)\.[a-zA-Z_]")
_INLINE_TYPE_RE = re.compile(r"`([A-Z][a-zA-Z0-9_]*)`")


def _extract_file_references(texts: list[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for m in _TRACEBACK_FRAME_RE.finditer(text):
            found.append(m.group(1).strip())
        for m in _FILE_PATH_RE.finditer(text):
            path = m.group(0).strip("`").strip()
            if "/" in path or "." in path:
                found.append(path)
    return _dedupe_ordered(found)


def _extract_error_messages(fence_texts: list[str], full_text: str) -> list[str]:
    found: list[str] = []
    # Fence blocks are the richest source of error text
    for fence in fence_texts:
        for m in _ERROR_LINE_RE.finditer(fence):
            line = m.group(0).strip()
            if line:
                found.append(line)
    # Also scan inline code for error patterns
    for m in _INLINE_CODE_RE.finditer(full_text):
        snippet = m.group(1).strip()
        if _ERROR_LINE_RE.search(snippet):
            found.append(snippet)
    # Scan prose for standalone error mentions
    for m in _ERROR_LINE_RE.finditer(full_text):
        line = m.group(0).strip()
        if line and len(line) > 4:
            found.append(line)
    return _dedupe_ordered(found)


def _extract_class_names(texts: list[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for m in _CAMEL_CLASS_RE.finditer(text):
            name = m.group(0)
            if not _is_noise(name):
                found.append(name)
        for m in _CLASS_DEF_RE.finditer(text):
            name = m.group(1)
            if not _is_noise(name):
                found.append(name)
        for m in _QUALIFIED_CLASS_RE.finditer(text):
            name = m.group(1)
            if not _is_noise(name):
                found.append(name)
        for m in _INLINE_TYPE_RE.finditer(text):
            name = m.group(1)
            if (
                not _is_noise(name)
                and any(c.islower() for c in name)
                and name not in ("True", "False", "None")
            ):
                found.append(name)
    return _dedupe_ordered(found)


def _extract_function_names(texts: list[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for m in _TRACEBACK_FRAME_RE.finditer(text):
            fn = m.group(3)
            if not _is_noise(fn) and fn != "<module>":
                found.append(fn)
        # snake_case multi-segment names: parse_config, handle_request
        for m in _SNAKE_FUNC_RE.finditer(text):
            name = m.group(1)
            if not _is_noise(name):
                found.append(name)
        # lowerCamelCase names: useDataFetcher, parseConfig
        for m in _LOWER_CAMEL_RE.finditer(text):
            name = m.group(1)
            if not _is_noise(name):
                found.append(name)
        # explicit call syntax: name(
        for m in _FUNC_CALL_RE.finditer(text):
            name = m.group(1)
            if not _is_noise(name) and name.lower() not in _STOPWORDS:
                found.append(name)
    return _dedupe_ordered(found)


def _extract_keywords(title: str, texts: list[str], exclude: set[str]) -> list[str]:
    """Extract technical tokens from all text — title tokens get priority."""
    found: list[str] = []

    # Title words always come first
    for word in re.split(r"[\s/\\,;:()\[\]\"']+", title):
        word = word.strip("`.!?")
        if word and not _is_noise(word) and word not in exclude:
            found.append(word)

    for text in texts:
        # dotted names (module.submodule, obj.method)
        for m in _DOTTED_NAME_RE.finditer(text):
            token = m.group(0)
            if not _is_noise(token) and token not in exclude:
                found.append(token)
        # ALL_CAPS constants
        for m in _ALL_CAPS_RE.finditer(text):
            token = m.group(0)
            if not _is_noise(token) and token not in exclude:
                found.append(token)
        # inline code tokens
        for m in _INLINE_CODE_RE.finditer(text):
            for word in re.split(r"[\s/\\.,;:()\[\]\"']+", m.group(1)):
                word = word.strip("`!?")
                if word and not _is_noise(word) and word not in exclude:
                    found.append(word)

    return _dedupe_ordered(found)


def _extract_behavior(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (expected, observed) sentence snippets or None.

    Tries structured markdown headings first (## Expected behavior, ## Actual
    behavior), then falls back to sentence-level heuristics.
    """
    expected: Optional[str] = None
    observed: Optional[str] = None

    # Heading-based extraction (highest precision)
    m = _EXPECTED_HEADING_RE.search(text)
    if m:
        expected = m.group(1).strip().splitlines()[0].strip()

    m = _OBSERVED_HEADING_RE.search(text)
    if m:
        observed = m.group(1).strip().splitlines()[0].strip()

    # Sentence-level fallback
    if expected is None:
        m = _EXPECTED_PHRASES.search(text)
        if m:
            expected = m.group(1).strip().rstrip(".")

    if observed is None:
        m = _OBSERVED_PHRASES.search(text)
        if m:
            observed = m.group(1).strip().rstrip(".")

    return expected, observed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_signals(issue: IssueDetail) -> IssueSignals:
    """Extract structured signals from *issue* using deterministic processing.

    Parameters
    ----------
    issue:
        The GitHub issue to analyse.

    Returns
    -------
    IssueSignals
        All signals extracted from the issue title, body, and comments.
        No field is ever ``None`` except the behavioral heuristics.
    """
    title, body = _collect_text(issue)
    fence_texts, body_no_fences = _strip_fences(body)
    title_and_fences = [title] + fence_texts
    all_texts = [title, body]

    file_refs = _extract_file_references(all_texts + fence_texts)
    error_msgs = _extract_error_messages(fence_texts, body_no_fences)
    class_names = _extract_class_names(title_and_fences + [body_no_fences])
    func_names = _extract_function_names(title_and_fences + [body_no_fences])

    # Build exclusion set so keywords don't just repeat class/func names
    already_named: set[str] = set(class_names + func_names)

    keywords = _extract_keywords(title, [body_no_fences] + fence_texts, already_named)

    expected, observed = _extract_behavior(body_no_fences)

    return IssueSignals(
        title=title,
        keywords=keywords,
        error_messages=error_msgs,
        file_references=file_refs,
        function_names=func_names,
        class_names=class_names,
        expected_behavior=expected,
        observed_behavior=observed,
    )
