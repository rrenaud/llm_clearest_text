"""Markdown-aware sentence and span segmentation with character offsets.

Shared by probe training and the server so that training units match rendering units.
"""
import re
from dataclasses import dataclass


@dataclass
class Segment:
    text: str
    start: int
    end: int


FENCE = re.compile(r"^(```|~~~)", re.M)
# Sentence end: . ! ? followed by space/newline, not after common abbreviations or digits like "3.5".
SENT_END = re.compile(r"(?<!\b[A-Z])(?<!e\.g)(?<!i\.e)(?<!etc)(?<!\bvs)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)([.!?])(?=\s+[\"'(*\[]?[A-Z0-9])")
LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#{1,6})\s+")


# LaTeX math rendered by KaTeX; a tag inside it breaks the formula. Inline $...$ must not
# start or end with a space, so prices like "$5 and $10" are not taken as math.
MATH = re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$(?=[^\s$])[^$\n]*?(?<=[^\s$])\$(?!\d)")
MATH_OPEN = re.compile(r"\$\$|\\\[|\\\(")


def skip_ranges(text):
    """Character ranges never annotated: fenced or inline code, and LaTeX math."""
    ranges = []
    fences = [m.start() for m in FENCE.finditer(text)]
    for a, b in zip(fences[::2], fences[1::2]):
        end = text.find("\n", b)
        ranges.append((a, len(text) if end < 0 else end))
    if len(fences) % 2:  # unterminated fence (still streaming)
        ranges.append((fences[-1], len(text)))
    for m in re.finditer(r"`[^`\n]+`", text):
        ranges.append((m.start(), m.end()))
    pos = 0
    for m in MATH.finditer(text):
        ranges.append((m.start(), m.end()))
        pos = m.end()
    # Display/paren math opened but not yet closed (still streaming): skip to the end.
    rest = [m for m in MATH_OPEN.finditer(text, pos) if not in_ranges(m.start(), ranges)]
    if rest:
        ranges.append((rest[0].start(), len(text)))
    return ranges


def in_ranges(i, ranges):
    return any(a <= i < b for a, b in ranges)


def split_sentences(text):
    """Split prose into sentence segments, skipping code blocks and blank lines.

    List bullets and heading markers are excluded from the segment text.
    """
    code = skip_ranges(text)
    segs = []
    pos = 0
    for line in text.splitlines(keepends=True):
        lstart, pos = pos, pos + len(line)
        body = line.rstrip("\n")
        if not body.strip() or in_ranges(lstart, code):
            continue
        m = LIST_PREFIX.match(body)
        off = m.end() if m else len(body) - len(body.lstrip())
        cuts = [off] + [x.end(1) for x in SENT_END.finditer(body) if x.end(1) > off] + [len(body)]
        for a, b in zip(cuts, cuts[1:]):
            chunk = body[a:b]
            lead = len(chunk) - len(chunk.lstrip())
            chunk = chunk.strip()
            start = lstart + a + lead
            end = start + len(chunk)
            # A tag must not open or close inside math, nor wrap display math.
            if not chunk or in_ranges(start, code) or in_ranges(end - 1, code):
                continue
            if any(start <= r0 and r1 <= end and text[r0:r0 + 2] in ("$$", "\\[") for r0, r1 in code):
                continue
            segs.append(Segment(chunk, start, end))
    return segs


# Candidate factual spans: numbers (with units/percent), capitalized multiword names, quoted titles.
SPAN_RE = re.compile(
    r"(?:[$£€]?\d[\d,]*(?:\.\d+)?(?:\s?(?:%|percent|million|billion|thousand|km|kg|mg|miles|years?|BC|AD|BCE|CE))?)"
    r"|(?:\b[A-Z][a-zA-Z'\-]+(?:(?:\s+(?:of|the|de|du|la|le|von|van|der))*\s+(?:d')?[A-Z][a-zA-Z'\-]+)*)"
)
STOP_CAPS = {
    "I", "The", "A", "An", "This", "That", "These", "Those", "It", "If", "In", "On", "For", "And", "But", "Or",
    "However", "Also", "Yes", "No", "Here", "There", "When", "While", "As", "To", "You", "Your", "We", "They",
    "Some", "Many", "Most", "Each", "Every", "Be", "Do", "Don't", "Can", "What", "How", "Why", "Note", "First",
    "Second", "Third", "Finally", "Additionally", "Overall", "Sure", "Of", "With", "At", "By", "From", "Is", "Are",
    "Dr", "Mr", "Mrs", "Ms", "St", "Step", "Example", "Remember", "Consider", "Try", "Make", "Use", "Keep",
}
# Sentence-initial capitalized words that are usually not names (checked lowercased, only at sentence start).
COMMON_INITIAL = {
    "visit", "avoid", "start", "choose", "take", "get", "add", "check", "look", "find", "set", "go", "drink", "eat",
    "after", "before", "since", "because", "although", "though", "once", "then", "now", "today", "generally",
    "typically", "usually", "often", "sometimes", "both", "one", "another", "other", "such", "all", "any", "not",
}


def factual_spans(text):
    """Heuristic spans that carry checkable facts: numbers, dates and proper names."""
    out = []
    for seg in split_sentences(text):
        line_start = text.rfind("\n", 0, seg.start) + 1
        heading = text[line_start:seg.start].lstrip().startswith("#")
        for span in _spans_in(seg.text):
            # Headings ("### Personal Life", "### 8. Join a Community") name topics, not claims.
            if heading:
                continue
            out.append(Segment(span.text, seg.start + span.start, seg.start + span.end))
    return out


def _spans_in(text):
    code = skip_ranges(text)
    out = []
    for m in SPAN_RE.finditer(text):
        s = m.group(0).strip()
        if not s or in_ranges(m.start(), code) or in_ranges(m.end() - 1, code):
            continue
        if s in STOP_CAPS:
            continue
        # Sentence-initial single capitalized words ("**Ergonomics**:", "Visit ...") are rarely
        # factual claims; multiword names and numbers there are kept.
        initial = bool(re.search(r"(?:^|[:.!?][*_]*\s)[\s*_\"'(#>-]*$", text[: m.start()]))
        if initial and " " not in s and not s[0].isdigit() and not s[0] in "$£€":
            continue
        # Lone acronyms (GPS, MLB, ERA) are vocabulary, not checkable claims.
        if re.fullmatch(r"[A-Z]{2,5}", s):
            continue
        # "Heading:" labels in lists.
        if re.match(r"\**\s*:", text[m.end():]):
            continue
        # Drop a sentence-initial common word glued to a name: "The Eiffel Tower" -> keep; "However Paris" rare.
        start = m.start()
        # Strip a leading common word glued to a name: "The Eiffel Tower", "Visit the Louvre".
        mm = re.match(r"(\S+)((?:\s+(?:of|the|de|du|la|le|von|van|der))*\s+)", s)
        if mm and (mm.group(1) in STOP_CAPS or mm.group(1).lower() in COMMON_INITIAL):
            start += mm.end()
            s = s[mm.end() :]
        if len(s) < 2 and not s.isdigit():
            continue
        out.append(Segment(s, start, start + len(s)))
    return out
