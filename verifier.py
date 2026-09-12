"""Exact numeric evaluation; never accept a numeric expression's prefix."""
from fractions import Fraction
import re

_NUMBER = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)')
_MARKER = re.compile(r'####|\\boxed\{')


def parse_number(text):
    """Parse a complete number or simple fraction, including LaTeX fractions."""
    if not isinstance(text, str):
        return None
    text = text.strip().removeprefix('$').strip()
    latex = re.fullmatch(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', text)
    if latex:
        text = latex[1] + '/' + latex[2]
    parts = text.split('/')
    if len(parts) > 2 or any(not _NUMBER.fullmatch(p.strip()) for p in parts):
        return None
    try:
        values = [Fraction(p.strip().replace(',', '')) for p in parts]
        return values[0] if len(values) == 1 else values[0] / values[1]
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text, lenient=False):
    """Last explicit marker wins; malformed explicit answers never use fallback.

    #### must terminate the completion; boxed answers require balanced braces.
    Lenient last-number fallback is diagnostic only.
    """
    if not isinstance(text, str):
        return None
    markers = list(_MARKER.finditer(text))
    if markers:
        marker = markers[-1]
        tail = text[marker.end():].strip()
        if marker[0] == '####':
            return parse_number(tail)
        depth = 1
        for i, char in enumerate(tail):
            depth += (char == '{') - (char == '}')
            if depth == 0:
                return parse_number(tail[:i])
        return None
    if lenient:
        numbers = list(_NUMBER.finditer(text))
        return parse_number(numbers[-1][0]) if numbers else None
    return None


def gold_answer(text):
    value = extract_answer(text) if _MARKER.search(text) else parse_number(text)
    if value is None:
        raise ValueError('Unparseable gold answer')
    return value


def is_correct(completion, gold, lenient=False):
    gold = gold_answer(gold) if isinstance(gold, str) else Fraction(str(gold))
    pred = extract_answer(completion, lenient=lenient)
    return pred is not None and pred == gold
