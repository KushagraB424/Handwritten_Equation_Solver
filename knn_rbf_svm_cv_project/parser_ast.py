import re
import sympy as sp
from typing import List


EQUAL_LIKE_PAIRS = {
    ('-', '-'),
    ('-', '/'),
    ('/', '-'),
    ('/', '/'),
}


def normalize_symbols(symbols: List[dict]) -> List[dict]:
    normalized = []
    i = 0
    while i < len(symbols):
        current = symbols[i]
        if i + 1 < len(symbols):
            nxt = symbols[i + 1]
            pair = (current.get('char'), nxt.get('char'))
            if pair in EQUAL_LIKE_PAIRS:
                merged = current.copy()
                merged['char'] = '='
                merged['is_superscript'] = bool(current.get('is_superscript') or nxt.get('is_superscript'))
                normalized.append(merged)
                i += 2
                continue
        normalized.append(current)
        i += 1
    return normalized


def insert_implied_multiplication(expr: str) -> str:
    # Between digit and variable or ')' and variable/digit
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
    expr = re.sub(r'(\))([a-zA-Z\d])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)
    return expr


def build_sequence(symbols: List[dict]) -> str:
    normalized_symbols = normalize_symbols(symbols)
    out = []
    for item in normalized_symbols:
        ch = item['char']
        if item.get('is_superscript'):
            out.append('^')
        out.append(ch)
    s = ''.join(out)
    s = insert_implied_multiplication(s)
    return s


def parse_to_sympy(seq: str):
    if '=' in seq:
        left, right = seq.split('=', 1)
        left = insert_implied_multiplication(left)
        right = insert_implied_multiplication(right)
        x = sp.Symbol('x')
        try:
            L = sp.sympify(left)
            R = sp.sympify(right)
            return sp.Eq(L, R)
        except Exception:
            return None
    else:
        try:
            return sp.sympify(seq)
        except Exception:
            return None