"""Safe reward-formula compiler.

Formulas are plain maths over a game's metric names, e.g. ``10*caught - miss``.
They are parsed with a small recursive-descent parser into Python closures, so
nothing the user types is ever passed to ``eval``.
"""
import math
import re

_NUM = re.compile(r"\d*\.?\d+(?:[eE][-+]?\d+)?")
_NAME = re.compile(r"[A-Za-z_]\w*")


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


FUNCS = {
    "abs": (abs, 1, 1),
    "min": (min, 1, 99),
    "max": (max, 1, 99),
    "sqrt": (math.sqrt, 1, 1),
    "exp": (math.exp, 1, 1),
    "log": (math.log, 1, 1),
    "tanh": (math.tanh, 1, 1),
    "clip": (_clip, 3, 3),
}


class FormulaError(ValueError):
    pass


def compile_formula(src, names):
    """Return ``f(metrics: dict) -> float``. Raises FormulaError with a friendly message."""
    names = set(names)
    pos = 0

    def ws():
        nonlocal pos
        while pos < len(src) and src[pos] in " \t":
            pos += 1

    def peek():
        ws()
        return src[pos] if pos < len(src) else None

    def expr():
        nonlocal pos
        a = term()
        while True:
            c = peek()
            if c in ("+", "-"):
                pos += 1
                b, l = term(), a
                a = (lambda e, l=l, b=b: l(e) + b(e)) if c == "+" else (lambda e, l=l, b=b: l(e) - b(e))
            else:
                return a

    def term():
        nonlocal pos
        a = unary()
        while True:
            c = peek()
            if c in ("*", "/"):
                pos += 1
                b, l = unary(), a
                a = (lambda e, l=l, b=b: l(e) * b(e)) if c == "*" else (lambda e, l=l, b=b: l(e) / b(e))
            else:
                return a

    def unary():
        nonlocal pos
        c = peek()
        if c == "-":
            pos += 1
            a = unary()
            return lambda e: -a(e)
        if c == "+":
            pos += 1
            return unary()
        return power()

    def power():
        nonlocal pos
        a = atom()
        if peek() == "^":
            pos += 1
            b = unary()
            return lambda e: a(e) ** b(e)
        return a

    def atom():
        nonlocal pos
        c = peek()
        if c == "(":
            pos += 1
            a = expr()
            if peek() != ")":
                raise FormulaError("Missing a closing bracket.")
            pos += 1
            return a
        m = _NUM.match(src, pos)
        if m:
            pos = m.end()
            v = float(m.group())
            return lambda e: v
        m = _NAME.match(src, pos)
        if m:
            n = m.group()
            pos = m.end()
            if peek() == "(":
                if n not in FUNCS:
                    raise FormulaError(f'"{n}" is not a function. Try {", ".join(FUNCS)}.')
                pos += 1
                args = []
                if peek() != ")":
                    while True:
                        args.append(expr())
                        if peek() == ",":
                            pos += 1
                            continue
                        break
                if peek() != ")":
                    raise FormulaError("Missing a closing bracket.")
                pos += 1
                f, lo, hi = FUNCS[n]
                if not lo <= len(args) <= hi:
                    raise FormulaError(f'"{n}" takes {lo if lo == hi else f"{lo} or more"} argument{"s" if hi > 1 else ""}.')
                return lambda e: f(*(a(e) for a in args))
            if n not in names:
                raise FormulaError(f'"{n}" is not a variable in this game. Use the names in the table.')
            return lambda e: e[n]
        raise FormulaError("The formula ends too early." if c is None else f'Unexpected "{c}".')

    if not src.strip():
        raise FormulaError("Write a formula first.")
    f = expr()
    if peek() is not None:
        raise FormulaError(f'Unexpected "{src[pos]}".')

    def safe(metrics):
        try:
            v = f(metrics)
            v = float(v.real if isinstance(v, complex) else v)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return math.nan
        return v

    return safe
