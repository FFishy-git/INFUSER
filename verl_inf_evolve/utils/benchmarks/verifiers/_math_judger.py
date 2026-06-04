"""Sympy-based math answer comparison for OlympiadBench-style problems.

Ported from the OlympiadBench evaluation toolkit.  All parsing is wrapped
in try/except so failures silently fall through to exact-string comparison
— never worse than the baseline ``is_correct_general()``.
"""

from __future__ import annotations

import re
from typing import Optional


def _can_compute_power(expr_str: str, max_exponent: int = 100) -> bool:
    """Guard against pathological power expressions that hang sympy."""
    # Look for ** or ^ with large exponents
    for pattern in [r'\*\*\s*(\d+)', r'\^\s*(\d+)']:
        for m in re.finditer(pattern, expr_str):
            if int(m.group(1)) > max_exponent:
                return False
    return True


def _split_by_comma(text: str) -> list[str]:
    """Split *text* by commas, respecting brackets and parentheses.

    ``"1, (2, 3), 4"`` -> ``["1", "(2, 3)", "4"]``
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in text:
        if ch in '([{':
            depth += 1
            current.append(ch)
        elif ch in ')]}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    remaining = ''.join(current).strip()
    if remaining:
        parts.append(remaining)

    return parts


def _clean_latex(text: str) -> str:
    """Strip common LaTeX noise that confuses sympy."""
    s = text.strip()
    # Strip surrounding $ signs
    if s.startswith('$') and s.endswith('$'):
        s = s[1:-1].strip()
    # Normalize \dfrac -> \frac (display-style fraction is semantically identical)
    s = s.replace(r'\dfrac', r'\frac')
    # Expand compact \frac notation: \frac43 -> \frac{4}{3}, \frac 59 -> \frac{5}{9}
    s = re.sub(r'\\frac\s*([^{])\s*([^{])', r'\\frac{\1}{\2}', s)
    # Normalize ^{x} -> ^x and _{x} -> _x for single-char exponents/subscripts
    s = re.sub(r'\^{([^}])}', r'^\1', s)
    s = re.sub(r'_{([^}])}', r'_\1', s)
    # Remove \text{...}, \mathrm{...}, etc.
    s = re.sub(r'\\(?:text|textbf|textit|textrm|mathrm|mathbf)\{([^}]*)\}', r'\1', s)
    # Remove trailing percent sign and \% (e.g. "10\%" or "10%" -> "10")
    s = re.sub(r'\\%$', '', s)
    s = re.sub(r'%$', '', s)
    # Remove ordinal suffixes (e.g. "12^{th}" -> "12", "12^th" -> "12")
    s = re.sub(r'\^\{?(?:st|nd|rd|th)\}?(?:\s+(?:grade|place|row))?$', '', s, flags=re.IGNORECASE)
    # Remove \left and \right
    s = s.replace(r'\left', '').replace(r'\right', '')
    # Remove \, spacing
    s = s.replace(r'\,', ' ')
    # Normalize whitespace
    s = ' '.join(s.split())
    return s


def _parse_number(text: str) -> Optional[float]:
    """Parse a number, handling percentage notation (e.g. '62.5%' -> 0.625)."""
    s = text.strip()
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except (ValueError, TypeError):
            return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def numerical_equal(
    predicted: str, ground_truth: str, precision: float = 1e-4
) -> Optional[bool]:
    """Compare as floats with tolerance. Handles percentage notation."""
    p = _parse_number(predicted)
    g = _parse_number(ground_truth)
    if p is None or g is None:
        return None

    if g == 0:
        return abs(p) < precision
    return abs(p - g) / max(abs(g), 1e-15) < precision


def expression_equal(predicted: str, ground_truth: str) -> Optional[bool]:
    """Compare as symbolic expressions via ``sympy.simplify(a - b) == 0``.

    Uses ``latex2sympy2_extended`` (bundled with ``math-verify``) for LaTeX
    parsing, which is more robust than ``sympy.parsing.latex.parse_latex``
    and works with the antlr4 4.9.x runtime required by hydra/omegaconf.
    """
    try:
        import sympy
        from latex2sympy2_extended import latex2sympy
    except ImportError:
        return None

    # Guard against pathological expressions
    if not _can_compute_power(predicted) or not _can_compute_power(ground_truth):
        return None

    try:
        p_expr = latex2sympy(_clean_latex(predicted))
        g_expr = latex2sympy(_clean_latex(ground_truth))
    except Exception:
        return None

    try:
        diff = sympy.simplify(p_expr - g_expr)
        if diff == 0:
            return True
        # Try numerical evaluation as fallback
        diff_val = complex(diff.evalf())
        if abs(diff_val) < 1e-6:
            return True
    except Exception:
        pass

    return None


def equation_equal(predicted: str, ground_truth: str) -> Optional[bool]:
    """Compare equations by normalizing both sides to 0 and checking ratio."""
    if '=' not in predicted or '=' not in ground_truth:
        return None

    try:
        import sympy
        from latex2sympy2_extended import latex2sympy
    except ImportError:
        return None

    def _to_zero_form(eq_str: str):
        """Parse ``lhs = rhs`` -> ``lhs - rhs``."""
        cleaned = _clean_latex(eq_str)
        parts = cleaned.split('=', 1)
        if len(parts) != 2:
            return None
        lhs = latex2sympy(parts[0].strip())
        rhs = latex2sympy(parts[1].strip())
        return sympy.simplify(lhs - rhs)

    try:
        p_zero = _to_zero_form(predicted)
        g_zero = _to_zero_form(ground_truth)
        if p_zero is None or g_zero is None:
            return None

        # Check if they differ by a constant factor (integer ratio)
        ratio = sympy.simplify(p_zero / g_zero)
        if ratio.is_number and ratio.is_real and ratio != 0:
            return True
    except Exception:
        pass

    return None


def interval_equal(predicted: str, ground_truth: str) -> Optional[bool]:
    """Compare interval notation like ``[0, 1)`` or ``(-\\infty, 5]``."""
    interval_re = re.compile(
        r'^([(\[])(.+),(.+)([)\]])$'
    )

    def _parse_interval(s: str):
        s = _clean_latex(s).replace(' ', '')
        s = s.replace(r'\infty', 'oo').replace('\\infty', 'oo')
        m = interval_re.match(s)
        if not m:
            return None
        left_bracket, left_val, right_val, right_bracket = m.groups()
        return (left_bracket, left_val.strip(), right_val.strip(), right_bracket)

    p_interval = _parse_interval(predicted)
    g_interval = _parse_interval(ground_truth)

    if p_interval is None or g_interval is None:
        return None

    # Compare brackets
    if p_interval[0] != g_interval[0] or p_interval[3] != g_interval[3]:
        return False

    # Compare endpoints numerically
    for p_val, g_val in [(p_interval[1], g_interval[1]), (p_interval[2], g_interval[2])]:
        if p_val == g_val:
            continue
        num_eq = numerical_equal(p_val, g_val)
        if num_eq is None or num_eq is False:
            return False

    return True


def judge(
    predicted: str,
    ground_truth: str,
    answer_type: str = "",
    precision: float = 1e-4,
) -> Optional[bool]:
    """Main entry point: cascading comparison for math answers.

    Tries methods in order of reliability:
    1. Exact string match (case-insensitive, after cleaning)
    2. Numerical comparison (float tolerance)
    3. Expression-level symbolic comparison
    4. Equation comparison (for equation answer_type)
    5. Interval comparison (for interval answer_type)

    For multi-answer ground truths (separated by ";"), all parts must match.

    Returns ``True`` if equivalent, ``False`` if definitely wrong,
    ``None`` if unable to determine.
    """
    if predicted is None:
        return None

    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    if not predicted:
        return None

    # Handle multi-answer (semicolon-separated in ground_truth)
    if ';' in ground_truth:
        gt_parts = [p.strip() for p in ground_truth.split(';')]
        pred_parts = _split_by_comma(predicted)

        # Also try semicolon-split on predicted
        if len(pred_parts) != len(gt_parts):
            pred_parts = [p.strip() for p in predicted.split(';')]

        if len(pred_parts) != len(gt_parts):
            return False

        results = []
        for p, g in zip(pred_parts, gt_parts):
            r = judge(p.strip(), g.strip(), answer_type=answer_type, precision=precision)
            results.append(r)

        if all(r is True for r in results):
            return True
        if any(r is False for r in results):
            return False
        return None

    # 1. Exact string match (after cleaning)
    p_clean = _clean_latex(predicted).lower()
    g_clean = _clean_latex(ground_truth).lower()
    if p_clean == g_clean:
        return True

    # 1b. Exact match after stripping all non-essential spaces
    p_compact = re.sub(r'\s+', '', p_clean)
    g_compact = re.sub(r'\s+', '', g_clean)
    if p_compact == g_compact:
        return True

    # 2. Numerical comparison
    num_result = numerical_equal(predicted, ground_truth, precision)
    if num_result is True:
        return True

    # 3. Type-specific comparison
    answer_type_lower = answer_type.lower() if answer_type else ""

    if answer_type_lower == "interval":
        iv_result = interval_equal(predicted, ground_truth)
        if iv_result is not None:
            return iv_result

    # Try equation comparison when answer_type says so, OR when both sides
    # contain '=' (auto-detect equations even without metadata).
    if answer_type_lower == "equation" or ('=' in predicted and '=' in ground_truth):
        eq_result = equation_equal(predicted, ground_truth)
        if eq_result is not None:
            return eq_result

    # 4. General symbolic comparison (for Expression, Numerical with LaTeX, etc.)
    expr_result = expression_equal(predicted, ground_truth)
    if expr_result is True:
        return True

    # Could not determine — fall back
    return None
