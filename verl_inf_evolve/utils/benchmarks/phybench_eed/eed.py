"""PHYBench EED scoring implementation vendored from the official repo."""

from __future__ import annotations

import logging
import os
import threading

from sympy import Add, Float, Function, Integer, Mul, Pow, Rational, Symbol
from sympy import expand, posify, simplify
from sympy.core.numbers import Exp1, Infinity, NegativeInfinity, Pi

from .extended_zss import ext_distance
from .latex_pre_process import master_convert

LOG = logging.getLogger(__name__)

INSERT_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}
DELETE_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}
UPDATE_COST = {"number": 1, "symbol": 1, "operator": 1, "function": 1}
CHANGE_TYPE_COST = 1
BAR_SIZE = 5
DISCOUNT_SLOPE = 0.6
SIMPLIFY_TIMEOUT_S = float(os.environ.get("VERL_PHYBENCH_EED_SIMPLIFY_TIMEOUT_S", "10.0"))
EQUALS_TIMEOUT_S = float(os.environ.get("VERL_PHYBENCH_EED_EQUALS_TIMEOUT_S", "5.0"))


class LaTeXError(Exception):
    pass


class SymPyError(Exception):
    pass


class DistError(Exception):
    pass


class TreeNode:
    def __init__(self, label, children=None, node_type="other"):
        self.label = label
        self.children = children if children is not None else []
        self.node_type = node_type
        self.subtree_size = 0

    def get_children(self):
        return self.children

    def __str__(self):
        return self.label


def _run_with_timeout(timeout_s: float, default, func, *args):
    """Run *func* with a thread-based timeout.

    Spawns a daemon thread so the main thread is never blocked by pathological
    symbolic computation.  Unlike the previous SIGALRM/setitimer approach, this
    cannot cause process-fatal signals (exit code 14) from stray alarms.

    If the function doesn't finish in time, the daemon thread is abandoned
    (cleaned up on process exit).  While daemon threads can't preempt
    GIL-holding C extensions, the abandoned thread is harmless and this avoids
    the deadly SIGALRM race that killed previous sol_eval runs.
    """
    if timeout_s <= 0:
        try:
            return func(*args)
        except Exception:
            return default

    result_box: list = [default]
    error_box: list = [None]

    def _target():
        try:
            result_box[0] = func(*args)
        except Exception as exc:
            error_box[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        LOG.warning("PHYBench EED timed out after %.2fs", timeout_s)
        return default

    if error_box[0] is not None:
        return default

    return result_box[0]


def _time_simplify(expr):
    return _run_with_timeout(SIMPLIFY_TIMEOUT_S, expr, simplify, expr)


def _time_equal(expr1, expr2):
    return _run_with_timeout(EQUALS_TIMEOUT_S, False, expr1.equals, expr2)


def update_func(node_a, node_b):
    if node_a.label == node_b.label:
        return 0
    if node_a.label.split("_")[0] == node_b.label.split("_")[0]:
        return UPDATE_COST[node_a.label.split("_")[0]]
    return CHANGE_TYPE_COST


def remove_func(node):
    return DELETE_COST[node.label.split("_")[0]]


def calc_tree_size(node):
    total = INSERT_COST[node.label.split("_")[0]]
    if node.children and node.subtree_size != 0:
        return node.subtree_size
    for child in node.children:
        total += calc_tree_size(child)
    node.subtree_size = total
    return total


def remove_tree_func(node):
    if not node.children:
        return remove_func(node)
    size = calc_tree_size(node)
    return min(size, DISCOUNT_SLOPE * (size - BAR_SIZE) + BAR_SIZE)


def insert_func(node):
    return INSERT_COST[node.label.split("_")[0]]


def insert_tree_func(node):
    return remove_tree_func(node)


def score_calc(tree_dist, tree_size):
    if tree_dist == 0.0:
        return 100.0
    return max(0.0, 100 * DISCOUNT_SLOPE - 100 * tree_dist / tree_size)


def sympy_to_tree(expr):
    if isinstance(expr, (Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity)):
        return TreeNode(label="number_" + str(expr), children=[])
    if isinstance(expr, Symbol):
        return TreeNode(label="symbol_" + str(expr), children=[])
    if isinstance(expr, (Add, Mul, Pow)):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="operator_" + op_name, children=children)
    if isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="function_" + func_name, children=children)
    raise ValueError(f"Unsupported SymPy type: {type(expr)}")


def compute_eed(
    answer_latex: str,
    test_latex: str,
    debug_mode: bool = False,
) -> tuple[float, float, int, float]:
    """Return ``(score, relative_distance, tree_size, raw_distance)``."""
    if not test_latex:
        return 0.0, -1.0, -1, -1.0
    if "\\int" in test_latex or "\\int" in answer_latex:
        return 0.0, -1.0, -1, -1.0
    if "\\sum" in test_latex or "\\sum" in answer_latex:
        return 0.0, -1.0, -1, 1.0
    if answer_latex == test_latex:
        return 100.0, 0.0, -1, 0.0
    if len(test_latex) > 3 * len(answer_latex):
        return 0.0, -1.0, -1, -1.0

    try:
        answer_exp = master_convert(answer_latex)
        test_exp = master_convert(test_latex)
    except Exception as exc:
        if debug_mode:
            raise LaTeXError(
                f"Fail to convert latex.\nGT:{answer_latex}\nGEN:{test_latex}"
            ) from exc
        return 0.0, -1.0, -1, -1.0

    try:
        answer_exp, rep1 = posify(answer_exp)
        answer_exp = _time_simplify(answer_exp)
        test_exp, rep2 = posify(test_exp)
        test_exp = _time_simplify(test_exp)
        answer_exp = answer_exp.subs(rep1)
        test_exp = test_exp.subs(rep2)
        zero_exp = _time_simplify(expand(answer_exp - test_exp))

        if answer_exp == test_exp or zero_exp == 0:
            return 100.0, 0.0, 0, 0.0
        if _time_equal(answer_exp, test_exp):
            return 100.0, 0.0, 0, 0.0
    except Exception as exc:
        if debug_mode:
            raise SymPyError(
                "Failed to simplify the sympy expression. "
                f"Expressions: answer_exp={answer_exp}, test_exp={test_exp}"
            ) from exc
        return 0.0, -1.0, -1, -1.0

    try:
        tree_answer = sympy_to_tree(answer_exp)
        tree_test = sympy_to_tree(test_exp)
    except Exception as exc:
        if debug_mode:
            raise SymPyError(
                f"Failed to build the sympy expression tree.\nGT:{answer_exp}\nGEN:{test_exp}"
            ) from exc
        return 0.0, -1.0, -1, -1.0

    try:
        distance = ext_distance(
            tree_test,
            tree_answer,
            get_children=lambda node: node.get_children(),
            single_insert_cost=insert_func,
            insert_cost=insert_tree_func,
            single_remove_cost=remove_func,
            remove_cost=remove_tree_func,
            update_cost=update_func,
        )
    except Exception as exc:
        if debug_mode:
            raise DistError(
                "Failed to calculate the distance between trees.\n"
                f"GT:{answer_latex}\nGEN:{test_latex}"
            ) from exc
        return 0.0, -1.0, calc_tree_size(tree_answer), -1.0

    tree_size = calc_tree_size(tree_answer)
    rel_distance = distance / tree_size
    score = score_calc(distance, tree_size)
    return score, rel_distance, tree_size, distance


def compute_eed_score(answer_latex: str, test_latex: str) -> float:
    """Return the official PHYBench EED score on the native ``0..100`` scale."""
    score, _rel, _tree_size, _distance = compute_eed(answer_latex, test_latex)
    return float(score)
