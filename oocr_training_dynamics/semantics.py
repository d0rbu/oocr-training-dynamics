"""Safe semantic scoring for generated one-argument lambda definitions."""

from __future__ import annotations

import ast

from beartype import beartype

from oocr_training_dynamics.data import evaluate_function


def _evaluate_expression(node: ast.expr, value: int, variable_name: str) -> int | float | bool:
    if isinstance(node, ast.Name) and node.id == variable_name:
        return value
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _evaluate_expression(node.operand, value, variable_name)
        if isinstance(operand, bool):
            raise ValueError("boolean negation is outside the function suite")
        return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_expression(node.left, value, variable_name)
        right = _evaluate_expression(node.right, value, variable_name)
        if isinstance(left, bool) or isinstance(right, bool):
            raise ValueError("boolean arithmetic is outside the function suite")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("unsupported binary operator")
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left = _evaluate_expression(node.left, value, variable_name)
        right = _evaluate_expression(node.comparators[0], value, variable_name)
        if isinstance(node.ops[0], ast.GtE):
            return left >= right
        if isinstance(node.ops[0], ast.Eq):
            return left == right
        raise ValueError("unsupported comparison")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
        and len(node.args) == 2
        and not node.keywords
    ):
        return max(
            _evaluate_expression(node.args[0], value, variable_name),
            _evaluate_expression(node.args[1], value, variable_name),
        )
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _strip_lambda(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines.pop()
        stripped = "\n".join(lines).strip()
    return stripped.splitlines()[0].strip() if stripped else ""


@beartype
def generated_lambda_matches(function_id: str, generated: str) -> bool:
    try:
        parsed = ast.parse(_strip_lambda(generated), mode="eval")
        if not isinstance(parsed.body, ast.Lambda):
            return False
        node = parsed.body
        if (
            len(node.args.args) != 1
            or node.args.posonlyargs
            or node.args.kwonlyargs
            or node.args.defaults
            or node.args.kw_defaults
            or node.args.vararg is not None
            or node.args.kwarg is not None
        ):
            return False
        variable = node.args.args[0].arg
        return all(
            _evaluate_expression(node.body, value, variable) == evaluate_function(function_id, value)
            for value in (-1_044, -294, -23, -9, -2, -1, 0, 1, 2, 3, 6, 45, 473, 1_023)
        )
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["generated_lambda_matches"]
