from __future__ import annotations

import ast
import math
import operator
import re
from typing import Iterable


_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    'abs': abs,
    'max': max,
    'min': min,
    'round': round,
    'sqrt': math.sqrt,
}


def normalize_parameter_name(name: str) -> str:
    normalized = re.sub(r'[^0-9a-zA-Z]+', '_', name.strip().lower())
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized


def _evaluate_ast_node(node: ast.AST, resolve_symbol) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_ast_node(node.body, resolve_symbol)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f'Unsupported literal in formula: {node.value!r}')
        return float(node.value)

    if isinstance(node, ast.Name):
        return float(resolve_symbol(node.id))

    if isinstance(node, ast.BinOp):
        operator_fn = _ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f'Unsupported operator in formula: {ast.dump(node.op)}')
        return operator_fn(_evaluate_ast_node(node.left, resolve_symbol), _evaluate_ast_node(node.right, resolve_symbol))

    if isinstance(node, ast.UnaryOp):
        operator_fn = _ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operator_fn is None:
            raise ValueError(f'Unsupported unary operator in formula: {ast.dump(node.op)}')
        return operator_fn(_evaluate_ast_node(node.operand, resolve_symbol))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise ValueError('Unsupported function in formula.')
        args = [_evaluate_ast_node(arg, resolve_symbol) for arg in node.args]
        return float(_ALLOWED_FUNCTIONS[node.func.id](*args))

    raise ValueError(f'Unsupported formula syntax: {ast.dump(node)}')


def evaluate_formula_expression(expression: str, resolve_symbol) -> float:
    parsed = ast.parse(expression, mode='eval')
    value = _evaluate_ast_node(parsed, resolve_symbol)
    if not math.isfinite(value):
        raise ValueError(f'Formula did not evaluate to a finite number: {expression}')
    return value


def _is_numeric_parameter(parameter) -> bool:
    from geophires_x.Parameter import floatParameter
    from geophires_x.Parameter import intParameter

    return isinstance(parameter, (intParameter, floatParameter))


def _assign_resolved_value(parameter, value: float) -> None:
    from geophires_x.Parameter import floatParameter
    from geophires_x.Parameter import intParameter

    if isinstance(parameter, intParameter):
        resolved_value = int(float(value))
        if len(parameter.AllowableRange) > 0 and resolved_value not in parameter.AllowableRange:
            raise ValueError(
                f'Error: Parameter given ({resolved_value}) for {parameter.Name} outside of valid range.'
            )
        parameter.value = resolved_value
        parameter.coerce_value_to_enum()
    elif isinstance(parameter, floatParameter):
        resolved_value = float(value)
        if resolved_value < float(parameter.Min) or resolved_value > float(parameter.Max):
            raise ValueError(
                f'Error: Parameter given ({resolved_value}) for {parameter.Name} outside of valid range.'
            )
        parameter.value = resolved_value
    else:
        raise TypeError(f'Unsupported formula parameter type for {parameter.Name}')

    parameter.Provided = True
    parameter.Valid = True
    parameter.EvaluatedFromFormula = True


def resolve_parameter_formulas(parameters: Iterable, logger) -> None:
    parameters = list(parameters)
    numeric_parameters = [parameter for parameter in parameters if _is_numeric_parameter(parameter)]
    normalized_lookup = {}

    for parameter in numeric_parameters:
        normalized_name = normalize_parameter_name(parameter.Name)
        existing = normalized_lookup.get(normalized_name)
        if existing is not None and existing is not parameter:
            raise ValueError(
                f'Ambiguous formula symbol "{normalized_name}" for parameters {existing.Name} and {parameter.Name}.'
            )
        normalized_lookup[normalized_name] = parameter

    resolved_values = {}
    resolving = []

    def resolve_symbol(symbol_name: str) -> float:
        normalized_symbol = normalize_parameter_name(symbol_name)

        if normalized_symbol in resolved_values:
            return resolved_values[normalized_symbol]

        parameter = normalized_lookup.get(normalized_symbol)
        if parameter is None:
            raise ValueError(f'Unknown formula symbol "{symbol_name}".')

        if normalized_symbol in resolving:
            cycle = ' -> '.join([*resolving, normalized_symbol])
            raise ValueError(f'Circular formula dependency detected: {cycle}')

        if parameter.FormulaExpression:
            resolving.append(normalized_symbol)
            try:
                value = evaluate_formula_expression(parameter.FormulaExpression, resolve_symbol)
                _assign_resolved_value(parameter, value)
                resolved_values[normalized_symbol] = float(parameter.value)
                logger.info(f'Resolved formula for {parameter.Name}: {parameter.FormulaExpression} -> {parameter.value}')
            except ValueError as exc:
                parameter.Valid = False
                raise ValueError(
                    f'Failed to resolve formula for {parameter.Name} ({parameter.FormulaExpression}): {exc}'
                ) from exc
            finally:
                resolving.pop()
        else:
            resolved_values[normalized_symbol] = float(parameter.value)

        return resolved_values[normalized_symbol]

    for parameter in numeric_parameters:
        if parameter.FormulaExpression:
            resolve_symbol(parameter.Name)


def _iter_model_parameters(model) -> Iterable:
    for attribute_name in (
        'reserv',
        'wellbores',
        'surfaceplant',
        'economics',
        'outputs',
        'addeconomics',
        'addoutputs',
        'sdacgteconomics',
        'sdacgtoutputs',
    ):
        component = getattr(model, attribute_name, None)
        if component is None:
            continue

        parameter_dict = getattr(component, 'ParameterDict', None)
        if parameter_dict is None:
            continue

        yield from parameter_dict.values()


def resolve_model_parameter_formulas(model) -> None:
    resolve_parameter_formulas(_iter_model_parameters(model), model.logger)
