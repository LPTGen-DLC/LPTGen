"""Conservative, intraprocedural configuration-predicate extraction (stdlib only).

No analyzed code is executed. Output expressions describe tensor metadata, not
the literal [shape, dtype] serialization. Unknown calls and effects are barriers.
Run: python lowering_def_use.py source.py --function lowering_name
"""
import argparse
import ast
import copy
import inspect
import json
import logging
import os
from pathlib import Path
import textwrap


class Unresolved(ValueError):
    pass


class Binding:
    def __init__(self, expression, slots=(), kind="attribute"):
        self.expression = expression
        self.slots = frozenset(slots)
        self.kind = kind


class Analyzer:
    # Only allow metadata reads.
    METADATA = {"get_size": "shape", "size": "shape", "get_dtype": "dtype",
                "dim": "ndim", "ndimension": "ndim"}
    ATTRIBUTES = {"shape", "dtype", "ndim"}
    BUILTINS = {"len", "abs", "min", "max", "int", "bool", "float", "pow"}
    DTYPES = {"float16", "bfloat16", "float32", "float64", "int8", "int16",
              "int32", "int64", "uint8", "bool", "complex64", "complex128"}

    def __init__(self, function, configuration_slots=None):
        args = function.args
        parameters = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        if configuration_slots is None:
            configuration_slots = {p: "attribute" for p in parameters if p not in {"self", "cls"}}
        elif not isinstance(configuration_slots, dict):
            configuration_slots = {p: "attribute" for p in configuration_slots}
        self.env = {
            p: Binding(ast.Name(id=p, ctx=ast.Load()), [p], configuration_slots[p])
            for p in parameters if p in configuration_slots
        }
        self.predicates = []
        self.skipped = []
        self.function = function
        self.local_names = set(parameters) | {
            n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    def resolve(self, node, env):
        value = self._resolve(node, env)
        if sum(1 for _ in ast.walk(value.expression)) > 512:
            raise Unresolved("expanded expression exceeds 512 AST nodes")
        return value

    def _resolve(self, node, env):
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise Unresolved("unknown or invalidated name: " + node.id)
            return env[node.id]
        if isinstance(node, ast.Constant):
            return Binding(copy.deepcopy(node))
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name) and node.value.id == "torch"
                    and node.value.id not in self.local_names and node.attr in self.DTYPES):
                return Binding(copy.deepcopy(node))
            base = self.resolve(node.value, env)
            if node.attr not in self.ATTRIBUTES or base.kind not in {"tensor", "attribute"}:
                raise Unresolved("unsupported attribute: " + node.attr)
            return Binding(ast.Attribute(value=copy.deepcopy(base.expression), attr=node.attr,
                                         ctx=ast.Load()), base.slots, "metadata")
        if isinstance(node, ast.Call):
            if node.keywords:
                raise Unresolved("keyword calls are not supported")
            if isinstance(node.func, ast.Attribute):
                base = self.resolve(node.func.value, env)
                method = node.func.attr
                if method in self.METADATA and base.kind in {"tensor", "attribute"}:
                    if node.args and method not in {"size", "get_size"}:
                        raise Unresolved("unexpected metadata method arguments")
                    if len(node.args) > 1:
                        raise Unresolved("unexpected size arguments")
                    result = Binding(ast.Attribute(value=copy.deepcopy(base.expression),
                                     attr=self.METADATA[method], ctx=ast.Load()), base.slots, "metadata")
                    if node.args:
                        index = self.resolve(node.args[0], env)
                        result = Binding(ast.Subscript(value=result.expression, slice=index.expression,
                                         ctx=ast.Load()), result.slots | index.slots, "metadata")
                    return result
                if method in {"numel", "get_numel"} and not node.args and base.kind in {"tensor", "attribute"}:
                    return Binding(ast.Call(func=ast.Attribute(value=copy.deepcopy(base.expression),
                                   attr="numel", ctx=ast.Load()), args=[], keywords=[]), base.slots, "metadata")
                raise Unresolved("unknown method: " + method)
            if not isinstance(node.func, ast.Name) or node.func.id not in self.BUILTINS:
                raise Unresolved("unknown helper call")
            if node.func.id in self.local_names:
                raise Unresolved("shadowed builtin: " + node.func.id)
            values = [self.resolve(arg, env) for arg in node.args]
            if any(v.kind == "tensor" for v in values):
                raise Unresolved("builtin applied to tensor contents")
            return Binding(ast.Call(func=copy.deepcopy(node.func),
                           args=[copy.deepcopy(v.expression) for v in values], keywords=[]),
                           set().union(*(v.slots for v in values)))
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare,
                             ast.Subscript, ast.Slice, ast.Tuple, ast.List)):
            result = copy.deepcopy(node)
            slots = set()
            for field, value in ast.iter_fields(node):
                if isinstance(value, ast.expr):
                    child = self.resolve(value, env)
                    if child.kind == "tensor":
                        raise Unresolved("operation on tensor contents")
                    setattr(result, field, copy.deepcopy(child.expression))
                    slots.update(child.slots)
                elif isinstance(value, list):
                    children = []
                    for item in value:
                        if isinstance(item, ast.expr):
                            child = self.resolve(item, env)
                            if child.kind == "tensor":
                                raise Unresolved("operation on tensor contents")
                            children.append(copy.deepcopy(child.expression))
                            slots.update(child.slots)
                        else:
                            children.append(copy.deepcopy(item))
                    setattr(result, field, children)
            return Binding(result, slots)
        raise Unresolved("unsupported expression: " + type(node).__name__)

    def guard(self, expression, env, kind):
        try:
            value = self.resolve(expression, env)
            if value.kind == "tensor" or not value.slots:
                raise Unresolved("no resolved configuration dependency")
            self.predicates.append({"guard": ast.unparse(expression),
                                    "predicate": ast.unparse(value.expression),
                                    "slots": sorted(value.slots),
                                    "line": expression.lineno, "kind": kind})
        except Unresolved as exc:
            self.skipped.append({"guard": ast.unparse(expression), "line": expression.lineno,
                                 "reason": str(exc), "kind": kind})

    def effect_free(self, expression, env):
        """Only expressions understood by the resolver can be assumed pure."""
        try:
            self.resolve(expression, env)
            return True
        except Unresolved:
            return False

    @staticmethod
    def merge(left, right):
        return {name: value for name, value in left.items()
                if name in right and value.kind == right[name].kind
                and value.slots == right[name].slots
                and ast.dump(value.expression) == ast.dump(right[name].expression)}

    def numeric(self, expression):
        """Recognize scalar results before treating += as a pure reassignment."""
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, (bool, int, float, complex))
        if isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                return expression.func.id in {"len", "int", "float", "bool"}
            return isinstance(expression.func, ast.Attribute) and expression.func.attr == "numel"
        if isinstance(expression, ast.Attribute):
            return expression.attr == "ndim"
        if isinstance(expression, ast.Subscript):
            return (isinstance(expression.value, ast.Attribute)
                    and expression.value.attr == "shape" and self.numeric(expression.slice))
        if isinstance(expression, ast.BinOp):
            return self.numeric(expression.left) and self.numeric(expression.right)
        if isinstance(expression, ast.UnaryOp):
            return self.numeric(expression.operand)
        return isinstance(expression, ast.Compare)

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            if value is None:
                env.pop(target.id, None)
            else:
                env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Only unpack literal tuples and lists.
            if (value and isinstance(value.expression, (ast.Tuple, ast.List))
                    and len(target.elts) == len(value.expression.elts)):
                for dest, item in zip(target.elts, value.expression.elts):
                    # Resolve against original input roots.
                    roots = {name: Binding(ast.Name(id=name, ctx=ast.Load()), [name])
                             for name in value.slots}
                    try:
                        child = self.resolve(item, roots)
                    except Unresolved:
                        child = None
                    self.assign(dest, child, env)
            else:
                for dest in target.elts:
                    self.assign(dest, None, env)
        else:
            # Writes may mutate aliased inputs.
            env.clear()

    def block(self, statements, env):
        env = dict(env)
        for node in statements:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                expression = node.value
                if isinstance(expression, ast.IfExp):
                    self.guard(expression.test, env, "ifexp")
                try:
                    value = self.resolve(expression, env) if expression is not None else None
                except Unresolved:
                    value = None
                # Unknown calls may mutate inputs.
                if expression is not None and any(isinstance(n, (ast.Call, ast.NamedExpr,
                        ast.Await, ast.Yield, ast.YieldFrom)) for n in ast.walk(expression)):
                    if not self.effect_free(expression, env):
                        env.clear()
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                try:
                    original = self.resolve(node.target, env)
                    operand = self.resolve(node.value, env)
                    if not self.numeric(original.expression) or not self.numeric(operand.expression):
                        raise Unresolved("augmented assignment may mutate aliases")
                    value = self.resolve(ast.BinOp(left=node.target, op=node.op, right=node.value), env)
                except Unresolved:
                    env.clear()
                    continue
                if not self.effect_free(node.value, env):
                    env.clear()
                self.assign(node.target, value, env)
            elif isinstance(node, ast.If):
                self.guard(node.test, env, "if")
                if not self.effect_free(node.test, env):
                    env.clear()
                left = self.block(node.body, env)
                right = self.block(node.orelse, env)
                env = self.merge(left, right)
            elif isinstance(node, ast.Assert):
                # Exclude invocation requirements from dispatch conditions.
                self.skipped.append({"guard": ast.unparse(node.test), "line": node.lineno,
                                     "reason": "assertion is a validity constraint", "kind": "assert"})
                if not self.effect_free(node.test, env):
                    env.clear()
            elif isinstance(node, ast.Return):
                if isinstance(node.value, ast.IfExp):
                    self.guard(node.value.test, env, "ifexp")
                return env
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip nested scopes; decorators and defaults may have effects.
                env.clear()
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Constant):
                    continue  # docstrings
                if not self.effect_free(node.value, env):
                    env.clear()
            elif isinstance(node, ast.Pass):
                continue
            else:
                # Loops, try/with, match, imports, deletes, break/continue, etc.
                # No fixed-point/alias/exception analysis is claimed here.
                env.clear()
        return env

    def run(self):
        self.block(self.function.body, self.env)
        return {"function": self.function.name, "predicates": self.predicates,
                "skipped": self.skipped}


def analyze_source(source, function_name=None, configuration_slots=None):
    """Return JSON-compatible guards and slot dependencies for one function.

    configuration_slots maps parameter names to 'tensor' or 'attribute'. If
    omitted, non-self parameters are assumed configuration attributes; adapters
    with schema types should provide the explicit mapping.
    """
    if len(source) > 500_000:
        raise ValueError("source exceeds analysis size limit")
    tree = ast.parse(textwrap.dedent(source))
    if sum(1 for _ in ast.walk(tree)) > 10_000:
        raise ValueError("source exceeds analysis AST limit")
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and (function_name is None or n.name == function_name)]
    if len(functions) != 1:
        raise ValueError("Select exactly one top-level function with --function")
    return Analyzer(functions[0], configuration_slots).run()


def select_conditions(fn, legacy_conditions, configuration_slots=None):
    """Compatibility API: list[str] in/out, with function-level legacy fallback."""
    if os.getenv("LPTGEN_DEF_USE", "1").lower() in {"0", "false", "off"}:
        return legacy_conditions
    try:
        fn = inspect.unwrap(fn)
        report = analyze_source(inspect.getsource(fn), fn.__name__, configuration_slots)
        conditions = list(dict.fromkeys(p["predicate"] for p in report["predicates"]))
        if conditions:
            return conditions
        logging.getLogger(__name__).info("No resolved predicates for %s; using legacy extraction", fn.__name__)
    except Exception as exc:
        logging.getLogger(__name__).warning("Def-use analysis failed; using legacy extraction: %s", exc)
    return legacy_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--function")
    parser.add_argument("--tensor", action="append", default=[], help="tensor parameter name (repeatable)")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    slots = None
    if args.tensor:
        tree = ast.parse(textwrap.dedent(source))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and (args.function is None or n.name == args.function)]
        if len(functions) != 1:
            parser.error("select exactly one function")
        parameters = functions[0].args
        slots = {a.arg: ("tensor" if a.arg in args.tensor else "attribute")
                 for a in parameters.posonlyargs + parameters.args + parameters.kwonlyargs}
    print(json.dumps(analyze_source(source, args.function, slots), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
