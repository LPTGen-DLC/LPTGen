import ast
import importlib
import inspect
import json
import pkgutil
import textwrap
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Set, Tuple


def extract_conditions_from_fn(fn) -> List[str]:
    """Extract branch predicates from a Python function body."""
    try:
        src = inspect.getsource(fn)
    except (OSError, IOError, TypeError):
        return []

    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return []

    conds: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node):
            conds.append(ast.unparse(node.test))
            self.generic_visit(node)

        def visit_Assert(self, node):
            conds.append(ast.unparse(node.test))
            self.generic_visit(node)

        def visit_IfExp(self, node):
            conds.append(ast.unparse(node.test))
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(set(conds))


def _get_tvm_topi_root():
    try:
        import tvm.topi as topi  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TVM is not installed in the current Python environment. "
            "Please run this script in an environment where `import tvm` works."
        ) from exc
    return topi


def _iter_topi_modules(topi_root: ModuleType, include_private: bool = False) -> Iterable[Tuple[str, ModuleType]]:
    """Walk tvm.topi and all importable submodules."""
    seen: Set[str] = set()
    queue: List[Tuple[str, ModuleType]] = [("topi", topi_root)]

    while queue:
        mod_name, mod = queue.pop(0)
        if mod_name in seen:
            continue
        seen.add(mod_name)
        yield mod_name, mod

        mod_path = getattr(mod, "__path__", None)
        if not mod_path:
            continue

        for info in pkgutil.iter_modules(mod_path):
            if not include_private and info.name.startswith("_"):
                continue
            child_name = f"{mod_name}.{info.name}"
            if child_name in seen:
                continue
            try:
                child = importlib.import_module(f"{mod.__name__}.{info.name}")
            except Exception as exc:
                print(f"[INFO] Skip import {child_name}: {exc}")
                continue
            queue.append((child_name, child))


def _is_topi_defined_function(obj, module_name: str) -> bool:
    if inspect.isfunction(obj):
        return getattr(obj, "__module__", "").startswith("tvm.topi")

    obj_type_name = type(obj).__name__.lower()
    if "genericfunc" in obj_type_name:
        return getattr(obj, "__module__", "").startswith("tvm.")

    if callable(obj) and getattr(obj, "__module__", "").startswith("tvm.topi"):
        return True

    return False


def _iter_module_public_callables(module_name: str, mod: ModuleType) -> Iterable[Tuple[str, object]]:
    seen: Set[int] = set()
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if not _is_topi_defined_function(obj, module_name):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield f"{module_name}.{name}", obj


def _maybe_collect_generic_dispatch(fn) -> Dict[str, List[str]]:
    """
    Best-effort extraction for TVM GenericFunc-style dispatch functions.

    TVM versions differ here, so this method is intentionally defensive.
    """
    impls: Dict[str, List[str]] = {}

    default_fn = getattr(fn, "fdefault", None)
    if callable(default_fn):
        conds = extract_conditions_from_fn(default_fn)
        if conds:
            impls["default"] = conds

    dispatch_dict = getattr(fn, "dispatch_dict", None)
    if isinstance(dispatch_dict, dict):
        for key, target_fn in dispatch_dict.items():
            if not callable(target_fn):
                continue
            conds = extract_conditions_from_fn(target_fn)
            if conds:
                impls[str(key)] = conds

    # Some wrappers may expose the underlying Python callable via __wrapped__.
    wrapped = getattr(fn, "__wrapped__", None)
    if callable(wrapped):
        conds = extract_conditions_from_fn(wrapped)
        if conds:
            impls.setdefault("wrapped", conds)

    return impls


def collect_topi_lowering_knowledge(
    module_prefixes: Optional[List[str]] = None,
    include_private_modules: bool = False,
) -> Dict[str, Dict[str, object]]:
    """
    Collect branch conditions from TOPI Python implementations.

    Returned schema:
    {
        "topi.nn.conv2d": {
            "module": "tvm.topi.nn",
            "source": "direct" | "generic_dispatch",
            "conditions": [...],
            "dispatch_conditions": {
                "default": [...],
                "cuda": [...],
            }
        }
    }
    """
    topi_root = _get_tvm_topi_root()
    knowledge: Dict[str, Dict[str, object]] = {}
    module_prefixes = module_prefixes or ["topi"]

    for module_name, mod in _iter_topi_modules(topi_root, include_private=include_private_modules):
        if not any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in module_prefixes):
            continue

        print(f"[INFO] Scan module {module_name}")
        for qualname, fn in _iter_module_public_callables(module_name, mod):
            entry: Dict[str, object] = {
                "module": getattr(fn, "__module__", ""),
                "source": "direct",
                "conditions": [],
            }

            direct_conds = extract_conditions_from_fn(fn)
            if direct_conds:
                entry["conditions"] = direct_conds

            dispatch_conds = _maybe_collect_generic_dispatch(fn)
            if dispatch_conds:
                entry["source"] = "generic_dispatch"
                entry["dispatch_conditions"] = dispatch_conds
                merged = set(direct_conds)
                for conds in dispatch_conds.values():
                    merged.update(conds)
                entry["conditions"] = sorted(merged)

            if entry["conditions"]:
                knowledge[qualname] = entry

    return knowledge


def collect_selected_topi_namespaces() -> Dict[str, Dict[str, Dict[str, object]]]:
    """
    Grouped entrypoint similar to the PyTorch version.
    """
    return {
        "topi": collect_topi_lowering_knowledge(
            module_prefixes=[
                "topi",
                "topi.nn",
                "topi.image",
                "topi.math",
                "topi.transform",
                "topi.reduction",
                "topi.broadcast",
                "topi.sort",
                "topi.scatter",
                "topi.searchsorted",
                "topi.signal",
            ]
        )
    }


if __name__ == "__main__":
    knowledge = collect_selected_topi_namespaces()
    total_conditions = 0
    total_ops = 0

    for group_name, group in knowledge.items():
        print(f"[INFO] {group_name}: {len(group)} ops")
        total_ops += len(group)
        for op_name, item in group.items():
            cond_count = len(item["conditions"])
            total_conditions += cond_count
            print(f"{op_name}: {cond_count}")

    print(f"[INFO] Total ops with conditions: {total_ops}")
    print(f"[INFO] Total unique extracted conditions: {total_conditions}")

    with open("topi_lowering_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)
