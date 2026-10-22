import ast
import inspect
import torch._inductor.lowering as lowering
import textwrap
import torch
import torch._inductor.utils as utils
import torch._inductor.ir as ir
import importlib.util
import logging
import os
from pathlib import Path
from functools import lru_cache


def get_lowering_func(_schema):
    ns, opname = _schema.name.split("::")
    overload = _schema.overload_name if _schema.overload_name else "default"

    try:
        packet = getattr(torch.ops, ns)
        op = getattr(getattr(packet, opname), overload)
    except AttributeError:
        return None
    if op in lowering.lowerings:
        return lowering.lowerings[op]
    return None


def _extract_legacy_conditions_from_fn(fn):
    try:
        src = inspect.getsource(fn)
    except (OSError, IOError):
        return []
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    conds = []

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node):
            conds.append(ast.unparse(node.test))
            self.generic_visit(node)

        def visit_Assert(self, node):
            conds.append(ast.unparse(node.test))

    Visitor().visit(tree)
    print(f"[INFO] The knowledge of is {conds}")
    return list(set(conds))


@lru_cache(maxsize=1)
def _load_def_use():
    path = Path(__file__).resolve().with_name("lowering_def_use.py")
    spec = importlib.util.spec_from_file_location("lptgen_lowering_def_use", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_conditions_from_fn(fn, configuration_slots=None):
    """Preserve list[str] API; retain legacy behavior on optional-analysis failure."""
    legacy = _extract_legacy_conditions_from_fn(fn)
    if os.getenv("LPTGEN_DEF_USE", "1").lower() in {"0", "false", "off"}:
        return legacy
    try:
        return _load_def_use().select_conditions(fn, legacy, configuration_slots)
    except Exception as exc:
        logging.getLogger(__name__).warning("Def-use unavailable; using legacy predicates: %s", exc)
        return legacy

def collect_op_lowering_knowledge():
    knowledge = {}
    for op, fn in lowering.lowerings.items():
        op = str(op)
        if not str(op).startswith("aten"):
            continue
        try:
            knowledge[op.replace('.', '::')] = extract_conditions_from_fn(fn)
        except OSError:
            print(f"[INFO] Failed to parse {op}")
    return knowledge


def collect_helper_conditions():
    helpers = [utils, ir]
    knowledge = {}

    for mod in helpers:
        for name in dir(mod):
            print(f"[INFO] Begin parse {name}")
            obj = getattr(mod, name)
            if callable(obj):
                try:
                    conds = extract_conditions_from_fn(obj)
                except Exception as e:
                    print(f"[INFO] Failed to parse {name} due to: {e}")
                if conds:
                    knowledge[f"{mod.__name__}.{name}"] = conds

    return knowledge

import torch._inductor.codegen.triton as triton

def collect_triton_conditions():
    knowledge = {}

    for name in dir(triton):
        obj = getattr(triton, name)
        if callable(obj):
            try:
                conds = extract_conditions_from_fn(obj)
            except Exception as e:
                print(f"[INFO] Failed to parse {name}")
            if conds:
                knowledge[f"triton.{name}"] = conds
    return knowledge


import torch._inductor.scheduler as scheduler

def collect_scheduler_conditions():
    knowledge = {}

    for name in dir(scheduler):
        obj = getattr(scheduler, name)
        if callable(obj):
            try:
                conds = extract_conditions_from_fn(obj)
            except Exception as e:
                print(f"[INFO] Failed to parse {name}")
            if conds:
                knowledge[f"scheduler.{name}"] = conds
    print(f"[INFO] Finish parsing scheduler")
    return knowledge


def collect_all_lowering_knowledge():
    return {
        "op_lowering": collect_op_lowering_knowledge(),
        # "helper_conditions": collect_helper_conditions(),
        # "triton_codegen": collect_triton_conditions(),
        # "scheduler": collect_scheduler_conditions(),
    }


if __name__ == '__main__':
    op_conditions_dicts = collect_all_lowering_knowledge()
    total_conditions_counts = 0
    for k, v in op_conditions_dicts.items():
        print(f"[INFO] {k}: {len(v)}")
        for op, cond in v.items():
            print(op, len(cond))
            total_conditions_counts += len(cond)
    print(total_conditions_counts)
