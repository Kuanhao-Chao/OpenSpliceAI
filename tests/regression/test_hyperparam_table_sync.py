"""Guard-test for the duplicated ``(W, AR, BATCH_SIZE)`` hyperparameter table.

CLAUDE.md flags this duplication as "the most common source of subtle bugs": the
flanking-size -> ``(W, AR, BATCH_SIZE)`` ladder is copy-pasted into four call sites
(``train``, ``transfer``, ``predict``, ``variant``) and must stay in lockstep. This test
parses each source file's ``if int(flanking_size) == ...`` ladder via the AST and asserts
all four copies are byte-identical per flanking size, and that the resulting context
length ``CL == flanking_size`` exactly.

It parses *source* rather than executing the loaders, so it needs no checkpoints or GPU
and pins the literal table even in branches the runtime tests skip (e.g. 2000/10000nt).
"""
import ast
import importlib

import numpy as np
import pytest

# (module, the function that owns the ladder) -- the function name is documentation only;
# the AST walk scans the whole module for the flanking-size ladder.
TABLE_SITES = [
    "openspliceai.train.train",          # initialize_model_and_optim
    "openspliceai.transfer.transfer",    # initialize_model_and_optim_transfer
    "openspliceai.predict.predict",      # load_pytorch_models.load_model
    "openspliceai.variant.utils",        # load_pytorch_models.load_model
    "openspliceai.calibrate.model_utils",  # initialize_model_and_optim (5th copy)
]
FLANKING_SIZES = [80, 400, 2000, 10000]


def _const_int_list(node):
    """Extract the int list from ``np.asarray([...])`` (or a bare list)."""
    if isinstance(node, ast.Call):          # np.asarray([...])
        assert node.args, ast.dump(node)
        node = node.args[0]
    assert isinstance(node, (ast.List, ast.Tuple)), ast.dump(node)
    out = []
    for e in node.elts:
        assert isinstance(e, ast.Constant) and isinstance(e.value, int), ast.dump(e)
        out.append(e.value)
    return out


def _extract_table(module_name):
    """Return ``{flanking: {'W': [...], 'AR': [...], 'BATCH_SIZE': '<ast dump>'}}``."""
    mod = importlib.import_module(module_name)
    with open(mod.__file__) as fh:
        tree = ast.parse(fh.read())

    table = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)):
            continue
        left, comp = test.left, test.comparators[0]
        # match exactly: int(flanking_size) == <int const>
        if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "int" and left.args
                and isinstance(left.args[0], ast.Name)
                and left.args[0].id == "flanking_size"):
            continue
        if not (isinstance(comp, ast.Constant) and isinstance(comp.value, int)):
            continue
        flank = comp.value
        entry = {}
        for stmt in node.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)):
                name = stmt.targets[0].id
                if name in ("W", "AR"):
                    entry[name] = _const_int_list(stmt.value)
                elif name == "BATCH_SIZE":
                    entry[name] = ast.dump(stmt.value)  # e.g. "18*N_GPUS", compared verbatim
        table[flank] = entry
    return table


@pytest.fixture(scope="module")
def tables():
    return {name: _extract_table(name) for name in TABLE_SITES}


def test_every_site_defines_every_flanking_size(tables):
    for name, table in tables.items():
        assert set(table) == set(FLANKING_SIZES), f"{name} ladder keys = {sorted(table)}"
        for flank, entry in table.items():
            assert set(entry) == {"W", "AR", "BATCH_SIZE"}, f"{name}[{flank}] = {entry}"


@pytest.mark.parametrize("flank", FLANKING_SIZES)
def test_table_identical_across_all_four_sites(tables, flank):
    items = {name: table[flank] for name, table in tables.items()}
    ref_name, ref = next(iter(items.items()))
    for name, entry in items.items():
        assert entry == ref, (
            f"hyperparameter table drift at flanking={flank}:\n"
            f"  {ref_name} = {ref}\n  {name} = {entry}"
        )


@pytest.mark.parametrize("flank", FLANKING_SIZES)
def test_cl_equals_flanking_size(tables, flank):
    entry = tables["openspliceai.train.train"][flank]
    W, AR = np.asarray(entry["W"]), np.asarray(entry["AR"])
    assert len(W) == len(AR) and len(W) in (4, 8, 12, 16)
    CL = 2 * int(np.sum(AR * (W - 1)))
    assert CL == flank, f"CL={CL} != flanking={flank}"
