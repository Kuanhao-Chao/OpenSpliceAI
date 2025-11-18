import json

import numpy as np

from openspliceai.rbp.expression import (
    RBPExpression,
    load_rbp_expression,
    save_rbp_expression,
    standardize_vector,
)
from openspliceai.scripts.prepare_rbp_expression import prepare_expression


def test_load_and_save_json(tmp_path):
    expr = RBPExpression(values=[0.5, -1.0], names=["RBFOX1", "MBNL1"])
    path = tmp_path / "rbp.json"
    save_rbp_expression(expr, path, fmt="json")
    loaded = load_rbp_expression(path)
    assert np.allclose(loaded.values, expr.values)
    assert loaded.names == expr.names


def test_standardize_vector_zscore():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    standardized = standardize_vector(vec, method="zscore")
    assert np.isclose(standardized.mean(), 0.0, atol=1e-6)


def test_load_json_mismatched_names(tmp_path):
    path = tmp_path / "bad.json"
    payload = {"rbp_names": ["A"], "values": [0.1, 0.2]}
    path.write_text(json.dumps(payload))
    try:
        load_rbp_expression(path)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched names and values")


def test_prepare_expression_with_hvg(tmp_path):
    rbp_matrix = tmp_path / "rbp.csv"
    rbp_matrix.write_text("tissue,R1,R2\nlimb,0.1,-0.2\n")
    hvg_matrix = tmp_path / "hvg.csv"
    hvg_matrix.write_text("tissue,H1\nlimb,1.5\n")
    output = tmp_path / "vector.json"

    expr = prepare_expression(
        rbp_matrix,
        "limb",
        output,
        "json",
        "none",
        hvg_matrix_path=hvg_matrix,
        hvg_standardize="none",
    )

    payload = json.loads(output.read_text())
    assert payload["rbp_names"] == ["R1", "R2", "H1"]
    assert np.allclose(expr.values, [0.1, -0.2, 1.5])
