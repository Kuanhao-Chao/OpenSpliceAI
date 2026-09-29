"""Behavioral checks for the revision's display data and document packaging."""
import importlib
import importlib.util

import pytest


def module():
    name = "validation.concordance_study.concise_figures"
    assert importlib.util.find_spec(name) is not None, "concise display module is missing"
    return importlib.import_module(name)


def test_site_call_rates_pool_counts_instead_of_averaging_rates():
    rows = [
        dict(distance="1-2", event="AG", threshold=0.5, n=10,
             both_positive=2, left_only=2, right_only=1),
        dict(distance="1-2", event="AG", threshold=0.5, n=90,
             both_positive=3, left_only=3, right_only=2),
        dict(distance="1-2", event="AG", threshold=0.2, n=100,
             both_positive=90, left_only=0, right_only=0),
    ]
    actual = module().pool_site_rates(rows, "AG", 0.5)
    assert actual == [{"distance": "1-2", "n": 100, "left": 10, "right": 8,
                       "left_rate": 0.1, "right_rate": 0.08, "ratio": 0.8}]


def test_site_call_rates_preserve_zero_and_undefined_denominators():
    rows = [dict(distance=">500", event="AL", threshold=0.5, n=100,
                 both_positive=0, left_only=0, right_only=0)]
    row = module().pool_site_rates(rows, "AL", 0.5)[0]
    assert row["left_rate"] == 0
    assert row["right_rate"] == 0
    assert row["ratio"] is None


def test_site_call_rates_reject_inconsistent_counts():
    rows = [dict(distance="1-2", event="AG", threshold=0.5, n=2,
                 both_positive=2, left_only=1, right_only=0)]
    with pytest.raises(ValueError, match="counts"):
        module().pool_site_rates(rows, "AG", 0.5)


def test_position_figure_exposes_rare_mismatches(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    figures = module()
    facts = {'primary': {'dp': {e: {'0.5': {'eligible': n, 'within_0bp_n': n-m,
                                          'within_0bp': (n-m)/n}}
                               for e,n,m in [('AG',100,3),('AL',1441730,1),('DG',100,2),('DL',1630987,4)]}}}
    captured = []
    def inspect(fig, *args):
        captured.extend(t.get_text() for ax in fig.axes for t in ax.texts)
        plt.close(fig)
    monkeypatch.setattr(figures, '_save', inspect)
    figures.position_figure(facts, tmp_path)
    assert '100.00%' not in captured
    assert any('Mismatches: 1' in t and '1,441,730' in t for t in captured)
    assert any('Mismatches: 4' in t and '1,630,987' in t for t in captured)


def test_context_positions_keep_absent_groups_and_pool_counts():
    from validation.concordance_study.concise_supplement import position_groups
    rows = [dict(event='AL', threshold='.5', tolerance_bp='0', distance='11-50', eligible=n, within=w)
            for n,w in [(100,99),(900,900)]]
    groups = position_groups(rows, 'AL', 0)
    assert len(groups) == 6
    assert groups[3]['rate'] == .999
    assert groups[3]['mismatches'] == 1
    assert groups[4]['rate'] is None
    assert groups[4]['eligible'] == 0


def test_supplement_position_table_preserves_exact_mismatch_counts():
    from validation.concordance_study.markdown_tables import render
    facts = {'primary': {'dp': {e: {'0.5': {'eligible': n, 'within_0bp_n': n-m}}
                               for e,n,m in [('AG',100,3),('AL',1441730,1),('DG',100,2),('DL',1630987,4)]}}}
    table = render('dp_agreement_counts', facts)
    assert '| 1,441,730 | 1,441,729 | 1 |' in table
    assert '| 1,630,987 | 1,630,983 | 4 |' in table
    assert '100.00%' not in table
