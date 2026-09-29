import pytest


def module():
    from validation.concordance_study import concise_release
    return concise_release


def test_figure_contract_rejects_missing_or_duplicate_figures():
    figures = [{"id": "a", "document": "main", "number": 1, "title": "Title", "caption": "Caption", "alt": "Alt"}]
    with pytest.raises(ValueError, match="figure"):
        module().expand_figures("<!-- figure:unknown -->", figures, "main", "markdown")
    with pytest.raises(ValueError, match="figure"):
        module().expand_figures("<!-- figure:a -->\n<!-- figure:a -->", figures, "main", "markdown")
    with pytest.raises(ValueError, match="figure"):
        module().expand_figures("No figures", figures, "main", "markdown")


def test_same_caption_and_number_in_both_reading_formats():
    figures = [{"id": "a", "document": "main", "number": 1, "title": "Result", "caption": "Evidence.", "alt": "A comparison"}]
    for mode in ("markdown", "mdx"):
        rendered = module().expand_figures("<!-- figure:a -->", figures, "main", mode)
        assert "Figure 1. Result" in rendered
        assert "Evidence." in rendered
        assert "figure:a" not in rendered


def test_progress_context_does_not_mutate_scientific_facts():
    facts = {"study": {"date": "2026-09-09"}, "primary": {"agreement": {}, "thresholds": {}, "dp": {}}}
    for event in ("AG", "AL", "DG", "DL"):
        facts["primary"]["agreement"][event] = {"pearson_r": .7}
        facts["primary"]["thresholds"][event] = {"0.5": {"jaccard": .4, "call_rate_ratio_right_over_left": .8}}
        facts["primary"]["dp"][event] = {"0.5": {"within_0bp": .9}}
    snapshot = {"observed_at_utc": "2026-09-10T12:00:00+00:00", "seeds": {
        seed: {"total_chunks": 10, "current_counts": {"valid": 8, "missing": 1, "invalid": 1, "unverified": 0},
               "audit_at_utc": "2026-08-08T12:00:00+00:00"} for seed in ("rs10", "rs13")}}
    context = module().template_context(facts, snapshot)
    assert "revision" not in facts
    assert context["primary"] is facts["primary"]
    assert "8 of 10" in context["revision"]["progress_summary"]
    snapshot["seeds"]["rs10"]["current_counts"]["valid"] = 11
    with pytest.raises(ValueError, match="progress"):
        module().template_context(facts, snapshot)


def test_reusing_scientific_figures_refreshes_progress(monkeypatch, tmp_path):
    release = module()
    calls = []
    snapshot = {"observed_at_utc": "new observation"}
    monkeypatch.setattr(release.concise_figures, "progress_figure", lambda f, p, d: calls.append((p, d)))
    monkeypatch.setattr(release.concise_figures, "render", lambda *args: pytest.fail("scientific figures should be reused"))
    release.render_revision_figures({}, snapshot, tmp_path, tmp_path, skip_scientific=True)
    assert calls == [(snapshot, tmp_path / "figures")]


def test_release_identity_cannot_overwrite_existing_downloads(tmp_path):
    release = module()
    (tmp_path/'full-snv-concordance-20260910.zip').write_bytes(b'published')
    with pytest.raises(ValueError, match='already exists'):
        release.protect_release(tmp_path,'full-snv-concordance-20260910')
    release.protect_release(tmp_path,'full-snv-concordance-20260910-r2')
    (tmp_path/'full-snv-concordance-20260910-r2').mkdir()
    with pytest.raises(ValueError, match='already exists'):
        release.protect_release(tmp_path,'full-snv-concordance-20260910-r2')
    assert (tmp_path/'full-snv-concordance-20260910.zip').read_bytes() == b'published'


def test_boundary_ratio_prose_uses_original_counts_without_double_rounding():
    from validation.concordance_study import report
    facts = {"primary": {"agreement": {}, "thresholds": {}, "dp": {}, "site_event_table": [
        dict(event="AL", distance="1-2", threshold=.5, n=2000000, both_positive=990795,
             left_only=42076, right_only=14670)]}}
    for e in ("AG", "AL", "DG", "DL"):
        facts["primary"]["agreement"][e] = {"pearson_r": .7}
        facts["primary"]["thresholds"][e] = {"0.5": {"jaccard": .4, "call_rate_ratio_right_over_left": .8}}
        facts["primary"]["dp"][e] = {"0.5": {"within_0bp": .9}}
    snapshot = {"observed_at_utc": "2026-09-11T01:00:00+00:00", "seeds": {
        e: {"total_chunks": 10, "current_counts": {"valid": 8, "missing": 1, "invalid": 1, "unverified": 0},
            "audit_at_utc": "2026-08-08T12:00:00+00:00"} for e in ("rs10", "rs13")}}
    context = module().template_context(facts, snapshot, "20260910-r2", "2026-09-10")
    assert report.resolve("{{revision.boundary_ratios.AL[1-2]|.3f}}", context) == "0.973"
    assert context["revision"]["date"] == "2026-09-10"
    assert context["revision"]["prefix"] == "full-snv-concordance-20260910-r2"
    assert context["revision"]["observed_at"].startswith("2026-09-11")


def test_unsupported_placeholder_syntax_cannot_silently_reach_reading_copy():
    from validation.concordance_study import report
    with pytest.raises(KeyError, match='unsupported placeholder syntax'):
        report.resolve('{{revision.boundary_ratios.AG[>500]|.3f}}', {})
