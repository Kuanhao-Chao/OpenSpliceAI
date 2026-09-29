"""Loading and contract validation for reduced concordance summaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional

SCORE_LABELS = ("AG", "AL", "DG", "DL", "MAX")
EVENTS = ("AG", "AL", "DG", "DL")
EVENT_NAMES = {
    "AG": "acceptor gain",
    "AL": "acceptor loss",
    "DG": "donor gain",
    "DL": "donor loss",
    "MAX": "maximum delta score",
}


class ContractError(RuntimeError):
    """A summary failed the finality/provenance contract required for reporting."""


@dataclass(frozen=True)
class Run:
    """One reduced comparison arm."""

    arm: str
    role: str
    path: Path
    summary: Mapping

    # -- identity -----------------------------------------------------------
    @property
    def kind(self) -> str:
        """The reducer writes ``reduced-concordance``/``reduced-seeds``; mappers write the bare form."""
        return self.summary["kind"]

    @property
    def is_concordance(self) -> bool:
        return "concordance" in self.summary["kind"]

    @property
    def left_label(self) -> str:
        return self.summary["left_label"]

    @property
    def right_label(self) -> str:
        return self.summary["right_label"]

    @property
    def comparison(self) -> str:
        return f"{self.left_label} vs {self.right_label}"

    @property
    def run_label(self) -> str:
        labels = self.summary.get("mapper_run_labels") or []
        return labels[0] if len(labels) == 1 else ",".join(labels)

    # -- contract -----------------------------------------------------------
    @property
    def finality(self) -> Mapping:
        return self.summary["finality"]

    @property
    def provenance(self) -> Mapping:
        return self.summary["verified_provenance"]

    @property
    def chunk_count(self) -> int:
        return int(self.finality["observed_pair_count"])

    # -- payload ------------------------------------------------------------
    @property
    def metrics(self) -> Mapping:
        return self.summary["metrics"]

    @property
    def raw(self) -> Mapping:
        return self.summary["raw"]

    @property
    def coverage(self) -> Mapping:
        return self.metrics["coverage"]

    def scores(self, label: str) -> Mapping:
        return self.metrics["scores"][label]

    def thresholds(self, label: str, threshold: float) -> Mapping:
        return self.metrics["thresholds"][label][_key(threshold)]

    def paired_n(self) -> int:
        return int(self.metrics["scores"]["MAX"]["n"])

    def joint_hist(self, label: str) -> list:
        return self.raw["joint_hist"][label]

    def score_hist(self, label: str) -> Mapping:
        return self.raw["score_hist"][label]

    @property
    def score_bins(self) -> int:
        return int(self.raw["config"]["score_bins"])


def _key(value: float) -> str:
    return f"{value:.10g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: str | Path, arm: str, role: str, *, expected_chunks: Optional[int] = None) -> Run:
    """Load one ``summary.json`` and enforce the contract needed to report it.

    The reducer already fails closed on missing tasks, duplicate chunks and
    changed inputs. What we re-assert here is what a *reader* of the report needs
    to trust: that the status is one of the two declared values, that a run
    calling itself final really carries the completeness contract, and that the
    pair count matches what this study intended to analyse.
    """
    path = Path(path)
    summary = json.loads(path.read_text())

    status = summary.get("finality", {}).get("status")
    if status not in ("provisional", "final"):
        raise ContractError(f"{path}: unknown finality status {status!r}")
    if status == "final" and not summary["finality"].get("pair_id_domain_complete"):
        raise ContractError(f"{path}: labelled final without a complete pair-id domain")
    if summary["verified_provenance"].get("status") != "verified":
        raise ContractError(f"{path}: provenance status is not verified")

    observed = int(summary["finality"]["observed_pair_count"])
    verified = int(summary["verified_provenance"]["verified_chunk_count"])
    if observed != verified:
        raise ContractError(f"{path}: {observed} pairs but {verified} verified chunks")
    if expected_chunks is not None and observed != expected_chunks:
        raise ContractError(f"{path}: expected {expected_chunks} chunks, found {observed}")
    if summary["finality"].get("unexpected_chunk_ids"):
        raise ContractError(f"{path}: reducer reported unexpected chunk ids")

    return Run(arm=arm, role=role, path=path, summary=summary)


def load_study(runs_root: str | Path, spec: Mapping[str, Mapping]) -> Dict[str, Run]:
    """Load every arm named in ``spec`` -> {arm: Run}."""
    runs_root = Path(runs_root)
    loaded: Dict[str, Run] = {}
    for arm, config in spec.items():
        loaded[arm] = load_run(
            runs_root / config["directory"] / "summary.json",
            arm=arm,
            role=config["role"],
            expected_chunks=config.get("expected_chunks"),
        )
    return loaded
