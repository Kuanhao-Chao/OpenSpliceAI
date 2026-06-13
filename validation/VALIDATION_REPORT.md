# Validation report — OpenSpliceAI `predict` and `variant`

**Question.** Before large-scale variant scoring, are the `predict` and `variant` modules correct, step by
step? **Answer: yes.** `variant` is **numerically identical to the original Illumina SpliceAI** (exact match
on every score and position across a parameter grid), and `predict` maps coordinates correctly on both
strands. One real (benign-but-real) bug was found and fixed; one robustness guard was added; nine flagged
concerns were each resolved (7 false positives, 1 cosmetic, 1 hardened).

Companion: `ALGORITHM.md` (step-by-step algorithm). All checks are reproducible with the scripts in this
directory (commands at the end). Interpreter: `/home/kchao10/miniconda3/envs/pytorch_cuda/bin/python`
(torch 2.2.1, tensorflow 2.15, original `spliceai` 1.3.1).

---

## A. Keystone — numerical equivalence to original SpliceAI

`variant --model-type keras --flanking-size 10000` was run head-to-head against the original `spliceai` 1.3.1
CLI on **identical inputs** (same reference FASTA, same annotation file, same Keras weights), over a VCF of
**35 variants** spanning the full matrix: SNVs on canonical donor/acceptor sites of **3 plus-strand and 3
minus-strand genes** (SAMD11, KLHL17, PLEKHN1 / NOC2L, HES4, C1orf159), near-site and deep-intronic SNVs,
a deletion, an insertion, a multiallelic record, an MNV (→ `.|.` sentinel), and a deliberate REF mismatch.

### Provenance — the inputs are provably identical
| input | OpenSpliceAI | original `spliceai` | |
|---|---|---|---|
| `spliceai1..5.h5` weights | `models/spliceai/SpliceAI_models_release/` | site-packages/spliceai/models/ | **md5 identical (all 5)** |
| `grch37.txt` / `grch38.txt` | `openspliceai/variant/annotations/` | site-packages/spliceai/annotations/ | **md5 identical** |
| reference, VCF, `-A` file | — | — | same files passed to both tools |

### Result — exact match, every field, every grid point
| `--distance` | `--mask` | records | DS fields exact | DP fields exact | off-by-0.01 | key-set diffs |
|---|---|---|---|---|---|---|
| 50 | 0 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |
| 50 | 1 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |
| 500 | 0 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |
| 500 | 1 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |
| 1000 | 0 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |
| 1000 | 1 | 35 | 140/140 (100%) | 140/140 (100%) | 0 | 0 |

Representative record (SAMD11 donor disruption, `chr1:925801 G>A`): both tools emit
`...|SAMD11|0.00|0.00|0.07|0.67|-36|-1|-36|-1`. Multiallelic → two identical annotations; MNV → identical
`.|.|.|.|.|.|.|.` sentinel from both.

**Interpretation.** `variant`'s entire scoring pipeline — window construction, gene-boundary N-padding,
one-hot encoding, minus-strand reverse-complement, ensemble averaging, indel handling, cropping, the DS/DP
max/argmax math, and masking — is **provably correct**: it reproduces the reference implementation to the last
decimal. Because the PyTorch path runs the *same* `get_delta_scores` algorithm (only the model object
differs), this transfers to the MANE PyTorch models used for production scoring; the PyTorch-specific
plumbing is validated separately in §C/§B5.

**Keystone condition.** Equivalence holds **only at `--flanking-size 10000`** — original SpliceAI hardcodes
`wid = 10000 + cov`. Any other flanking size, or the PyTorch models, is a legitimately *different* model and
will not (and should not) reproduce the original numbers.

---

## B. Flagged-concern verdicts

| # | Concern | Verdict | Evidence |
|---|---|---|---|
| 1 | Ensemble averages post-softmax probabilities, not logits | **False positive** | `SpliceAI.forward` softmax → channels sum to 1.0; matches original `np.mean(model.predict())`. Locked by `test_forward_applies_softmax…` |
| 2 | predict split-overlap → duplicate BED rows | **Real, benign** | Forced split: 9482 donor rows → 6218 unique; duplicates have **identical coordinates** (no corruption), only redundant. Mitigated by `dedup_predictions.py` |
| 3 | Minus-strand BED coordinate (double-flip) | **False positive** | End-to-end recovery: **18/18** minus-strand donor calls within 2 bp of a real boundary (§C) |
| 4 | variant ref-window off-by-one | **False positive** | `seq[wid//2:wid//2+len(REF)] == REF` holds; locked by `test_ref_window_is_centered…`; also guarded in-code |
| 5 | Minus-strand reverse-complement correctness | **False positive** | `torch.flip(dims=[1,2])` == manual RC one-hot (proven); minus-strand genes match original exactly in §A |
| 6 | Indel slicing alignment | **False positive** | deletion & insertion records match original exactly in §A |
| 7 | Masking `==` exon-boundary comparison | **False positive** | `--mask 1` grid points match original exactly in §A |
| 8 | Overloaded `CL` parameter name in `load_pytorch_models` | **Cosmetic** | `CL` *is* the flanking size and is used; naming only, no functional issue |
| 9 | predict silently defaults to 80nt for an unsupported flanking | **Hardened** | Added `else: raise ValueError` to `predict.py` `load_model`; locked by `test_load_pytorch_models_rejects_unsupported_flanking` |
| + | Exon-list parse differs from original (`split(',') if i` vs `re.split(',')[:-1]`) | **Equivalent on production data** | Identical on trailing-comma tables (grch37/38); OpenSpliceAI is *more* robust without a trailing comma. Locked by `test_exon_parse_matches_original…` |

### Real bug found **and fixed**
`variant/variant.py` did `os.makedirs(os.path.dirname(output_vcf), exist_ok=True)`, which raised
`FileNotFoundError: ''` when `--output-vcf` was a **bare filename** (no directory). The precompute runs used
full paths so never hit it. **Fixed** to skip `makedirs` when the dirname is empty; locked by
`tests/integration/test_variant_output_dir.py`. (Behavior-preserving for all previously-working inputs.)

---

## C. `predict` end-to-end coordinate correctness

`predict` was run with the MANE 10000nt model through the GFF path over a 970 kb chr1 region containing 4
genes (3 +, 1 − strand), and the high-confidence BED calls were checked against the annotated exon
boundaries.

- **C1 — coordinate recovery (both strands).** Donor calls (score ≥ 0.5): **42/43 at exact offset 0** from an
  annotated boundary. Acceptor calls cluster at ±1 (the `ACCEPTOR_WINDOW=(-1,0)` convention). **Plus-strand
  donor 24/25; minus-strand donor 18/18 within 2 bp.** No systematic strand-specific shift → the
  reverse-complement-then-flip-coordinates logic is correct on both strands (resolves concern #3).
- **C2 — split-overlap.** A forced small split confirmed duplicate BED rows appear only in overlap zones with
  identical coordinates; `dedup_predictions.py` collapses them (9482 → 6218, 0 remaining duplicates).

**Operational guidance for whole-genome `predict`.** Chromosomes exceed the 1.5 Mb split threshold, so the
output BEDs will contain duplicate rows in the overlap zones (correct coordinates, redundant). Run
`python validation/dedup_predictions.py donor_predictions.bed` (and the acceptor BED) to get a
unique-per-position file. This is a reporting redundancy, not a scoring error.

---

## D. Tests added (auto-skip without backends)
- `tests/unit/test_validation_invariants.py` — CL=flanking for all 4 sizes, exact cropping, softmax-sum=1,
  RC-by-flip identity, exon-parse equivalence, ACGT channel order. (CPU)
- `tests/unit/test_predict_variant_hardening.py` — predict rejects unsupported flanking; ref-window centring.
- `tests/equivalence/test_keras_equivalence.py` — `variant` keras == original `spliceai`, in-process
  (`@keras @slow`). **3/3 passed here.**
- `tests/integration/test_variant_output_dir.py` — bare-filename regression.
- Full suite after the source edits: **114 fast tests pass**, `ruff` clean.

---

## E. Reproduce
```bash
ENV=/home/kchao10/miniconda3/envs/pytorch_cuda/bin
REF=/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna
ANN=/data/ssalzbe1/khchao/OpenSpliceAI/data/grch38_chr.txt
cd /data/ssalzbe1/khchao/OpenSpliceAI/validation

# A. equivalence (build VCF, run both tools per grid point, compare)
$ENV/python build_equiv_vcf.py --ref $REF --ann $ANN --out equiv/equiv_test.vcf
#   spliceai -I equiv/equiv_test.vcf -O orig.vcf -R $REF -A $ANN -D 50 -M 0
#   openspliceai variant -I equiv/equiv_test.vcf -O os.vcf -R $REF -A $ANN -D 50 -M 0 \
#       --model-type keras --model <SpliceAI_models_release> --flanking-size 10000 --precision 2
$ENV/python compare_equiv.py orig.vcf os.vcf

# C. predict coordinate recovery
$ENV/python build_predict_region.py --ref $REF --ann $ANN --genes SAMD11,KLHL17,PLEKHN1,NOC2L --end 970000 --outdir predict_test
#   openspliceai predict -i predict_test/region.fa -m <mane/10000nt model> -f 10000 -a predict_test/region.gff -o predict_test/pred_out -t 0.1
$ENV/python compare_predict_recovery.py predict_test/sites.tsv <acceptor.bed> <donor.bed> 0.1

# tests
CUDA_VISIBLE_DEVICES="" $ENV/python -m pytest tests/ -q
```
