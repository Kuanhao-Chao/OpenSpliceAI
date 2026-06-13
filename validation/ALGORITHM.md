# OpenSpliceAI `predict` and `variant` — algorithm reference

A step-by-step description of how the two scoring subcommands turn DNA into splice-site predictions and
variant delta scores, with the exact array math and coordinate conventions. File:line references point at
the current source. Companion: `VALIDATION_REPORT.md` (correctness evidence).

---

## 0. Shared foundations

### The model — `SpliceAI` (`openspliceai/train_base/openspliceai.py:83`)
A deep residual dilated 1-D CNN. Forward shape flow for a batch of one-hot DNA:

```
input  (N, 4, SL+CL)                     # 4 = one-hot A,C,G,T ; CL = flanking context
 ├─ Conv1d(4 → L=32, k=1)                (N, 32, SL+CL)
 ├─ ResidualUnit × {4,8,12,16}           (N, 32, SL+CL)   # BN→LeakyReLU(0.1)→Conv1d, twice, + residual
 │    + Skip every 4 units               accumulates 1×1 conv projections
 ├─ Cropping1D(CL//2 each end)           (N, 32, SL)      # trims the flanking context
 ├─ Conv1d(32 → 3, k=1)                  (N, 3, SL)
 └─ softmax over the 3 channels          (N, 3, SL)
```

- **Channel order is fixed: 0 = non-splice, 1 = acceptor, 2 = donor** (`SpliceAI.forward`; used identically
  in `variant/utils.py:562` and `predict.py:803`). This ordering makes the reverse-complement-by-flip identity
  hold (§2.3).
- **Output is post-softmax probabilities** (each position sums to 1 across the 3 channels). Consequence:
  averaging several models' outputs averages *probabilities* — matching original Illumina SpliceAI.
- **Context-length invariant:** `CL = 2·Σ(AR·(W−1))` and **equals the flanking size exactly**. The
  `(W, AR)` schedule is keyed by `--flanking-size ∈ {80,400,2000,10000}` → `{4,8,12,16}` residual units and is
  **duplicated, byte-for-byte identically,** in `predict.py` (`load_model`, ~line 487) and `variant/utils.py`
  (~line 61). Global constants: `CL_max=10000`, `SL=5000` (`openspliceai/constants.py`).

### One-hot encoding
- `variant`: `one_hot_encode` (`variant/utils.py`) maps A,C,G,T→ rows of the identity; N/other → all-zero.
- `predict`/`create-data`: `IN_MAP` (`create_data/utils.py`) maps A,C,G,T→1,2,3,4, N→0, then indexes the
  4-vector table. Both yield the same A,C,G,T channel order.

---

## 1. `variant` — VCF delta-score annotation

Entry `variant/variant.py:variant(args)`; core `variant/utils.py:get_delta_scores` (~line 396). Annotates each
VCF record's INFO with `OpenSpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`.

### 1.1 Setup (`Annotator.__init__`, `variant/utils.py:254`)
Loads (a) the reference genome via pyfaidx; (b) the gene annotation table — builtin `grch37`/`grch38`
(`_resolve_builtin_annotation`) or a custom TSV with `#NAME,CHROM,STRAND,TX_START,TX_END,EXON_START,EXON_END`
(`TX_START`/`EXON_START` are 0-based in the file, `+1` on load); (c) the model(s): PyTorch state_dicts built
into `SpliceAI` (default), or Keras `.h5` (`--model-type keras`, including the bundled original SpliceAI via
`--model SpliceAI`). A directory of checkpoints is ensembled.

### 1.2 Per-record geometry (`get_delta_scores`)
```
cov = 2·dist + 1                 # window centred on the variant (dist = --distance)
wid = flanking_size + cov        # full model input width
seq = ref_fasta[chrom][pos - wid//2 - 1 : pos + wid//2]   # length wid, REF base at index wid//2
```
Guards: skip if `seq[wid//2 : wid//2+len(REF)] != REF` (ref mismatch), `len(seq) != wid` (chromosome end), or
`len(REF) > 2·dist` (REF too long).

### 1.3 Per allele × gene
- Skip alleles containing `. - * < >`; for an MNV (`len(REF)>1 and len(ALT)>1`) emit the sentinel
  `ALT|SYMBOL|.|.|.|.|.|.|.|.`.
- **Gene-boundary N-padding:** `pad_size` is derived from the variant's distance to the gene's TX start/end so
  positions outside the annotated gene are masked to `N` (the model gets no signal from outside the gene).
- Build `x_ref` (the padded reference window) and `x_alt = x_ref[:wid//2] + ALT + x_ref[wid//2+ref_len:]`
  (substitute the allele at the centre). One-hot encode → `(1, wid, 4)`.

### 1.4 Inference + strand (`variant/utils.py:483–533`)
- **PyTorch:** transpose to `(1, 4, wid)`; for a minus-strand gene `torch.flip(dims=[1,2])` (reverse the
  length axis *and* the channel axis = reverse-complement); ensemble = `mean` over models of the post-softmax
  output; permute back to `(1, wid, 3)`; for minus strand `torch.flip(dims=[1])` the predictions back.
- **Keras:** `x[:, ::-1, ::-1]` for minus strand; `np.mean` over models; `y[:, ::-1]` back.

### 1.5 Crop, indels, delta scores
- Crop the output to the `cov` window centred on the variant.
- **Indels:** deletion → insert `del_len` zero rows at the centre; insertion → collapse the inserted span to
  its per-channel max. (Identical to original SpliceAI.)
- Stack `y = [y_ref, y_alt]`, shape `(2, cov, 3)`. With acceptor=channel 1, donor=channel 2:
```
DS_AG = max_p (y_alt[p,1] − y_ref[p,1])     # acceptor gain   (idx_pa = argmax)
DS_AL = max_p (y_ref[p,1] − y_alt[p,1])     # acceptor loss   (idx_na)
DS_DG = max_p (y_alt[p,2] − y_ref[p,2])     # donor gain      (idx_pd)
DS_DL = max_p (y_ref[p,2] − y_alt[p,2])     # donor loss      (idx_nd)
DP_*  = (argmax position) − cov//2          # signed offset from the variant
```
- **Masking (`--mask 1`):** zero a *gain* whose position is at an annotated exon boundary, and a *loss* whose
  position is **not** at a boundary (`(idx − cov//2 == dist_exon_bdry)` etc.). `--mask 0` keeps all four.
- Format each DS to `--precision` decimals; DP as signed ints. Multiple overlapping genes → comma-joined
  annotations. Reads stdin / writes stdout by default.

---

## 2. `predict` — genome/FASTA splice-site BED prediction

Entry `predict/predict.py:predict_cli`; scales to whole genomes. Two modes: **turbo** (`predict_and_write`,
streamed, default) and `--predict-all` (writes `predict.h5` then `generate_bed`).

### 2.1 Sequence extraction
- **With `-a/--annotation` GFF** (`process_gff`, `predict.py:20`): extract every `gene` feature's sequence;
  **reverse-complement minus-strand genes** so the model always sees 5'→3'.
- **Without GFF:** use the FASTA records directly.

### 2.2 Splitting long sequences (`split_fasta`, `predict.py:86`)
Sequences longer than `--split-threshold` (default 1.5 Mb) are cut into segments that **overlap by
`CL_max//2`** so predictions stay seamless across the cut. Segment headers carry absolute genome coordinates.
*(Consequence: positions in overlap zones are predicted — and written — twice; see VALIDATION_REPORT §C2.)*

### 2.3 One-hot windows (`create_datapoints`, `predict.py:253`)
Pad each sequence with `CL_max//2` `N` on each end, integer-encode, and tile into windows of length
`SL + CL_max` spaced by `SL`. Because the model crops `CL_max//2` per side, the *output* windows are
contiguous and non-overlapping and reassemble to one prediction track per input sequence. Stored chunked in
`dataset.h5` (or `.pt`), 100 sequences per chunk; `LEN[i]` = number of windows for sequence i.

### 2.4 Inference (`load_pytorch_models` + the prediction loop)
Build `SpliceAI` from `--flanking-size` (now raises on an unsupported size — see VALIDATION_REPORT §B9);
batches transposed to `(N, 4, SL+CL_max)`; **ensemble = `mean` of post-softmax outputs** over all checkpoints
in a `--model` directory. Output per window `(N, 3, SL)`.

### 2.5 BED coordinate mapping (`write_batch_to_bed`, `predict.py:766`)
Flatten predictions to `(positions, 3)`; acceptor = channel 1, donor = channel 2. `ACCEPTOR_WINDOW=(-1,0)`,
`DONOR_WINDOW=(0,1)` define the 1-bp interval reported. From a header `…chrN:START-END(strand)…`:
```
strand +:  bed_start = (START − 1) + seq_pos          # 0-based
strand −:  bed_start = END − seq_pos                  # flips the RC'd local position back to the genome
```
Each position scoring above `--threshold` is written as
`chrom  start  end  NAME_{Acceptor|Donor}  score  strand` to `acceptor_predictions.bed` /
`donor_predictions.bed`. For a minus-strand gene this single inversion *undoes* the §2.1 reverse-complement —
not a double flip (validated end-to-end, VALIDATION_REPORT §C1).

---

## 3. The equivalence condition (most important operational fact)

OpenSpliceAI `variant --model-type keras` reproduces the **original Illumina SpliceAI** scores **exactly — but
only at `--flanking-size 10000`**, because original SpliceAI hardcodes `wid = 10000 + cov`. At any other
flanking size, or with the PyTorch models, OpenSpliceAI is a legitimately *different* model and will not (and
should not) reproduce the original numbers. See `VALIDATION_REPORT.md` for the proof.
