#!/usr/bin/env python
"""Score a fully-custom (wild-type vs mutant) sequence pair with OpenSpliceAI.

This is the recipe for sequences that do NOT correspond to a reference locus (so they can't be
written as a VCF record for ``openspliceai variant``) -- e.g. designed constructs, or several
non-adjacent edits in a short window. It is the OpenSpliceAI equivalent of SpliceAI's
"custom sequence" instructions: one-hot encode the reference and the alternate window, run the
model on both, and take the per-position difference of the acceptor/donor channels.

If your edited window IS anchored to a reference genome, prefer writing it as a single
multi-nucleotide VCF record and running ``openspliceai variant`` (see ``custom_sequence_mnv.vcf``
and the "Scoring custom sequences" section of the variant docs) -- that handles flanking context
and gene strand for you.

Usage
-----
    python score_custom_sequence.py \
        --model  ../../models/openspliceai-mane/10000nt/ \
        --flanking-size 10000 \
        --ref  <wild-type sequence>  --alt  <mutant sequence>

``--ref`` and ``--alt`` must be the SAME length: ``flanking_size + L``, where ``L`` is the number
of positions you want scored (the model trims ``flanking_size/2`` from each end). Pad the ends
with ``N`` if you don't have the full genomic context. With no ``--ref/--alt`` a tiny built-in
demo (N-padded) runs so you can see the output shape.
"""
import argparse

import torch

from openspliceai.variant.utils import load_pytorch_models, one_hot_encode, setup_device


def score(models, device, seq):
    """Return the model's (L_out, 3) probability matrix [null, acceptor, donor] for ``seq``."""
    x = one_hot_encode(seq)[None, :].transpose(0, 2, 1)          # (1, 4, len(seq))
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        y = torch.mean(torch.stack([m(x).detach().cpu() for m in models]), axis=0)
    return y.permute(0, 2, 1).numpy()[0]                         # (L_out, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "-m", required=True, help="PyTorch checkpoint file or directory of checkpoints")
    ap.add_argument("--flanking-size", "-f", type=int, required=True, choices=[80, 400, 2000, 10000])
    ap.add_argument("--ref", help="wild-type sequence (length flanking_size + L)")
    ap.add_argument("--alt", help="mutant sequence (same length as --ref)")
    args = ap.parse_args()

    if bool(args.ref) != bool(args.alt):
        ap.error("provide both --ref and --alt, or neither (to run the built-in demo)")

    if not args.ref:
        # tiny demo: L=9 scored positions (the #15 example), N-padded to the model's context
        pad = "N" * (args.flanking_size // 2)
        args.ref = pad + "ATGATTCCT" + pad
        args.alt = pad + "ACGAATCCA" + pad
        print(f"[demo] scoring {len(args.ref)}-nt N-padded ATGATTCCT -> ACGAATCCA")

    if len(args.ref) != len(args.alt):
        ap.error(f"--ref ({len(args.ref)}) and --alt ({len(args.alt)}) must be the same length")

    device = setup_device()
    models = load_pytorch_models(args.model, args.flanking_size)

    y_ref = score(models, device, args.ref)
    y_alt = score(models, device, args.alt)
    d_acc = y_alt[:, 1] - y_ref[:, 1]   # acceptor delta per position (>0 gain, <0 loss)
    d_don = y_alt[:, 2] - y_ref[:, 2]   # donor delta per position

    # Summarize like the variant subcommand: strongest gain/loss for each event.
    print("scored positions:", len(d_acc))
    print(f"acceptor gain : +{d_acc.max():.4f} at position {int(d_acc.argmax())}")
    print(f"acceptor loss : -{(-d_acc).max():.4f} at position {int((-d_acc).argmax())}")
    print(f"donor gain    : +{d_don.max():.4f} at position {int(d_don.argmax())}")
    print(f"donor loss    : -{(-d_don).max():.4f} at position {int((-d_don).argmax())}")


if __name__ == "__main__":
    main()
