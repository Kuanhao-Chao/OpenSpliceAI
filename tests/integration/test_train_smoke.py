"""End-to-end smoke test: run a real 1-epoch training run on tiny synthetic shards.

Exercises the full train pipeline (load datasets -> indices -> build model/optimizer ->
train/validate/test epoch -> checkpointing + metric logging). The output directory layout
asserted below was grounded by reading initialize_paths_inner in train_base/utils.py and by
running train.train once on the train_datasets fixture.
"""
import types

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_train_one_epoch_writes_checkpoints_and_logs(tmp_path, train_datasets):
    from openspliceai.train import train

    out = tmp_path / "trainout"
    out.mkdir()

    args = types.SimpleNamespace(
        epochs=1,
        scheduler="MultiStepLR",
        early_stopping=False,
        patience=2,
        output_dir=str(out),
        project_name="t",
        exp_num="0",
        flanking_size=80,
        random_seed=42,
        # train derives the validation path by swapping 'train'->'validation'; the
        # fixture already writes dataset_validation.h5 in the same directory.
        train_dataset=train_datasets["train"],
        test_dataset=train_datasets["test"],
        loss="cross_entropy_loss",
        model="SpliceAI",
    )

    train.train(args)

    # initialize_paths_inner: {output_dir}/SpliceAI_{project}_{flank}_{exp}_rs{seed}/{exp}/
    run_dir = out / "SpliceAI_t_80_0_rs42" / "0"
    models_dir = run_dir / "models"
    assert models_dir.is_dir()

    # One checkpoint per epoch (model_{epoch}.pt) plus the best-model checkpoint.
    assert (models_dir / "model_0.pt").exists()
    assert (models_dir / "model_best.pt").exists()

    # The saved checkpoints are plain state_dicts (loadable into a fresh SpliceAI).
    import torch
    from openspliceai.train_base.openspliceai import SpliceAI
    import numpy as np

    state = torch.load(models_dir / "model_best.pt", map_location="cpu")
    assert isinstance(state, dict)
    fresh = SpliceAI(32, np.asarray([11, 11, 11, 11]), np.asarray([1, 1, 1, 1]))
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert not missing and not unexpected

    # Metric .txt files are written under LOG/TRAIN.
    train_log = run_dir / "LOG" / "TRAIN"
    assert train_log.is_dir()
    txt_files = list(train_log.glob("*.txt"))
    assert txt_files, "expected metric .txt files under LOG/TRAIN"
    # A couple of the named metric files from create_metric_files must be present.
    assert (train_log / "loss_batch.txt").exists()
    assert (train_log / "acceptor_accuracy.txt").exists()

    # VAL and TEST log directories are created too.
    assert (run_dir / "LOG" / "VAL").is_dir()
    assert (run_dir / "LOG" / "TEST").is_dir()
