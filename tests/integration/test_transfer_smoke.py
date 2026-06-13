"""End-to-end smoke test for the `transfer` subcommand.

Runs transfer.transfer(args) for a single epoch starting from a real packaged 80nt
checkpoint on tiny synthetic train datasets, and asserts the per-epoch and best
checkpoints land in the location built by initialize_paths_inner:
    {output_dir}/SpliceAI_{project}_{flank}_{exp}_rs{seed}/{exp}/models/
"""
import types

import pytest

from openspliceai.transfer import transfer as tr


@pytest.mark.integration
@pytest.mark.slow
def test_transfer_one_epoch_writes_checkpoints(tmp_path, train_datasets, packaged_80nt_state):
    out = tmp_path / "trout"
    args = types.SimpleNamespace(
        epochs=1,
        scheduler="MultiStepLR",
        early_stopping=False,
        patience=2,
        output_dir=str(out),
        project_name="tr",
        exp_num="0",
        flanking_size=80,
        random_seed=42,
        pretrained_model=packaged_80nt_state,
        train_dataset=train_datasets["train"],
        test_dataset=train_datasets["test"],
        loss="cross_entropy_loss",
        unfreeze=1,
        unfreeze_all=True,
    )

    tr.transfer(args)

    # Path grounded against initialize_paths_inner: SpliceAI_{project}_{flank}_{exp}_rs{seed}/{exp}/models/
    models_dir = out / "SpliceAI_tr_80_0_rs42" / "0" / "models"
    assert models_dir.is_dir()
    assert (models_dir / "model_0.pt").exists()
    assert (models_dir / "model_best.pt").exists()
