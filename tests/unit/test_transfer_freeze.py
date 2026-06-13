"""Unit tests for layer-freezing logic in
openspliceai/transfer/transfer.py::initialize_model_and_optim_transfer.

Grounded by running the function on the real packaged 80nt checkpoint: the 80nt
schedule has W/AR of length 4, so ``model.residual_units`` is
[ResidualUnit, ResidualUnit, ResidualUnit, ResidualUnit, Skip] (a Skip is inserted
after every 4th ResidualUnit). The freeze logic selects the last ``unfreeze``
ResidualUnit instances (skipping Skip layers).
"""
import torch

from openspliceai.transfer.transfer import initialize_model_and_optim_transfer
from openspliceai.train_base.openspliceai import ResidualUnit, Skip


def _build(packaged_80nt_state, unfreeze, unfreeze_all):
    return initialize_model_and_optim_transfer(
        device=torch.device("cpu"),
        flanking_size=80,
        epochs=10,
        scheduler="MultiStepLR",
        pretrained_model=packaged_80nt_state,
        unfreeze=unfreeze,
        unfreeze_all=unfreeze_all,
    )


def _residual_units(model):
    return [m for m in model.residual_units if isinstance(m, ResidualUnit)]


def test_unfreeze_all_keeps_every_param_trainable(packaged_80nt_state):
    model, optimizer, scheduler, params = _build(packaged_80nt_state, unfreeze=1, unfreeze_all=True)
    assert all(p.requires_grad for p in model.parameters())
    # sanity: the 80nt schedule yields 4 ResidualUnits + 1 Skip
    types = [type(m).__name__ for m in model.residual_units]
    assert types == ["ResidualUnit", "ResidualUnit", "ResidualUnit", "ResidualUnit", "Skip"]


def test_unfreeze_two_trains_only_last_two_residual_units(packaged_80nt_state):
    model, _, _, _ = _build(packaged_80nt_state, unfreeze=2, unfreeze_all=False)
    res_units = _residual_units(model)
    assert len(res_units) == 4

    # The last 2 ResidualUnits are fully trainable; the first 2 are fully frozen.
    trainable = [any(p.requires_grad for p in ru.parameters()) for ru in res_units]
    assert trainable == [False, False, True, True]
    for ru in res_units[-2:]:
        assert all(p.requires_grad for p in ru.parameters())

    # Those unfrozen modules are ResidualUnit instances, not Skip.
    for ru in res_units[-2:]:
        assert isinstance(ru, ResidualUnit)
        assert not isinstance(ru, Skip)

    # Every other parameter in the whole model is frozen.
    unfrozen_ids = {id(p) for ru in res_units[-2:] for p in ru.parameters()}
    others = [p for p in model.parameters() if id(p) not in unfrozen_ids]
    assert others, "expected the model to have other (frozen) parameters"
    assert all(not p.requires_grad for p in others)


def test_unfreeze_one_trains_only_last_residual_unit(packaged_80nt_state):
    model, _, _, _ = _build(packaged_80nt_state, unfreeze=1, unfreeze_all=False)
    res_units = _residual_units(model)
    assert len(res_units) == 4

    trainable = [any(p.requires_grad for p in ru.parameters()) for ru in res_units]
    assert trainable == [False, False, False, True]

    last = res_units[-1]
    assert isinstance(last, ResidualUnit) and not isinstance(last, Skip)
    assert all(p.requires_grad for p in last.parameters())

    unfrozen_ids = {id(p) for p in last.parameters()}
    others = [p for p in model.parameters() if id(p) not in unfrozen_ids]
    assert all(not p.requires_grad for p in others)


def test_optimizer_only_holds_trainable_params(packaged_80nt_state):
    """AdamW is built from filter(requires_grad). With unfreeze=1 it must hold exactly
    the last ResidualUnit's parameters and nothing else."""
    model, optimizer, _, _ = _build(packaged_80nt_state, unfreeze=1, unfreeze_all=False)
    last = _residual_units(model)[-1]
    expected_ids = {id(p) for p in last.parameters()}
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert opt_ids == expected_ids
