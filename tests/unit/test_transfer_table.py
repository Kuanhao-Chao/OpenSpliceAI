"""Coverage for the transfer model/optimiser builder across all flanking sizes + both
schedulers (the smoke test only exercises 80nt + MultiStepLR)."""
import torch
import pytest

from openspliceai.transfer.transfer import initialize_model_and_optim_transfer


@pytest.mark.parametrize("flank", [80, 400, 2000, 10000])
def test_transfer_builder_each_flanking_size(packaged_80nt_state, flank):
    """Build every flanking architecture; the 80nt checkpoint loads with strict=False
    (size-mismatched keys filtered) so larger sizes initialise cleanly."""
    model, optimizer, scheduler, params = initialize_model_and_optim_transfer(
        torch.device("cpu"), flank, 10, "MultiStepLR", packaged_80nt_state,
        unfreeze=0, unfreeze_all=True)
    assert params["CL"] == flank and params["L"] == 32
    assert optimizer is not None and scheduler is not None


def test_transfer_builder_cosine_scheduler(packaged_80nt_state):
    _model, _opt, scheduler, _params = initialize_model_and_optim_transfer(
        torch.device("cpu"), 80, 10, "CosineAnnealingWarmRestarts", packaged_80nt_state,
        unfreeze=2, unfreeze_all=False)   # also exercises the partial-unfreeze path
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
