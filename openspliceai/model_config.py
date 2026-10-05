"""Architecture schedules shared by training, transfer, calibration and inference."""
import numpy as np


def model_hyperparameters(flanking_size):
    """Return the published (L, N_GPUS, W, AR, batch size) configuration.

    Fresh arrays prevent callers from mutating another model's configuration.
    N_GPUS is a compatibility field in the legacy tuple. It neither enables
    multi-device execution nor truncates partial batches.
    """
    flank = int(flanking_size)
    schedules = {
        80: ([11]*4, [1]*4, 36),
        400: ([11]*8, [1]*4 + [4]*4, 36),
        2000: ([11]*8 + [21]*4, [1]*4 + [4]*4 + [10]*4, 24),
        10000: ([11]*8 + [21]*4 + [41]*4, [1]*4 + [4]*4 + [10]*4 + [25]*4, 12),
    }
    if flank not in schedules:
        raise ValueError(f"Unsupported flanking size: {flanking_size}")
    windows, rates, batch_size = schedules[flank]
    return 32, 2, np.asarray(windows), np.asarray(rates), batch_size
