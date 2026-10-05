"""Fit class temperatures on validation data and report complete-split calibration."""
import json
from pathlib import Path
from contextlib import ExitStack, nullcontext

import h5py

from openspliceai.checkpoints import calibrated_checkpoint, atomic_torch_save
from openspliceai.data_schema import shard_indices
from openspliceai.train_base.utils import setup_environment, resolve_validation_dataset
from openspliceai.calibrate.temperature_scaling import ModelWithTemperature, get_validation_loader
from openspliceai.calibrate.streaming import LogitCache, evaluate_cache
from openspliceai.calibrate.model_utils import initialize_model_and_optim
from openspliceai.calibrate.calibrate_utils import save_calibration_data
from openspliceai.calibrate.visualization import (plot_score_distribution, plot_calibration_curves,
                                                plot_brier_scores, plot_calibration_map)


def get_logits_labels(model, loader, device, params, maximum_observations=100000):
    """Return small explicit logits/labels tensors; use LogitCache for larger splits."""
    with LogitCache(model, loader, device, params) as cache:
        if len(cache) > maximum_observations:
            raise ValueError('Explicit logits exceed the memory bound; use the streaming LogitCache interface')
        logits, labels = cache.preview(maximum_observations)
        return logits.to(device), labels.to(device)


def evaluate_and_visualize(calibrated_model, data_loader, device, output_base_dir,
                           dataset_name, params, flanking_size, cache=None):
    """Report exact full-split metrics/curves and bounded sampled score histograms."""
    results = Path(output_base_dir)/'results'/dataset_name
    curves_dir, plots_dir = results/'calibration_data', results/'plots'
    curves_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    context = nullcontext(cache) if cache is not None else LogitCache(
        calibrated_model.model, data_loader, device, params)
    with context as cached:
        original, scaled, probs, probs_scaled, labels = evaluate_cache(
            cached, calibrated_model.temperature_scale, device, seed=params.get('RANDOM_SEED', 42))
        for name, stats in (('original', original), ('calibrated', scaled)):
            prefix = name.capitalize()
            (results/f'metrics_{name}.txt').write_text(
                f'{prefix}_NLL\t{prefix}_ECE\n{stats.nll:.8f}\t{stats.ece:.8f}\n')
        classes = ['Non-splice site', 'Acceptor site', 'Donor site']
        before_curves, after_curves = [], []
        for index, name in enumerate(classes):
            before_curves.append(original.curve(index))
            after_curves.append(scaled.curve(index))
            save_calibration_data(str(curves_dir), name, flanking_size, *before_curves[-1], 'original')
            save_calibration_data(str(curves_dir), name, flanking_size, *after_curves[-1], 'calibrated')
            plot_score_distribution(probs, probs_scaled, labels, str(plots_dir), index)
        plot_calibration_curves(before_curves, after_curves, classes, str(plots_dir))
        plot_brier_scores(original.brier, scaled.brier, classes, str(plots_dir))
        plot_calibration_map(calibrated_model, device, str(plots_dir))
        (results/'summary.json').write_text(json.dumps({
            'observations': original.count, 'plot_sample_count': len(labels),
            'plot_sample_limit': 100000, 'plot_sample_seed': params.get('RANDOM_SEED', 42),
            'metrics_scope': 'complete_selected_split', 'histogram_scope': 'bounded_random_sample',
            'original': {'nll': original.nll, 'ece': original.ece, 'brier': original.brier.tolist()},
            'calibrated': {'nll': scaled.nll, 'ece': scaled.ece, 'brier': scaled.brier.tolist()},
        }, indent=2)+'\n')


def calibrate(args):
    """Fit on validation only, evaluate test afterward, and publish portable checkpoints.

    Side effects: temperature.pt/.txt, calibrated_model.pt, optimization.json,
    and exact metrics/curves plus sampled histograms under calibration/results/.
    Temporary logits are removed on successful and failed exits.
    """
    if getattr(args, 'loss', 'cross_entropy_loss') != 'cross_entropy_loss':
        raise ValueError('Calibration uses negative log likelihood, not focal loss')
    device = setup_environment(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        validation = stack.enter_context(h5py.File(resolve_validation_dataset(args), 'r'))
        test = stack.enter_context(h5py.File(args.test_dataset, 'r'))
        model, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model)
        params['RANDOM_SEED'] = getattr(args, 'random_seed', 42)
        wrapped = ModelWithTemperature(model, 3).to(device)
        valid_loader = get_validation_loader(validation, shard_indices(validation), params['BATCH_SIZE'])
        test_loader = get_validation_loader(test, shard_indices(test), params['BATCH_SIZE'])
        with LogitCache(model, valid_loader, device, params, directory=output) as cache:
            if args.temperature_file:
                wrapped.load_temperature(args.temperature_file)
            else:
                wrapped.fit_cache(cache, getattr(args, 'epochs', 10),
                                  getattr(args, 'early_stopping', False), getattr(args, 'patience', 2))
            atomic_torch_save(wrapped.temperature.detach().cpu(), output/'temperature.pt')
            (output/'temperature.txt').write_text(str(wrapped.temperature.detach().cpu().tolist())+'\n')
            atomic_torch_save(calibrated_checkpoint(model, wrapped.temperature, args.flanking_size),
                              output/'calibrated_model.pt')
            (output/'optimization.json').write_text(json.dumps({
                'objective': 'validation_negative_log_likelihood', 'observations': len(cache),
                'epochs_requested': getattr(args, 'epochs', 10), 'epochs_completed': max(0, len(wrapped.history)-1),
                'early_stopping': getattr(args, 'early_stopping', False), 'patience': getattr(args, 'patience', 2),
                'temperature_restored': bool(args.temperature_file),
                'project_name': getattr(args, 'project_name', None), 'exp_num': getattr(args, 'exp_num', None),
                'history': wrapped.history,
            }, indent=2)+'\n')
            evaluate_and_visualize(wrapped, valid_loader, device, output/'calibration', 'validation',
                                   params, args.flanking_size, cache=cache)
        evaluate_and_visualize(wrapped, test_loader, device, output/'calibration', 'test', params, args.flanking_size)
