import torch
import torchmetrics
from typing import Dict, Any, Optional, cast
from utils.metrics import update_perplexity_from_ntp


def init_metrics(configs, accelerator) -> Dict[str, Any]:
    """Initialize optional auxiliary metrics used in training/validation.

    Args:
        configs: Hydrated config object with loss toggles (NTP, TikTok).
        accelerator: ``accelerate.Accelerator`` whose device hosts the metrics.

    Returns:
        Dict[str, Any]: Optional metric instances keyed by name. Coordinate
        metrics are intentionally omitted because the active RNA objective is
        tracked directly through loss terms. Only language-model style metrics
        remain when their heads are enabled.
    """
    metrics: Dict[str, Any] = {
        'perplexity': None,
        'tik_tok_padding_accuracy': None,
    }

    if getattr(configs.train_settings, 'losses', None) and \
            getattr(configs.train_settings.losses, 'next_token_prediction', None) and \
            configs.train_settings.losses.next_token_prediction.enabled:
        from torchmetrics.text import Perplexity
        # cast to Any to satisfy type checkers that try to unify dict value types
        metrics['perplexity'] = cast(Any, Perplexity(ignore_index=-100).to(accelerator.device))

    tik_tok_cfg = getattr(configs.model.vqvae.vector_quantization, 'tik_tok', None)
    compression_factor = getattr(configs.model.vqvae.vector_quantization.tik_tok, 'compression_factor', 1)
    if tik_tok_cfg is not None and getattr(tik_tok_cfg, 'enabled', False) and compression_factor > 1:
        from torchmetrics.classification import MulticlassAccuracy
        num_classes = int(getattr(tik_tok_cfg, 'compression_factor', 1))
        metrics['tik_tok_padding_accuracy'] = cast(
            Any,
            MulticlassAccuracy(num_classes=num_classes, average='macro').to(accelerator.device),
        )

    return metrics


def reset_metrics(metrics: Dict[str, Any]) -> None:
    """Reset all metric states (to start a new epoch or phase).

    Args:
        metrics: Dict returned by :func:`init_metrics`.
    """
    if metrics.get('perplexity') is not None:
        metrics['perplexity'].reset()
    if metrics.get('tik_tok_padding_accuracy') is not None:
        metrics['tik_tok_padding_accuracy'].reset()


def update_metrics(metrics: Dict[str, Any],
                   trans_pred_coords: torch.Tensor,
                   trans_true_coords: torch.Tensor,
                   masks: torch.Tensor,
                   output_dict: Dict[str, torch.Tensor],
                   ignore_index: int = -100) -> None:
    """Update optional auxiliary metric states with a new batch.

    Args:
        metrics: Dict returned by :func:`init_metrics`.
        trans_pred_coords: Predicted coordinates ``(B, L, 3_atoms, 3_xyz)``.
        trans_true_coords: Ground-truth coordinates with the same shape.
        masks: Boolean/float mask ``(B, L)`` of valid residues.
        output_dict: Model outputs; may contain NTP logits/indices/masks and
            TikTok classifier logits/targets.
        ignore_index: Index ignored when computing perplexity.

    Notes:
        - Coordinate metrics are intentionally disabled in the active RNA path.
        - Updates perplexity when the metric exists and NTP outputs are present.
        - Updates padding-accuracy when the TikTok metric exists and logits / targets are available.
    """
    del trans_pred_coords, trans_true_coords, masks

    # Optional perplexity from NTP
    if metrics.get('perplexity') is not None:
        update_perplexity_from_ntp(
            metrics['perplexity'],
            output_dict.get('ntp_logits', None),
            output_dict.get('indices', None),
            output_dict.get('ntp_mask', None),
            ignore_index=ignore_index,
        )

    tik_tok_metric = metrics.get('tik_tok_padding_accuracy', None)
    if tik_tok_metric is not None:
        logits = output_dict.get('tik_tok_padding_logits', None)
        targets = output_dict.get('tik_tok_padding_targets', None)
        if logits is not None and targets is not None and targets.numel() > 0:
            tik_tok_metric.update(logits.detach(), targets.detach())


def compute_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Compute scalar values from metric states.

    Args:
        metrics: Dict returned by :func:`init_metrics`.

    Returns:
        Dict[str, float]: Optional non-structural metrics. Metrics that were not
        enabled are reported as ``NaN``.
    """
    out = {
        'perplexity': float('nan'),
        'tik_tok_padding_accuracy': float('nan'),
    }
    if metrics.get('perplexity') is not None:
        out['perplexity'] = metrics['perplexity'].compute().cpu().item()
    if metrics.get('tik_tok_padding_accuracy') is not None:
        out['tik_tok_padding_accuracy'] = metrics['tik_tok_padding_accuracy'].compute().cpu().item()
    return out


def init_accumulator(accum_iter: int) -> Dict[str, Any]:
    """Create a running accumulator dict for losses and codebook activation.

    Inputs:
    - accum_iter: Number of micro-batches per optimizer step (gradient accumulation).

    Returns:
    - Dict[str, Any] with keys tracking per-accumulation values, totals, a step counter,
      and a running set of unique codebook indices.
    """
    return {
        # per-accumulation running values (averaged per micro-step)
        'train_step_loss': 0.0,
        'train_rec_loss': 0.0,
        'train_final_fape_loss': 0.0,
        'train_aux_fape_loss': 0.0,
        'train_vq_loss': 0.0,
        'train_ntp_loss': 0.0,
        'train_tik_tok_padding_loss': 0.0,
        # unscaled per-accumulation running values
        'train_unscaled_step_loss': 0.0,
        'train_unscaled_rec_loss': 0.0,
        'train_unscaled_final_fape_loss': 0.0,
        'train_unscaled_aux_fape_loss': 0.0,
        'train_unscaled_vq_loss': 0.0,
        'train_unscaled_ntp_loss': 0.0,
        'train_unscaled_tik_tok_padding_loss': 0.0,
        # totals across finalized steps
        'total_step_loss': 0.0,
        'total_rec_loss': 0.0,
        'total_final_fape_loss': 0.0,
        'total_aux_fape_loss': 0.0,
        'total_vq_loss': 0.0,
        'total_ntp_loss': 0.0,
        'total_tik_tok_padding_loss': 0.0,
        # unscaled totals across finalized steps
        'total_unscaled_step_loss': 0.0,
        'total_unscaled_rec_loss': 0.0,
        'total_unscaled_final_fape_loss': 0.0,
        'total_unscaled_aux_fape_loss': 0.0,
        'total_unscaled_vq_loss': 0.0,
        'total_unscaled_ntp_loss': 0.0,
        'total_unscaled_tik_tok_padding_loss': 0.0,
        'counter': 0,
        'accum_iter': accum_iter,
        'unique_indices': set(),
    }


def _gather_mean(accelerator, value: torch.Tensor, repeat: Optional[int] = None) -> torch.Tensor:
    """All-gather a tensor across processes and return its mean.

    Inputs:
    - accelerator: accelerate.Accelerator for distributed gather.
    - value: Scalar or 1D batch-sized tensor.
    - repeat: If set, repeat gathered values this many times before averaging (emulates
      legacy averaging behavior that scaled by batch size).

    Returns:
    - torch.Tensor: Scalar tensor of the mean value across processes.
    """
    # value is scalar tensor or batch-sized tensor; we gather then optionally repeat to emulate original averaging pattern
    if value.dim() == 0:
        value = value.unsqueeze(0)
    gathered = accelerator.gather(value.detach())
    if repeat is not None and repeat > 1:
        gathered = gathered.repeat(repeat)
    return gathered.mean()


def accumulate_losses(acc: Dict[str, Any],
                      loss_dict: Dict[str, torch.Tensor],
                      output_dict: Dict[str, torch.Tensor],
                      configs,
                      accelerator,
                      use_output_vq: bool = False) -> None:
    """Accumulate loss values for the current micro-batch.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    - loss_dict: Dict with 'step_loss', 'rec_loss', 'vq_loss', and optionally 'ntp_loss' tensors.
    - output_dict: Model outputs, used to source 'vq_loss' when use_output_vq=True (validation pattern).
    - configs: Global config to access train_settings.batch_size.
    - accelerator: accelerate.Accelerator for distributed mean.
    - use_output_vq: If True, use output_dict['vq_loss'] instead of loss_dict['vq_loss'].

    Behavior:
    - Scales each loss contribution by 1/acc['accum_iter'] for gradient accumulation.
    - Averages across processes to get a stable scalar before accumulating.
    """
    bs = configs.train_settings.batch_size
    acc['train_step_loss'] += _gather_mean(accelerator, loss_dict['step_loss'], repeat=bs).item() / acc['accum_iter']
    acc['train_rec_loss'] += _gather_mean(accelerator, loss_dict['rec_loss'], repeat=bs).item() / acc['accum_iter']
    # Validation and training should use the same scaled VQ term for logging.
    vq_src = loss_dict['vq_loss']
    acc['train_vq_loss'] += _gather_mean(accelerator, vq_src, repeat=bs).item() / acc['accum_iter']
    if 'final_fape_loss' in loss_dict and loss_dict['final_fape_loss'] is not None:
        acc['train_final_fape_loss'] += _gather_mean(
            accelerator, loss_dict['final_fape_loss'], repeat=bs
        ).item() / acc['accum_iter']
    if 'aux_fape_loss' in loss_dict and loss_dict['aux_fape_loss'] is not None:
        acc['train_aux_fape_loss'] += _gather_mean(
            accelerator, loss_dict['aux_fape_loss'], repeat=bs
        ).item() / acc['accum_iter']
    # ntp loss may be missing; default to zero
    if 'ntp_loss' in loss_dict and loss_dict['ntp_loss'] is not None:
        acc['train_ntp_loss'] += _gather_mean(accelerator, loss_dict['ntp_loss'], repeat=bs).item() / acc['accum_iter']
    if 'tik_tok_padding_loss' in loss_dict and loss_dict['tik_tok_padding_loss'] is not None:
        acc['train_tik_tok_padding_loss'] += _gather_mean(
            accelerator,
            loss_dict['tik_tok_padding_loss'],
            repeat=bs,
        ).item() / acc['accum_iter']

    # Unscaled contributions
    if 'unscaled_step_loss' in loss_dict:
        acc['train_unscaled_step_loss'] += _gather_mean(accelerator, loss_dict['unscaled_step_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_rec_loss' in loss_dict:
        acc['train_unscaled_rec_loss'] += _gather_mean(accelerator, loss_dict['unscaled_rec_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_vq_loss' in loss_dict:
        unscaled_vq_src = loss_dict['unscaled_vq_loss']
        acc['train_unscaled_vq_loss'] += _gather_mean(accelerator, unscaled_vq_src, repeat=bs).item() / acc['accum_iter']
    if 'unscaled_final_fape_loss' in loss_dict and loss_dict['unscaled_final_fape_loss'] is not None:
        acc['train_unscaled_final_fape_loss'] += _gather_mean(
            accelerator, loss_dict['unscaled_final_fape_loss'], repeat=bs
        ).item() / acc['accum_iter']
    if 'unscaled_aux_fape_loss' in loss_dict and loss_dict['unscaled_aux_fape_loss'] is not None:
        acc['train_unscaled_aux_fape_loss'] += _gather_mean(
            accelerator, loss_dict['unscaled_aux_fape_loss'], repeat=bs
        ).item() / acc['accum_iter']
    if 'unscaled_ntp_loss' in loss_dict and loss_dict['unscaled_ntp_loss'] is not None:
        acc['train_unscaled_ntp_loss'] += _gather_mean(accelerator, loss_dict['unscaled_ntp_loss'], repeat=bs).item() / acc['accum_iter']
    if 'unscaled_tik_tok_padding_loss' in loss_dict and loss_dict['unscaled_tik_tok_padding_loss'] is not None:
        acc['train_unscaled_tik_tok_padding_loss'] += _gather_mean(
            accelerator,
            loss_dict['unscaled_tik_tok_padding_loss'],
            repeat=bs,
        ).item() / acc['accum_iter']


def finalize_step(acc: Dict[str, Any]) -> None:
    """Move per-accumulation running sums into epoch totals and reset micro-trackers.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    """
    acc['total_step_loss'] += acc['train_step_loss']
    acc['total_rec_loss'] += acc['train_rec_loss']
    acc['total_final_fape_loss'] += acc['train_final_fape_loss']
    acc['total_aux_fape_loss'] += acc['train_aux_fape_loss']
    acc['total_vq_loss'] += acc['train_vq_loss']
    acc['total_ntp_loss'] += acc['train_ntp_loss']
    acc['total_tik_tok_padding_loss'] += acc['train_tik_tok_padding_loss']

    acc['total_unscaled_step_loss'] += acc['train_unscaled_step_loss']
    acc['total_unscaled_rec_loss'] += acc['train_unscaled_rec_loss']
    acc['total_unscaled_final_fape_loss'] += acc['train_unscaled_final_fape_loss']
    acc['total_unscaled_aux_fape_loss'] += acc['train_unscaled_aux_fape_loss']
    acc['total_unscaled_vq_loss'] += acc['train_unscaled_vq_loss']
    acc['total_unscaled_ntp_loss'] += acc['train_unscaled_ntp_loss']
    acc['total_unscaled_tik_tok_padding_loss'] += acc['train_unscaled_tik_tok_padding_loss']

    acc['train_step_loss'] = 0.0
    acc['train_rec_loss'] = 0.0
    acc['train_final_fape_loss'] = 0.0
    acc['train_aux_fape_loss'] = 0.0
    acc['train_vq_loss'] = 0.0
    acc['train_ntp_loss'] = 0.0
    acc['train_tik_tok_padding_loss'] = 0.0

    acc['train_unscaled_step_loss'] = 0.0
    acc['train_unscaled_rec_loss'] = 0.0
    acc['train_unscaled_final_fape_loss'] = 0.0
    acc['train_unscaled_aux_fape_loss'] = 0.0
    acc['train_unscaled_vq_loss'] = 0.0
    acc['train_unscaled_ntp_loss'] = 0.0
    acc['train_unscaled_tik_tok_padding_loss'] = 0.0

    acc['counter'] += 1


def average_losses(acc: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-step averages from totals accumulated so far.

    Inputs:
    - acc: Accumulator dict from init_accumulator.

    Returns:
    - Dict[str, float]: {'avg_step_loss', 'avg_rec_loss', 'avg_vq_loss', 'avg_ntp_loss'}
      averaged over acc['counter'] finalized optimizer steps.
    """
    denom = max(1, acc['counter'])
    return {
        'avg_step_loss': acc['total_step_loss'] / denom,
        'avg_rec_loss': acc['total_rec_loss'] / denom,
        'avg_final_fape_loss': acc['total_final_fape_loss'] / denom,
        'avg_aux_fape_loss': acc['total_aux_fape_loss'] / denom,
        'avg_vq_loss': acc['total_vq_loss'] / denom,
        'avg_ntp_loss': acc['total_ntp_loss'] / denom,
        'avg_tik_tok_padding_loss': acc['total_tik_tok_padding_loss'] / denom,
        'avg_unscaled_step_loss': acc['total_unscaled_step_loss'] / denom,
        'avg_unscaled_rec_loss': acc['total_unscaled_rec_loss'] / denom,
        'avg_unscaled_final_fape_loss': acc['total_unscaled_final_fape_loss'] / denom,
        'avg_unscaled_aux_fape_loss': acc['total_unscaled_aux_fape_loss'] / denom,
        'avg_unscaled_vq_loss': acc['total_unscaled_vq_loss'] / denom,
        'avg_unscaled_ntp_loss': acc['total_unscaled_ntp_loss'] / denom,
        'avg_unscaled_tik_tok_padding_loss': acc['total_unscaled_tik_tok_padding_loss'] / denom,
    }


def update_unique_indices(acc: Dict[str, Any], indices: torch.Tensor, accelerator) -> None:
    """Update the set of unique codebook indices seen so far.

    Inputs:
    - acc: Accumulator dict from init_accumulator (mutated in-place).
    - indices: Tensor of shape (B, L) containing codebook indices for tokens.
    - accelerator: accelerate.Accelerator for distributed gather.
    """
    gathered_indices = accelerator.gather(indices)
    acc['unique_indices'].update(gathered_indices.unique().cpu().tolist())


def compute_activation(acc: Dict[str, Any], codebook_size: int) -> float:
    """Compute codebook activation ratio based on unique indices.

    Inputs:
    - acc: Accumulator dict from init_accumulator.
    - codebook_size: Total number of entries in the codebook.

    Returns:
    - float in [0, 1]: fraction of codebook entries used so far.
    """
    if codebook_size <= 0:
        return 0.0
    return len(acc['unique_indices']) / float(codebook_size)


def progress_postfix(optimizer, loss_dict: Dict[str, torch.Tensor], global_step: int) -> Dict[str, Any]:
    """Build a compact dict for tqdm postfix with current train stats.

    Inputs:
    - optimizer: torch.optim.Optimizer for current learning rate.
    - loss_dict: Dict with current 'step_loss', 'rec_loss', and optionally 'vq_loss'.
    - global_step: Global optimizer step counter (int).

    Returns:
    - Dict[str, Any] with keys {'lr', 'step_loss', 'rec_loss', 'vq_loss', 'global_step'}
      for logging in progress bars.
    """
    return {
        'lr': optimizer.param_groups[0]['lr'],
        'step_loss': float(loss_dict['step_loss'].detach().item()),
        'rec_loss': float(loss_dict['rec_loss'].detach().item()),
        'vq_loss': float(loss_dict['vq_loss'].detach().item()) if 'vq_loss' in loss_dict else float('nan'),
        'vq_raw': float(loss_dict['unscaled_vq_loss'].detach().item()) if 'unscaled_vq_loss' in loss_dict else float('nan'),
        'global_step': int(global_step),
    }


def log_tensorboard_epoch(writer,
                          avgs: Dict[str, float],
                          metrics_values: Dict[str, float],
                          epoch: int,
                          activation_percent: float,
                          include_ntp: bool = False) -> None:
    """Log epoch-level losses and metrics to TensorBoard.

    Args:
        writer: TensorBoard writer (or ``None`` to skip logging).
        avgs: Output of :func:`average_losses`; may include TikTok padding loss.
        metrics_values: Output of :func:`compute_metrics`; may include perplexity
            and TikTok padding accuracy.
        epoch: Epoch index used as the TensorBoard step.
        activation_percent: Codebook activation ratio (%).
        include_ntp: Whether to log the NTP loss scalars.

    Notes:
        Logs the standard loss/metric suite; optional scalars (NTP loss,
        TikTok padding loss/accuracy) are emitted only when present in ``avgs`` /
        ``metrics_values``.
    """
    if writer is None:
        return

    writer.add_scalar('loss/total', avgs['avg_step_loss'], epoch)
    writer.add_scalar('loss/rec_loss', avgs['avg_rec_loss'], epoch)
    writer.add_scalar('loss/final_fape', avgs['avg_final_fape_loss'], epoch)
    writer.add_scalar('loss/aux_fape', avgs['avg_aux_fape_loss'], epoch)
    writer.add_scalar('loss/vq', avgs['avg_vq_loss'], epoch)
    if 'avg_tik_tok_padding_loss' in avgs:
        writer.add_scalar('loss/tik_tok_padding', avgs['avg_tik_tok_padding_loss'], epoch)
    if include_ntp:
        writer.add_scalar('loss/ntp', avgs['avg_ntp_loss'], epoch)

    # Unscaled epoch logs (if present)
    if 'avg_unscaled_step_loss' in avgs:
        writer.add_scalar('unscaled_loss/total', avgs['avg_unscaled_step_loss'], epoch)
    if 'avg_unscaled_rec_loss' in avgs:
        writer.add_scalar('unscaled_loss/rec_loss', avgs['avg_unscaled_rec_loss'], epoch)
    if 'avg_unscaled_final_fape_loss' in avgs:
        writer.add_scalar('unscaled_loss/final_fape', avgs['avg_unscaled_final_fape_loss'], epoch)
    if 'avg_unscaled_aux_fape_loss' in avgs:
        writer.add_scalar('unscaled_loss/aux_fape', avgs['avg_unscaled_aux_fape_loss'], epoch)
    if 'avg_unscaled_vq_loss' in avgs:
        writer.add_scalar('unscaled_loss/vq', avgs['avg_unscaled_vq_loss'], epoch)
    if 'avg_unscaled_tik_tok_padding_loss' in avgs:
        writer.add_scalar('unscaled_loss/tik_tok_padding', avgs['avg_unscaled_tik_tok_padding_loss'], epoch)
    if include_ntp and 'avg_unscaled_ntp_loss' in avgs:
        writer.add_scalar('unscaled_loss/ntp', avgs['avg_unscaled_ntp_loss'], epoch)

    tik_tok_acc = metrics_values.get('tik_tok_padding_accuracy', float('nan'))
    if tik_tok_acc == tik_tok_acc:
        writer.add_scalar('metric/padding_accuracy', tik_tok_acc, epoch)

    writer.add_scalar('codebook_activation', activation_percent, epoch)

    perplexity = metrics_values.get('perplexity', float('nan'))
    if perplexity == perplexity:  # check not NaN
        writer.add_scalar('metric/perplexity', perplexity, epoch)
