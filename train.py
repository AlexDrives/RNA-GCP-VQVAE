import argparse
import numpy as np
import yaml
import os
import sys
import torch
import math
from utils.custom_losses import calculate_decoder_loss, log_per_loss_grad_norms
from utils.utils import (
    save_backbone_pdb,
    load_configs,
    load_checkpoints,
    prepare_saving_dir,
    get_logging,
    prepare_optimizer,
    prepare_tensorboard,
    save_checkpoint,
    load_encoder_decoder_configs,
    get_decoder_config_file_path,
)
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from tqdm import tqdm
import time
from data.dataset import prepare_gcpnet_vqvae_dataloaders, load_rna_h5_file
from models.super_model import (
    prepare_model,
    compile_non_gcp_and_exclude_vq,
    compile_gcp_encoder,
)
from utils.training_helpers import (
    init_metrics,
    reset_metrics,
    update_metrics,
    compute_metrics,
    init_accumulator,
    accumulate_losses,
    finalize_step,
    average_losses,
    update_unique_indices,
    compute_activation,
    progress_postfix,
    log_tensorboard_epoch,
)


def _resolve_data_modality(configs) -> str:
    modality = (
        getattr(configs.train_settings, "data_modality", None)
        or getattr(configs.train_settings, "modality", None)
        or getattr(configs, "data_modality", None)
        or "protein"
    )
    return str(modality).lower()


def _backbone_atom_names(configs):
    if _resolve_data_modality(configs) == "rna":
        return ("C4'", "C1'", "N1/N9")
    return ("N", "CA", "C")


def _new_rigid_enabled(configs) -> bool:
    nested = getattr(configs.train_settings, "rna_rigid_template", None)
    if nested is not None and hasattr(nested, "get"):
        return bool(nested.get("new_rigid", getattr(configs.train_settings, "new_rigid", False)))
    return bool(getattr(configs.train_settings, "new_rigid", False))


def _new_rigid_max_samples(configs) -> int:
    nested = getattr(configs.train_settings, "rna_rigid_template", None)
    if nested is not None and hasattr(nested, "get"):
        value = nested.get("max_samples", getattr(configs.train_settings, "new_rigid_max_samples", 100))
    else:
        value = getattr(configs.train_settings, "new_rigid_max_samples", 100)
    return max(int(value), 1)


def _progress_update_every(configs) -> int:
    value = getattr(configs.train_settings, "progress_update_every", 20)
    return max(int(value), 1)


def _use_tqdm_progress(configs) -> bool:
    return bool(configs.tqdm_progress_bar and sys.stderr.isatty())


def _inject_rna_template_into_decoder_configs(decoder_configs, template_coords, stats):
    decoder_configs.rna_rigid_template_coords = [[float(v) for v in row] for row in template_coords.tolist()]
    decoder_configs.rna_rigid_template_stats = {
        "c4_c1_mean": float(stats["c4_c1_mean"]),
        "c1_n_mean": float(stats["c1_n_mean"]),
        "c4_n_mean": float(stats["c4_n_mean"]),
        "used_files": int(stats["used_files"]),
        "used_residues": int(stats["used_residues"]),
    }
    decoder_configs.rna_rigid_template_source = "estimated_from_training_data"


def _save_decoder_config(decoder_configs, result_path, decoder_name):
    decoder_cfg_path = os.path.join(
        result_path,
        os.path.basename(get_decoder_config_file_path(decoder_name)),
    )
    with open(decoder_cfg_path, "w") as f:
        yaml.safe_dump(decoder_configs.to_dict(), f, sort_keys=False)


def _estimate_rna_template_from_paths(sample_paths):
    c4_c1_list = []
    c1_n_list = []
    c4_n_list = []
    used_files = 0
    used_residues = 0

    for sample_path in sample_paths:
        try:
            _, coords, _ = load_rna_h5_file(sample_path)
        except Exception:
            continue

        coords_t = torch.as_tensor(coords, dtype=torch.float32)
        if coords_t.ndim != 3 or coords_t.shape[1:] != (3, 3):
            continue

        valid = torch.isfinite(coords_t).all(dim=-1).all(dim=-1)
        if not valid.any():
            continue

        residue_coords = coords_t[valid]
        c4 = residue_coords[:, 0]
        c1 = residue_coords[:, 1]
        n_atom = residue_coords[:, 2]

        c4_c1_list.append(torch.linalg.norm(c4 - c1, dim=-1))
        c1_n_list.append(torch.linalg.norm(c1 - n_atom, dim=-1))
        c4_n_list.append(torch.linalg.norm(c4 - n_atom, dim=-1))

        used_files += 1
        used_residues += int(valid.sum().item())

    if not c4_c1_list:
        raise RuntimeError("No valid RNA residues found to estimate rigid template.")

    c4_c1_mean = torch.cat(c4_c1_list).mean().item()
    c1_n_mean = torch.cat(c1_n_list).mean().item()
    c4_n_mean = torch.cat(c4_n_list).mean().item()

    if c1_n_mean <= 1e-8:
        raise RuntimeError("Estimated C1'-N mean is too small to construct a template.")

    x_coord = (c4_c1_mean ** 2 + c1_n_mean ** 2 - c4_n_mean ** 2) / (2.0 * c1_n_mean)
    y_sq = max(c4_c1_mean ** 2 - x_coord ** 2, 0.0)
    y_coord = math.sqrt(y_sq)

    template = torch.tensor(
        [
            [x_coord, y_coord, 0.0],
            [0.0, 0.0, 0.0],
            [c1_n_mean, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    stats = {
        "c4_c1_mean": c4_c1_mean,
        "c1_n_mean": c1_n_mean,
        "c4_n_mean": c4_n_mean,
        "used_files": used_files,
        "used_residues": used_residues,
    }
    return template, stats


def _set_runtime_rna_template(net, template_coords):
    decoder = getattr(getattr(net, "vqvae", None), "decoder", None)
    if decoder is None:
        return False

    if hasattr(decoder, "template_coords"):
        buffer = decoder.template_coords
    elif hasattr(decoder, "affine_output_projection") and hasattr(decoder.affine_output_projection, "template_coords"):
        buffer = decoder.affine_output_projection.template_coords
    else:
        return False

    buffer.copy_(template_coords.to(device=buffer.device, dtype=buffer.dtype))
    return True


def train_loop(net, train_loader, epoch, adaptive_loss_coeffs, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    configs = kwargs.pop('configs')
    writer = kwargs.pop('writer')
    logging = kwargs.pop('logging')
    profiler = kwargs.pop('profiler')
    profile_train_loop = kwargs.pop('profile_train_loop')
    atom_names = _backbone_atom_names(configs)
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation
    alignment_strategy = configs.train_settings.losses.alignment_strategy
    progress_update_every = _progress_update_every(configs)
    use_tqdm = _use_tqdm_progress(configs) and accelerator.is_main_process

    # Initialize optional auxiliary metrics and accumulators
    metrics = init_metrics(configs, accelerator)
    acc = init_accumulator(accum_iter)

    optimizer.zero_grad()

    global_step = kwargs.get('global_step', 0)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / accum_iter))),
                        leave=False, disable=not use_tqdm)
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(net):
            if profile_train_loop:
                profiler.step()
                if i >= 1000:  # Profile only the first 1000 steps
                    logging.info("Profiler finished, exiting train step loop.")
                    break

            masks = torch.logical_and(data['masks'], data['nan_masks'])

            output_dict = net(data)

            # Compute the loss components (function unwraps tensors internally)
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                output_dict=output_dict,
                data=data,
                configs=configs,
                alignment_strategy=alignment_strategy,
                adaptive_loss_coeffs=adaptive_loss_coeffs,
            )

            # Apply sample weights to loss if enabled
            if configs.train_settings.sample_weighting.enabled:
                sample_weights = data['sample_weights']
                # Use the mean sample weight for the batch (could be weighted by batch size)
                batch_weight = sample_weights.mean()
                for key in (
                    'final_fape_loss',
                    'aux_fape_loss',
                    'mse_loss',
                    'backbone_distance_loss',
                    'backbone_direction_loss',
                    'binned_direction_classification_loss',
                    'binned_distance_classification_loss',
                ):
                    if key in loss_dict:
                        loss_dict[key] = loss_dict[key] * batch_weight
                loss_dict['rec_loss'] = loss_dict['rec_loss'] * batch_weight
                loss_dict['unscaled_rec_loss'] = loss_dict['unscaled_rec_loss'] * batch_weight
                loss_dict['step_loss'] = loss_dict['rec_loss'] + loss_dict['vq_loss'] + loss_dict['ntp_loss']
                loss_dict['unscaled_step_loss'] = (
                    loss_dict['unscaled_rec_loss']
                    + loss_dict['unscaled_vq_loss']
                    + loss_dict['unscaled_ntp_loss']
                )

            # Log per-loss gradient norms and adjust adaptive coefficients
            adaptive_loss_coeffs = log_per_loss_grad_norms(
                loss_dict, net, configs, writer, accelerator,
                global_step, adaptive_loss_coeffs
            )


            if accelerator.is_main_process and epoch % configs.train_settings.save_pdb_every == 0 and epoch != 0 and i == 0:
                logging.info(f"Building PDB files for training data in epoch {epoch}")
                save_backbone_pdb(trans_pred_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files',
                                               f'train_outputs_epoch_{epoch}_step_{i + 1}'),
                                  atom_names=atom_names,
                                  residue_sequences=data['seq'])
                save_backbone_pdb(trans_true_coords.detach().squeeze(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files', f'train_labels_step_{i + 1}'),
                                  atom_names=atom_names,
                                  residue_sequences=data['seq'])
                logging.info("PDB files are built")

            # Update optional metrics and accumulators
            update_metrics(metrics, trans_pred_coords, trans_true_coords, masks, output_dict, ignore_index=-100)
            accumulate_losses(acc, loss_dict, output_dict, configs, accelerator, use_output_vq=False)
            update_unique_indices(acc, output_dict["indices"], accelerator)

            accelerator.backward(loss_dict['step_loss'])
            if accelerator.sync_gradients:
                if global_step % configs.train_settings.gradient_norm_logging_freq == 0 and global_step > 0:
                    # Calculate the gradient norm every configs.train_settings.gradient_norm_logging_freq steps
                    grad_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in net.parameters() if p.grad is not None and p.requires_grad]),
                        2)
                    if accelerator.is_main_process and configs.tensorboard_log:
                        writer.add_scalar('gradient norm/total_amp_scaled', grad_norm.item(), global_step)

                # Accelerate Gradient clipping: unscale the gradients (only when using FP16 AMP) and then apply clipping
                accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                finalize_step(acc)

                if accelerator.is_main_process and configs.tensorboard_log:
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

                if global_step % progress_update_every == 0 or global_step == 1:
                    avgs = average_losses(acc)
                    if use_tqdm:
                        progress_bar.set_description(f"epoch {epoch} "
                                                     + f"[loss: {avgs['avg_unscaled_step_loss']:.3f}, "
                                                     + f"rec loss: {avgs['avg_unscaled_rec_loss']:.3f}, "
                                                     + f"fape: {avgs['avg_unscaled_final_fape_loss']:.3f}, "
                                                     + f"aux: {avgs['avg_unscaled_aux_fape_loss']:.3f}, "
                                                     + f"vq raw: {avgs['avg_unscaled_vq_loss']:.3f}, "
                                                     + f"vq scaled: {avgs['avg_vq_loss']:.3f}]")
                        progress_bar.set_postfix(
                            progress_postfix(optimizer, loss_dict, global_step)
                        )
                    elif accelerator.is_main_process:
                        logging.info(
                            "epoch %d step %d/%d - loss %.3f rec %.3f final_fape %.3f aux_fape %.3f vq_raw %.3f vq_scaled %.3f lr %.2e",
                            epoch,
                            global_step,
                            int(np.ceil(len(train_loader) / accum_iter)),
                            avgs['avg_unscaled_step_loss'],
                            avgs['avg_unscaled_rec_loss'],
                            avgs['avg_unscaled_final_fape_loss'],
                            avgs['avg_unscaled_aux_fape_loss'],
                            avgs['avg_unscaled_vq_loss'],
                            avgs['avg_vq_loss'],
                            optimizer.param_groups[0]['lr'],
                        )

    # Compute average losses and metrics
    avgs = average_losses(acc)
    metrics_values = compute_metrics(metrics)
    avg_activation = compute_activation(acc, codebook_size)

    # Log metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        include_ntp = getattr(configs.train_settings.losses, 'next_token_prediction', None) and \
                      configs.train_settings.losses.next_token_prediction.enabled
        log_tensorboard_epoch(
            writer,
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
        )

    # Reset the metrics for the next epoch
    reset_metrics(metrics)

    return_dict = {
        "loss": avgs['avg_unscaled_step_loss'],
        "rec_loss": avgs['avg_unscaled_rec_loss'],
        "final_fape_loss": avgs['avg_unscaled_final_fape_loss'],
        "aux_fape_loss": avgs['avg_unscaled_aux_fape_loss'],
        "ntp_loss": avgs['avg_unscaled_ntp_loss'],
        "vq_loss": avgs['avg_unscaled_vq_loss'],
        "perplexity": metrics_values['perplexity'],
        "padding_accuracy": metrics_values.get('tik_tok_padding_accuracy', float('nan')),
        "activation": np.round(avg_activation * 100, 1),
        "counter": acc['counter'],
        "global_step": global_step,
        "adaptive_loss_coeffs": adaptive_loss_coeffs
    }
    return return_dict


def valid_loop(net, valid_loader, epoch, **kwargs):
    optimizer = kwargs.pop('optimizer')
    configs = kwargs.pop('configs')
    accelerator = kwargs.pop('accelerator')
    writer = kwargs.pop('writer')
    logging = kwargs.pop('logging')
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    alignment_strategy = configs.train_settings.losses.alignment_strategy
    atom_names = _backbone_atom_names(configs)
    progress_update_every = _progress_update_every(configs)
    use_tqdm = _use_tqdm_progress(configs) and accelerator.is_main_process

    # Initialize optional metrics and accumulators for validation
    metrics = init_metrics(configs, accelerator)
    acc = init_accumulator(accum_iter=1)

    optimizer.zero_grad()

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(len(valid_loader))),
                        leave=False, disable=not use_tqdm)
    progress_bar.set_description(f"Validation epoch {epoch}")

    net.eval()
    for i, data in enumerate(valid_loader):
        with torch.inference_mode():
            masks = torch.logical_and(data['masks'], data['nan_masks'])

            output_dict = net(data)

            update_unique_indices(acc, output_dict["indices"], accelerator)

            # Compute the loss components using dict-style outputs like train loop
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                output_dict=output_dict,
                data=data,
                configs=configs,
                alignment_strategy=alignment_strategy,
                adaptive_loss_coeffs=None,
            )

            if accelerator.is_main_process and epoch % configs.valid_settings.save_pdb_every == 0 and epoch != 0 and i == 0:
                logging.info(f"Building PDB files for validation data in epoch {epoch}")
                save_backbone_pdb(trans_pred_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files',
                                               f'valid_outputs_epoch_{epoch}_step_{i + 1}'),
                                  atom_names=atom_names,
                                  residue_sequences=data['seq'])
                save_backbone_pdb(trans_true_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files', f'valid_labels_step_{i + 1}'),
                                  atom_names=atom_names,
                                  residue_sequences=data['seq'])
                logging.info("PDB files are built")

            # Update optional metrics and losses
            update_metrics(metrics, trans_pred_coords, trans_true_coords, masks, output_dict, ignore_index=-100)
            accumulate_losses(acc, loss_dict, output_dict, configs, accelerator, use_output_vq=False)
            # Finalize this validation step so totals/averages are updated
            finalize_step(acc)

        progress_bar.update(1)
        if (i + 1) % progress_update_every == 0 or (i + 1) == len(valid_loader):
            avgs = average_losses(acc)
            if use_tqdm:
                progress_bar.set_description(f"validation epoch {epoch} "
                                             + f"[loss: {avgs['avg_unscaled_step_loss']:.3f}, "
                                             + f"rec loss: {avgs['avg_unscaled_rec_loss']:.3f}, "
                                             + f"fape: {avgs['avg_unscaled_final_fape_loss']:.3f}, "
                                             + f"aux: {avgs['avg_unscaled_aux_fape_loss']:.3f}, "
                                             + f"vq raw: {avgs['avg_unscaled_vq_loss']:.3f}]")
            elif accelerator.is_main_process:
                logging.info(
                    "validation epoch %d step %d/%d - loss %.3f rec %.3f final_fape %.3f aux_fape %.3f vq_raw %.3f",
                    epoch,
                    i + 1,
                    len(valid_loader),
                    avgs['avg_unscaled_step_loss'],
                    avgs['avg_unscaled_rec_loss'],
                    avgs['avg_unscaled_final_fape_loss'],
                    avgs['avg_unscaled_aux_fape_loss'],
                    avgs['avg_unscaled_vq_loss'],
                )

    # Compute averages and metrics
    avgs = average_losses(acc)
    avg_activation = compute_activation(acc, codebook_size)
    metrics_values = compute_metrics(metrics)

    # Log metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        include_ntp = getattr(configs.train_settings.losses, 'next_token_prediction', None) and \
                      configs.train_settings.losses.next_token_prediction.enabled
        log_tensorboard_epoch(
            writer,
            avgs,
            metrics_values,
            epoch,
            activation_percent=np.round(avg_activation * 100, 1),
            include_ntp=include_ntp,
        )

    # Reset metrics for the next epoch
    reset_metrics(metrics)

    return_dict = {
        "loss": avgs['avg_unscaled_step_loss'],
        "rec_loss": avgs['avg_unscaled_rec_loss'],
        "final_fape_loss": avgs['avg_unscaled_final_fape_loss'],
        "aux_fape_loss": avgs['avg_unscaled_aux_fape_loss'],
        "vq_loss": avgs['avg_unscaled_vq_loss'],
        "ntp_loss": avgs['avg_unscaled_ntp_loss'],
        "perplexity": metrics_values['perplexity'],
        "padding_accuracy": metrics_values.get('tik_tok_padding_accuracy', float('nan')),
        "activation": np.round(avg_activation * 100, 1),
        "counter": acc['counter'],
    }
    return return_dict


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)
    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    # Set find_unused_parameters to True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=configs.find_unused_parameters)
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=configs.dispatch_batches,
        even_batches=configs.even_batches,
        non_blocking=configs.non_blocking,
        split_batches=configs.split_batches,
        # use_stateful_dataloader=True
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, InitProcessGroupKwargs(timeout=timedelta(minutes=20))],
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dataloader_config=dataloader_config
    )

    # Initialize paths to avoid unassigned variable warnings
    result_path, checkpoint_path = None, None

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        paths = [result_path, checkpoint_path]
    else:
        # Initialize with placeholders.
        paths = [None, None]

    if accelerator.num_processes > 1:
        import torch.distributed as dist
        # Broadcast the list of strings from the main process (src=0) to all others.
        dist.broadcast_object_list(paths, src=0)

        # Now every process has the shared values.
        result_path, checkpoint_path = paths

    encoder_configs, decoder_configs = load_encoder_decoder_configs(configs, result_path)

    logging = get_logging(result_path, configs)

    train_dataloader, valid_dataloader = prepare_gcpnet_vqvae_dataloaders(
        logging, accelerator, configs, encoder_configs=encoder_configs, decoder_configs=decoder_configs
    )
    logging.info('preparing dataloaders are done')

    modality = _resolve_data_modality(configs)
    runtime_rna_template = None
    if modality == "rna" and _new_rigid_enabled(configs):
        template_payload = [None]
        if accelerator.is_main_process:
            try:
                train_dataset = train_dataloader.dataset
                sample_paths = list(getattr(train_dataset, "h5_samples", []))
                max_samples = _new_rigid_max_samples(configs)
                sample_count = min(max_samples, len(sample_paths))
                if sample_count == 0:
                    raise RuntimeError("RNA training dataset is empty.")

                template_coords, template_stats = _estimate_rna_template_from_paths(sample_paths[:sample_count])
                template_payload[0] = {
                    "coords": template_coords.tolist(),
                    "stats": template_stats,
                    "sample_count": sample_count,
                }
            except Exception as exc:
                logging.warning(f"Failed to estimate new RNA rigid template; fallback to configured template. Error: {exc}")

        if accelerator.num_processes > 1:
            import torch.distributed as dist
            dist.broadcast_object_list(template_payload, src=0)

        if template_payload[0] is not None:
            runtime_rna_template = torch.tensor(template_payload[0]["coords"], dtype=torch.float32)
            _inject_rna_template_into_decoder_configs(
                decoder_configs,
                runtime_rna_template,
                template_payload[0]["stats"],
            )

            if accelerator.is_main_process:
                _save_decoder_config(decoder_configs, result_path, configs.model.vqvae.decoder.name)
                logging.info(
                    "RNA rigid template refreshed from first %d train samples: "
                    "C4'-C1'=%.6f, C1'-N=%.6f, C4'-N=%.6f (files=%d, residues=%d)",
                    template_payload[0]["sample_count"],
                    template_payload[0]["stats"]["c4_c1_mean"],
                    template_payload[0]["stats"]["c1_n_mean"],
                    template_payload[0]["stats"]["c4_n_mean"],
                    template_payload[0]["stats"]["used_files"],
                    template_payload[0]["stats"]["used_residues"],
                )
        else:
            logging.info("RNA rigid template kept as configured (no new template generated).")

    net = prepare_model(
        configs, logging,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    logging.info('preparing models is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)
    if runtime_rna_template is not None:
        if _set_runtime_rna_template(net, runtime_rna_template):
            logging.info("Applied refreshed RNA rigid template to decoder runtime buffer.")
        else:
            logging.warning("Could not locate decoder template buffer to apply refreshed RNA rigid template.")

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        if hasattr(net, 'encoder') and configs.model.encoder.name == "gcpnet":
            net = compile_gcp_encoder(net, mode=None, backend="inductor")
            logging.info('GCP encoder compiled.')
        net = compile_non_gcp_and_exclude_vq(net, mode=None, backend="inductor")
        logging.info('All GCP-VQVAE layers compiled except VQ layer.')
    net, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, valid_dataloader, scheduler
    )

    net.to(accelerator.device)

    if accelerator.is_main_process:
        # initialize tensorboards
        train_writer, valid_writer = prepare_tensorboard(result_path)
    else:
        train_writer, valid_writer = None, None

    if accelerator.is_main_process:
        train_steps = np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
        logging.info(f'number of train steps per epoch: {int(train_steps)}')

    # Maybe monitor resource usage during training.
    prof = None
    profile_train_loop = configs.train_settings.profile_train_loop

    if profile_train_loop:
        from pathlib import Path
        train_profile_path = os.path.join(result_path, 'train', 'profile')
        Path(train_profile_path).mkdir(parents=True, exist_ok=True)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=30, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(train_profile_path),
            profile_memory=True,
        )
        prof.start()

    # Use this to keep track of the global step across all processes.
    # This is useful for continuing training from a checkpoint.
    global_step = 0

    # Initialize adaptive loss coefficients (persistent across epochs)
    adaptive_loss_coeffs = {
        'mse': 1.0,
        'backbone_distance': 1.0,
        'backbone_direction': 1.0,
        'binned_direction_classification': 1.0,
        'binned_distance_classification': 1.0,
        'vq': 1.0,
        'ntp': 1.0,
        'tik_tok_padding': 1.0,
    }

    best_valid_metrics = {
        'loss': float('inf'),
        'rec_loss': float('inf'),
        'final_fape_loss': float('inf'),
        'aux_fape_loss': float('inf'),
        'vq_loss': float('inf'),
        'perplexity': 1000.0,
        'padding_accuracy': 0.0,
    }
    for epoch in range(1, configs.train_settings.num_epochs + 1):
        start_time = time.time()
        training_loop_reports = train_loop(net, train_dataloader, epoch, adaptive_loss_coeffs,
                                           accelerator=accelerator,
                                           optimizer=optimizer,
                                           scheduler=scheduler, configs=configs,
                                           logging=logging, global_step=global_step,
                                           writer=train_writer, result_path=result_path,
                                           profiler=prof, profile_train_loop=profile_train_loop)

        if profile_train_loop:
            prof.stop()
            logging.info("Profiler stopped, exiting train epoch loop.")
            break

        end_time = time.time()
        training_time = end_time - start_time
        logging.info(
            f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
            f'global steps {training_loop_reports["global_step"]}, loss {training_loop_reports["loss"]:.4f}, '
            f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
            f'final fape {training_loop_reports["final_fape_loss"]:.4f}, '
            f'aux fape {training_loop_reports["aux_fape_loss"]:.4f}, '
            f'ntp loss {training_loop_reports["ntp_loss"]:.4f}, '
            f'perplexity {training_loop_reports.get("perplexity", float("nan")):.2f}, '
            f'padding acc {training_loop_reports.get("padding_accuracy", float("nan")):.4f}, '
            f'vq loss {training_loop_reports["vq_loss"]:.4f}, '
            f'activation {training_loop_reports["activation"]:.1f}')

        global_step = training_loop_reports["global_step"]
        # Update adaptive coefficients from training loop
        adaptive_loss_coeffs = training_loop_reports.get("adaptive_loss_coeffs", adaptive_loss_coeffs)
        accelerator.wait_for_everyone()

        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # Set the path to save the models checkpoint.
                model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler,
                                configs=configs)
                logging.info(f'\tcheckpoint models in {model_path}')

        if epoch % configs.valid_settings.do_every == 0:
            start_time = time.time()
            valid_loop_reports = valid_loop(net, valid_dataloader, epoch,
                                            accelerator=accelerator,
                                            optimizer=optimizer,
                                            scheduler=scheduler, configs=configs,
                                            logging=logging, global_step=global_step,
                                            writer=valid_writer, result_path=result_path)
            end_time = time.time()
            valid_time = end_time - start_time
            accelerator.wait_for_everyone()
            logging.info(
                f'validation epoch {epoch} ({valid_loop_reports["counter"]} steps) - time {np.round(valid_time, 2)}s, '
                f'loss {valid_loop_reports["loss"]:.4f}, '
                f'rec loss {valid_loop_reports["rec_loss"]:.4f}, '
                f'final fape {valid_loop_reports["final_fape_loss"]:.4f}, '
                f'aux fape {valid_loop_reports["aux_fape_loss"]:.4f}, '
                f'ntp loss {valid_loop_reports["ntp_loss"]:.4f}, '
                f'perplexity {valid_loop_reports.get("perplexity", float("nan")):.2f}, '
                f'padding acc {valid_loop_reports.get("padding_accuracy", float("nan")):.4f}, '
                f'vq loss {valid_loop_reports["vq_loss"]:.4f}, '
                f'activation {valid_loop_reports["activation"]:.1f}'
                # f'lddt {valid_loop_reports["lddt"]:.4f}'
            )

            # Save the best model based on the active validation objective.
            if valid_loop_reports["loss"] < best_valid_metrics['loss']:
                best_valid_metrics['loss'] = valid_loop_reports["loss"]
                best_valid_metrics['rec_loss'] = valid_loop_reports["rec_loss"]
                best_valid_metrics['final_fape_loss'] = valid_loop_reports["final_fape_loss"]
                best_valid_metrics['aux_fape_loss'] = valid_loop_reports["aux_fape_loss"]
                best_valid_metrics['vq_loss'] = valid_loop_reports["vq_loss"]
                best_valid_metrics['perplexity'] = valid_loop_reports.get("perplexity", float("nan"))
                best_valid_metrics['padding_accuracy'] = valid_loop_reports.get("padding_accuracy", float("nan"))

                tools = dict()
                tools['net'] = net
                tools['optimizer'] = optimizer
                tools['scheduler'] = scheduler

                accelerator.wait_for_everyone()

                # Set the path to save the model checkpoint.
                model_path = os.path.join(checkpoint_path, f'best_valid.pth')
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler,
                                configs=configs)
                logging.info(f'\tsaving the best models in {model_path}')
                logging.info(f'\tbest valid loss: {best_valid_metrics["loss"]:.4f}')

    logging.info("Training is completed!\n")

    if np.isfinite(best_valid_metrics['loss']):
        logging.info(f"best valid rec loss: {best_valid_metrics['rec_loss']:.4f}")
        logging.info(f"best valid final fape: {best_valid_metrics['final_fape_loss']:.4f}")
        logging.info(f"best valid aux fape: {best_valid_metrics['aux_fape_loss']:.4f}")
        logging.info(f"best valid vq loss: {best_valid_metrics['vq_loss']:.4f}")
        logging.info(f"best valid perplexity: {best_valid_metrics['perplexity']:.2f}")
        logging.info(f"best valid padding accuracy: {best_valid_metrics['padding_accuracy']:.4f}")
        logging.info(f"best valid loss: {best_valid_metrics['loss']:.4f}")
    else:
        logging.info("best valid metrics unavailable because validation did not run.")

    if accelerator.is_main_process:
        train_writer.close()
        valid_writer.close()

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()
    torch.cuda.empty_cache()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE models.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config_vqvae_dihedral.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)



