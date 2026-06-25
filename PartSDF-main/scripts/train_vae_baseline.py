"""
Train a PointNet-VAE-conditioned SDF model from surface points and SDF samples.
"""

import argparse
import json
import logging
import os
import os.path
import sys
import time
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import visualization as viz
from src import workspace as ws
from src.data import SurfaceSdfDataset, samples_from_array
from src.loss import get_loss_recon
from src.mesh import SdfGridFiller, create_mesh
from src.metric import chamfer_distance
from src.model import get_model
from src.optimizer import get_optimizer, get_scheduler
from src.utils import clamp_sdf, configure_logging, get_gradient, get_device, set_seed


def parser(argv=None):
    parser_ = argparse.ArgumentParser(
        description="Train a VAE-conditioned implicit neural representation model."
    )
    parser_.add_argument(
        "experiment",
        help="path to the experiment directory. If existing, will try to resume it (see --no-resume)",
    )
    parser_.add_argument(
        "--debug", action="store_true", help="increase verbosity to print debugging messages"
    )
    parser_.add_argument(
        "--load-epoch",
        type=int,
        default=None,
        help="specific epoch to resume from (will throw an error if not possible)",
    )
    parser_.add_argument(
        "--no-resume",
        action="store_true",
        help="do not resume the experiment if existing and start from epoch 0",
    )
    parser_.add_argument(
        "--no-test", action="store_true", help="do not perform final test reconstruction"
    )
    parser_.add_argument(
        "-q", "--quiet", dest="verbose", action="store_false", help="disable verbosity"
    )
    parser_.add_argument(
        "--reset", action="store_true", help="force a reset of the experimental directory"
    )
    parser_.add_argument("--seed", type=int, default=0, help="initial RNG seed")
    parser_.add_argument(
        "--workers",
        type=int,
        default=16,
        help="number of worker subprocesses preparing batches (use 0 to load in the main process)",
    )
    return parser_.parse_args(argv)


def append_condition_to_xyz(xyz, cond):
    if cond is None:
        return xyz
    return torch.cat([xyz, cond], dim=-1)


def get_default_query_condition(specs, npz=None, device=None):
    network_specs = specs.get("NetworkSpecs", {})
    decoder_specs = network_specs.get("decoder_specs", {})
    cond_dim = max(
        int(decoder_specs.get("input_dim", network_specs.get("input_dim", 3))) - 3,
        0,
    )
    if cond_dim <= 0:
        return None

    cond_value = specs.get("ConditionValue", None)
    if cond_value is None and npz is not None:
        cond_columns = []
        for key in ["pos", "neg"]:
            if key in npz and npz[key].shape[1] > 4:
                cond_columns.append(npz[key][:, 4:])
        if cond_columns:
            cond_all = np.concatenate(cond_columns, axis=0)
            cond_unique = np.unique(cond_all, axis=0)
            cond_value = cond_unique[len(cond_unique) // 2]

    if cond_value is None:
        cond_value = np.zeros(cond_dim, dtype=np.float32)

    cond = torch.as_tensor(cond_value, dtype=torch.float32, device=device)
    if cond.ndim == 0:
        cond = cond.view(1)
    return cond


def load_surface_points(filename, n_surface_samples):
    surface = np.load(filename).astype(np.float32)
    surface = np.random.permutation(surface)[:n_surface_samples, :3]
    return surface


@torch.no_grad()
def infer_latent_from_surface(model, surface_points, device, use_mean=True):
    surface_tensor = torch.from_numpy(surface_points).float().unsqueeze(0).to(device)
    return model.encode_surface(surface_tensor, sample=not use_mean)


@torch.no_grad()
def evaluate_instance(
    model,
    specs,
    instance,
    n_surface_samples,
    n_sdf_samples,
    loss_recon,
    clampD,
    use_mean_latent,
    device,
    grid_filler=None,
):
    sample_dir = os.path.join(specs["DataSource"], specs["SamplesDir"], instance)
    npz = np.load(os.path.join(sample_dir, specs["SamplesFile"]))
    surface_points = load_surface_points(
        os.path.join(sample_dir, specs.get("SurfaceSamplesFile", "surface.npy")),
        n_surface_samples,
    )
    latent = infer_latent_from_surface(model, surface_points, device, use_mean=use_mean_latent)

    sampled = samples_from_array(npz["pos"], npz["neg"], n_sdf_samples, balance=True)
    if len(sampled) == 3:
        xyz, sdf_gt, cond = sampled
        cond = torch.from_numpy(cond).float().unsqueeze(0).to(device)
    else:
        xyz, sdf_gt = sampled
        cond = None

    xyz = torch.from_numpy(xyz).float().unsqueeze(0).to(device)
    sdf_gt = torch.from_numpy(sdf_gt).float().unsqueeze(0).to(device)
    xyz_model = append_condition_to_xyz(xyz, cond)
    sdf_pred = model(latent.unsqueeze(1), xyz_model)

    if getattr(model, "use_occ", False):
        sdf_gt = (sdf_gt <= 0.0).float()
    elif clampD is not None and clampD > 0.0:
        sdf_pred = clamp_sdf(sdf_pred, clampD, ref=sdf_gt)
        sdf_gt = clamp_sdf(sdf_gt, clampD)

    err = float(loss_recon(sdf_pred, sdf_gt).mean().item())
    query_cond = get_default_query_condition(specs, npz=npz, device=device)
    mesh = create_mesh(
        model,
        latent,
        grid_filler=grid_filler,
        device=device,
        query_cond=query_cond,
    )
    return err, latent.detach().cpu(), mesh


def main(args=None):
    if args is None:
        args = parser()
    args.workers = min(args.workers, cpu_count())
    set_seed(args.seed)
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    device = specs.get("Device", get_device())
    if args.reset:
        ws.reset_experiment_dir(expdir)
    else:
        ws.build_experiment_dir(expdir)
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "trainlog.txt"))

    logging.info(f"Command: python {' '.join(sys.argv)}")
    logging.info(
        f"Date: {time.strftime('%d %B %Y at %H:%M:%S (%Y%m%d-%H%M%S)')}"
    )
    logging.info(f"Running experiment in {expdir}. (on {device})")
    logging.info(f"arguments = {args}")

    if specs.get("Network", "").lower() not in ["vaedeepsdf", "vae-deepsdf"]:
        raise ValueError(
            'This script expects `"Network": "VAEDeepSDF"` in the experiment specs.'
        )

    n_surface_samples = int(specs.get("SurfaceSamplesPerScene", 2048))
    dataset = SurfaceSdfDataset(
        specs["DataSource"],
        specs["TrainSplit"],
        specs["SamplesPerScene"],
        n_surface_samples,
        sampledir=specs["SamplesDir"],
        samplefile=specs["SamplesFile"],
        surface_samplefile=specs.get("SurfaceSamplesFile", "surface.npy"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=specs["ScenesPerBatch"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    len_dataset = len(dataset)

    valid_frequency = specs.get("ValidFrequency", None)
    valid_split = specs.get("ValidSplit", specs["TestSplit"])
    if valid_frequency is not None and valid_split is not None:
        with open(valid_split) as f:
            valid_split = json.load(f)
    else:
        valid_frequency, valid_split = None, None

    logging.info(f"{len_dataset} shapes in training dataset.")
    if valid_frequency is not None:
        logging.info(f"{len(valid_split)} shapes in validation dataset.")
    logging.info(f"{args.workers} worker processes created.")

    latent_dim = specs["LatentDim"]
    use_occ = specs.get("ImplicitField", "SDF").lower() in ["occ", "occupancy"]
    model = get_model(
        specs["Network"],
        **specs.get("NetworkSpecs", {}),
        latent_dim=latent_dim,
        use_occ=use_occ,
    ).to(device)

    model_pretrain = specs.get("NetworkPretrained", None)
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain))

    logging.info(
        f"Model has {sum([x.nelement() for x in model.parameters()]):,} parameters."
        + (" (pretrained)" if model_pretrain is not None else "")
    )

    loss_recon = get_loss_recon(specs["ReconLoss"], reduction="none").to(device)
    latent_reg = specs.get("LatentRegLambda", None)
    kl_lambda = float(specs.get("KLLossLambda", 1e-4))
    eikonal_lambda = specs.get("EikonalLossLambda", None)
    weight_norm_reg = specs.get("WeightNormRegLambda", None)
    use_mean_latent_eval = bool(specs.get("UseMeanLatentAtEval", True))

    optimizer_specs = dict(specs["Optimizer"])
    optimizer_type = optimizer_specs.pop("Type")
    optimizer_lrs = optimizer_specs.pop("LearningRates")
    if isinstance(optimizer_lrs, list):
        optimizer_lrs = optimizer_lrs[0]
    optimizer = get_optimizer(
        model,
        type=optimizer_type,
        lrs=optimizer_lrs,
        **optimizer_specs,
    )
    scheduler = get_scheduler(optimizer, **specs["LearningRateSchedule"])

    history = {"epoch": 0}
    if not args.no_resume:
        if args.load_epoch is None and os.path.isfile(ws.get_checkpoint_filename(expdir)):
            checkpoint = ws.load_checkpoint(expdir)
            ws.use_checkpoint(checkpoint, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, checkpoint["epoch"])
            del checkpoint
        elif args.load_epoch is not None and os.path.isfile(
            os.path.join(expdir, ws.HISTORY_FILE)
        ):
            ws.load_experiment(expdir, args.load_epoch, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, args.load_epoch)
        if history["epoch"] > 0:
            logging.info(f"Loaded checkpoint from epoch={history['epoch']}.")

    log_frequency = specs.get("LogFrequency", 10)
    snapshot_epochs = set(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    for cp in specs["AdditionalSnapshots"]:
        snapshot_epochs.add(cp)
    render_frequency = specs.get("RenderFrequency", None)

    n_epochs = specs["NumEpochs"]
    clampD = specs["ClampingDistance"]

    loss_names = ["loss", "loss_recon", "loss_kl"]
    if latent_reg is not None and latent_reg > 0.0:
        loss_names += ["loss_reg"]
    if valid_frequency is not None:
        loss_names += ["loss-val"]
    if eikonal_lambda is not None and eikonal_lambda > 0.0:
        loss_names += ["loss_eik"]
    if weight_norm_reg is not None and weight_norm_reg > 0.0:
        loss_names += ["loss_wnreg"]
    for key in loss_names + ["lr", "lat_norm"]:
        if key not in history:
            history[key] = []

    for epoch in range(history["epoch"] + 1, n_epochs + 1):
        time_epoch = time.time()
        running_losses = {name: 0.0 for name in loss_names if not name.endswith("val")}
        running_latent_norm = 0.0
        model.train()

        for batch in dataloader:
            if len(batch) == 5:
                _, surface_points, xyz, sdf_gt, cond = batch
                cond = cond.to(device)
            else:
                _, surface_points, xyz, sdf_gt = batch
                cond = None

            batch_size = xyz.shape[0]
            surface_points = surface_points.to(device).float()
            xyz = xyz.to(device).requires_grad_(
                eikonal_lambda is not None and eikonal_lambda > 0.0
            )
            sdf_gt = sdf_gt.to(device)
            xyz_model = append_condition_to_xyz(xyz, cond)

            latent, mu, logvar = model.encode_surface(
                surface_points, sample=True, return_dist=True
            )
            sdf_pred = model(latent.unsqueeze(1), xyz_model)
            sdf_pred_noclamp = sdf_pred

            if use_occ:
                sdf_gt = (sdf_gt <= 0.0).float()
            elif clampD is not None and clampD > 0.0:
                sdf_pred = clamp_sdf(sdf_pred, clampD, ref=sdf_gt)
                sdf_gt = clamp_sdf(sdf_gt, clampD)

            loss_recon_val = loss_recon(sdf_pred, sdf_gt).mean()
            loss_kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
            loss = loss_recon_val + kl_lambda * loss_kl

            running_losses["loss_recon"] += loss_recon_val.detach() * batch_size
            running_losses["loss_kl"] += loss_kl.detach() * batch_size

            if eikonal_lambda is not None and eikonal_lambda > 0.0:
                grads = get_gradient(xyz, sdf_pred_noclamp)
                loss_eikonal = (grads.norm(dim=-1) - 1.0).square().mean()
                loss = loss + eikonal_lambda * loss_eikonal
                running_losses["loss_eik"] += loss_eikonal.detach() * batch_size

            if weight_norm_reg is not None and weight_norm_reg > 0.0:
                loss_wnreg = 0.0
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear) and hasattr(module, "weight_g"):
                        loss_wnreg = loss_wnreg + module.weight_g.square().sum()
                loss = loss + weight_norm_reg * loss_wnreg
                running_losses["loss_wnreg"] += loss_wnreg.detach() * batch_size

            if latent_reg is not None and latent_reg > 0.0:
                loss_reg = latent.square().mean()
                loss = loss + latent_reg * loss_reg
                running_losses["loss_reg"] += loss_reg.detach() * batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_losses["loss"] += loss.detach() * batch_size
            running_latent_norm += latent.norm(dim=-1).mean().detach() * batch_size

        if valid_frequency is not None and epoch % valid_frequency == 0:
            valid_metrics = {"loss": [], "CD": []}
            valid_meshes = []
            grid_filler = SdfGridFiller(256, device=device) if not use_occ else None
            model.eval()

            for i, instance in enumerate(valid_split):
                err, _, mesh = evaluate_instance(
                    model,
                    specs,
                    instance,
                    n_surface_samples,
                    specs["SamplesPerScene"],
                    loss_recon,
                    clampD,
                    use_mean_latent_eval,
                    device,
                    grid_filler=grid_filler,
                )
                valid_metrics["loss"].append(err)
                if i < 8:
                    valid_meshes.append(mesh)
                if mesh.is_empty:
                    continue
                gt_samples = np.load(
                    os.path.join(
                        specs["DataSource"],
                        specs["SamplesDir"],
                        instance,
                        "surface.npy",
                    )
                )[:, :3]
                valid_metrics["CD"].append(
                    chamfer_distance(
                        np.random.permutation(gt_samples)[:30000, :3],
                        mesh.sample(30000),
                    )
                )
            for key in valid_metrics:
                valid_metrics[key] = (
                    np.mean(valid_metrics[key]) if len(valid_metrics[key]) > 0 else -1
                )
            del grid_filler
            optimizer.zero_grad()

        history["epoch"] += 1
        for name in running_losses:
            history[name].append(float(running_losses[name].item() / len_dataset))
        history["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        history["lat_norm"].append(float(running_latent_norm.item() / len_dataset))
        if valid_frequency is not None and epoch % valid_frequency == 0:
            history["loss-val"].append(valid_metrics["loss"])

        if scheduler is not None:
            scheduler.step()

        if render_frequency is not None and epoch % render_frequency == 0:
            render_count = min(len(dataset), 16)
            render_indices = list(range(min(8, render_count)))
            if render_count > 8:
                extra = (torch.randperm(render_count - 8)[: min(8, render_count - 8)] + 8).tolist()
                render_indices.extend(extra)

            render_meshes = []
            for idx in render_indices:
                item = dataset[idx]
                surface_points = item[1]
                sample_dir = os.path.join(
                    specs["DataSource"], specs["SamplesDir"], dataset.instances[idx]
                )
                npz = np.load(os.path.join(sample_dir, specs["SamplesFile"]))
                latent = infer_latent_from_surface(
                    model, surface_points, device, use_mean=use_mean_latent_eval
                )
                render_meshes.append(
                    create_mesh(
                        model,
                        latent,
                        grid_filler=True,
                        device=device,
                        query_cond=get_default_query_condition(specs, npz=npz, device=device),
                    )
                )

            renders = viz.render_meshes(render_meshes, size=224, aa_factor=2)
            ws.save_renders(expdir, renders, epoch)

            if valid_frequency is not None and epoch % valid_frequency == 0:
                renders = viz.render_meshes(valid_meshes, size=224, aa_factor=2)
                ws.save_renders(expdir, renders, f"valid_{epoch}")

        if epoch in snapshot_epochs:
            ws.save_experiment(expdir, epoch, model, None, optimizer, scheduler)
        if epoch % log_frequency == 0:
            ws.save_history(expdir, history)
            checkpoint = ws.build_checkpoint(epoch, model, None, optimizer, scheduler)
            ws.save_checkpoint(expdir, checkpoint)
            del checkpoint

        msg = f"Epoch {epoch}/{n_epochs}:"
        for name in running_losses:
            msg += f"{name}={history[name][-1]:.6f} - "
        if valid_frequency is not None and epoch % valid_frequency == 0:
            for key in valid_metrics:
                msg += f"{key}-val={valid_metrics[key]:.6f} - "
        msg = msg[:-3] + f" ({time.time() - time_epoch:.1f}s/epoch)"
        logging.info(msg)

    last_epoch = history["epoch"]
    checkpoint = ws.build_checkpoint(last_epoch, model, None, optimizer, scheduler)
    ws.save_checkpoint(expdir, checkpoint)
    ws.save_history(expdir, history)
    ws.save_experiment(expdir, last_epoch, model, None, optimizer, scheduler)
    torch.cuda.empty_cache()

    if not args.no_test:
        model.eval()
        test_surface_samples = int(specs.get("SurfaceSamplesPerScene", 2048))

        for sname, split in zip(["valid", "test"], [specs["ValidSplit"], specs["TestSplit"]]):
            if split is None:
                continue
            with open(split) as f:
                test_instances = json.load(f)
            idx = list(range(8))[: len(test_instances)] + (
                (torch.randperm(max(0, len(test_instances) - 8))[:8] + 8).tolist()
            )
            test_instances = [test_instances[i] for i in idx]

            time_test = time.time()
            meshes = []
            latents = []
            errors = []
            for instance in test_instances:
                err, latent, mesh = evaluate_instance(
                    model,
                    specs,
                    instance,
                    test_surface_samples,
                    specs["SamplesPerScene"],
                    loss_recon,
                    clampD,
                    use_mean_latent_eval,
                    device,
                    grid_filler=True,
                )
                errors.append(err)
                meshes.append(mesh)
                latents.append(latent)
            logging.info(
                f"{sname.capitalize()} reconstruction ({len(test_instances)} shapes, "
                f"{time.time() - time_test:.0f}s): final error={np.mean(errors):.6f}"
            )

            renders = viz.render_meshes(meshes, size=224, aa_factor=2)
            ws.save_renders(expdir, renders, str(history["epoch"]) + "_" + sname)

            latent_subdir = ws.get_recon_latent_subdir(expdir, history["epoch"])
            mesh_subdir = ws.get_recon_mesh_subdir(expdir, history["epoch"])
            os.makedirs(latent_subdir, exist_ok=True)
            os.makedirs(mesh_subdir, exist_ok=True)
            for i, instance in enumerate(test_instances):
                torch.save(latents[i], os.path.join(latent_subdir, instance + ".pth"))
                meshes[i].export(os.path.join(mesh_subdir, instance + ".obj"))

    torch.cuda.empty_cache()

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(
        duration // 3600, (duration // 60) % 60, duration % 60
    )
    logging.info(f"End of training after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    main(parser())
