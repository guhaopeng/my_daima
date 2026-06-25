"""
Train a first-version Part-aware VAE on top of a frozen PartSDF decoder.
"""

import argparse
import json
import logging
import os
import os.path
import sys
import time
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.data import PartLatentPoseDataset
from src.loss import get_loss_recon
from src.model import get_model
from src.optimizer import get_optimizer, get_scheduler
from src.utils import configure_logging, set_seed, get_device, clamp_sdf


def parser(argv=None):
    parser_ = argparse.ArgumentParser(
        description="Train a Part-aware VAE on top of a frozen PartSDF decoder."
    )
    parser_.add_argument("experiment", help="path to the experiment directory")
    parser_.add_argument("--debug", action="store_true")
    parser_.add_argument("--load-epoch", type=int, default=None)
    parser_.add_argument("--no-resume", action="store_true")
    parser_.add_argument("--reset", action="store_true")
    parser_.add_argument("--seed", type=int, default=0)
    parser_.add_argument("--workers", type=int, default=16)
    parser_.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    return parser_.parse_args(argv)


def load_partsdf_training_state(expdir):
    checkpoint_path = os.path.join(expdir, ws.CHECKPOINT_FILE)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = int(checkpoint["epoch"])
        model_state = checkpoint["model_state_dict"]
        latents_state = checkpoint["latents_state_dict"]
    else:
        history = ws.load_history(expdir)
        epoch = int(history["epoch"])
        model_state = torch.load(
            os.path.join(expdir, "model", f"model_{epoch}.pth"), map_location="cpu"
        )
        latents_state = torch.load(
            os.path.join(expdir, "latent", f"latents_{epoch}.pth"), map_location="cpu"
        )

    pose_path = os.path.join(expdir, "latent", f"poses_{epoch}.pth")
    if not os.path.isfile(pose_path):
        pose_path = os.path.join(expdir, "latent", "poses.pth")
    pose_state = torch.load(pose_path, map_location="cpu")
    return epoch, model_state, latents_state, pose_state


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())


def prepare_partsdf_inputs(pred_pose):
    rotation = pred_pose[..., :4]
    translation = pred_pose[..., 4:7]
    scale = pred_pose[..., 7:10]
    return rotation.unsqueeze(1), translation.unsqueeze(1), scale.unsqueeze(1)


def run_epoch(
    model,
    frozen_partsdf,
    dataloader,
    device,
    loss_recon,
    recon_lambda,
    latent_lambda,
    pose_lambda,
    kl_lambda,
    clampD,
    optimizer=None,
    use_occ=False,
):
    is_train = optimizer is not None
    model.train(is_train)
    frozen_partsdf.eval()

    total = {
        "loss": 0.0,
        "loss_sdf": 0.0,
        "loss_lat": 0.0,
        "loss_pose": 0.0,
        "loss_kl": 0.0,
    }
    n_shapes = 0

    for batch in dataloader:
        _, surface, xyz, sdf_gt, input_pose, target_part_lat, target_pose = batch
        batch_size = surface.shape[0]
        n_shapes += batch_size

        surface = surface.to(device)
        xyz = xyz.to(device)
        sdf_gt = sdf_gt.to(device)
        input_pose = input_pose.to(device)
        target_part_lat = target_part_lat.to(device)
        target_pose = target_pose.to(device)

        pred_part_lat, pred_pose, mu_g, logvar_g, mu_p, logvar_p, _, _ = model(
            surface,
            input_pose,
            sample=is_train,
            return_dist=True,
        )

        R, t, s = prepare_partsdf_inputs(pred_pose)
        sdf_pred = frozen_partsdf(pred_part_lat.unsqueeze(1), xyz, R=R, t=t, s=s)
        if use_occ:
            sdf_gt = (sdf_gt <= 0.0).float()
        elif clampD is not None and clampD > 0.0:
            sdf_pred = clamp_sdf(sdf_pred, clampD, ref=sdf_gt)
            sdf_gt = clamp_sdf(sdf_gt, clampD)

        loss_sdf = loss_recon(sdf_pred, sdf_gt).mean()
        loss_lat = F.mse_loss(pred_part_lat, target_part_lat)
        loss_pose = F.mse_loss(pred_pose, target_pose)
        loss_kl = kl_divergence(mu_g, logvar_g) + kl_divergence(mu_p, logvar_p)
        loss = (
            recon_lambda * loss_sdf
            + latent_lambda * loss_lat
            + pose_lambda * loss_pose
            + kl_lambda * loss_kl
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Encountered non-finite Part-aware VAE loss: "
                f"loss={loss.detach().item()} "
                f"loss_sdf={loss_sdf.detach().item()} "
                f"loss_lat={loss_lat.detach().item()} "
                f"loss_pose={loss_pose.detach().item()} "
                f"loss_kl={loss_kl.detach().item()} "
                f"surface_finite={torch.isfinite(surface).all().item()} "
                f"xyz_finite={torch.isfinite(xyz).all().item()} "
                f"sdf_gt_finite={torch.isfinite(sdf_gt).all().item()} "
                f"input_pose_finite={torch.isfinite(input_pose).all().item()} "
                f"target_part_lat_finite={torch.isfinite(target_part_lat).all().item()} "
                f"target_pose_finite={torch.isfinite(target_pose).all().item()}"
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total["loss"] += float(loss.detach()) * batch_size
        total["loss_sdf"] += float(loss_sdf.detach()) * batch_size
        total["loss_lat"] += float(loss_lat.detach()) * batch_size
        total["loss_pose"] += float(loss_pose.detach()) * batch_size
        total["loss_kl"] += float(loss_kl.detach()) * batch_size

    for key in total:
        total[key] /= max(n_shapes, 1)
    return total


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
    logging.info(f"Running experiment in {expdir}. (on {device})")

    partsdf_expdir = specs["PartSdfExperimentDir"]
    partsdf_specs = ws.load_specs(partsdf_expdir)
    loaded_epoch, partsdf_state, parts_lat_state, parts_pose_state = load_partsdf_training_state(
        partsdf_expdir
    )
    logging.info(f"Loaded frozen PartSDF checkpoint from epoch={loaded_epoch}.")

    parts_cfg = partsdf_specs["Parts"]
    n_parts = int(parts_cfg["NumParts"])
    part_latent_dim = int(parts_cfg["LatentDim"])
    pose_dim = int(parts_pose_state["weight"].shape[-1])
    use_occ = (
        partsdf_specs.get("ImplicitField", "SDF").lower() in ["occ", "occupancy"]
    )
    frozen_partsdf = get_model(
        partsdf_specs.get("Network", "PartSDF-PartSDF"),
        **partsdf_specs.get("NetworkSpecs", {}),
        n_parts=n_parts,
        part_dim=part_latent_dim,
        use_occ=use_occ,
    ).to(device)
    frozen_partsdf.load_state_dict(partsdf_state)
    frozen_partsdf.eval()
    for param in frozen_partsdf.parameters():
        param.requires_grad_(False)

    model = get_model(
        specs.get("Network", "PartAwareVAE"),
        **specs.get("NetworkSpecs", {}),
        n_parts=n_parts,
        part_latent_dim=part_latent_dim,
        pose_dim=pose_dim,
    ).to(device)

    train_dataset = PartLatentPoseDataset(
        datadir=specs["DataSource"],
        split=specs["TrainSplit"],
        latent_order_split=specs["LatentOrderSplit"],
        checkpoint_latent_file=os.path.join(partsdf_expdir, "latent", f"latents_{loaded_epoch}.pth"),
        checkpoint_pose_file=os.path.join(partsdf_expdir, "latent", "poses.pth"),
        sdf_n_samples=specs["SamplesPerScene"],
        surface_n_samples=specs["SurfaceSamplesPerScene"],
        samples_dir=specs["SamplesDir"],
        sdf_sample_file=specs["SamplesFile"],
        surface_sample_file=specs["SurfaceSamplesFile"],
        pose_param_dir=specs["PartsPoseDir"],
        balance=specs.get("BalanceSamples", True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=specs["ScenesPerBatch"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    valid_loader = None
    if specs.get("ValidSplit", None) is not None:
        valid_dataset = PartLatentPoseDataset(
            datadir=specs["DataSource"],
            split=specs["ValidSplit"],
            latent_order_split=specs["LatentOrderSplit"],
            checkpoint_latent_file=os.path.join(partsdf_expdir, "latent", f"latents_{loaded_epoch}.pth"),
            checkpoint_pose_file=os.path.join(partsdf_expdir, "latent", "poses.pth"),
            sdf_n_samples=specs["SamplesPerScene"],
            surface_n_samples=specs["SurfaceSamplesPerScene"],
            samples_dir=specs["SamplesDir"],
            sdf_sample_file=specs["SamplesFile"],
            surface_sample_file=specs["SurfaceSamplesFile"],
            pose_param_dir=specs["PartsPoseDir"],
            balance=specs.get("BalanceSamples", True),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=specs["ScenesPerBatch"],
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    loss_recon = get_loss_recon(specs["ReconLoss"], reduction="none").to(device)
    optimizer = get_optimizer(
        [model],
        type=specs["Optimizer"].pop("Type"),
        lrs=specs["Optimizer"].pop("LearningRates"),
        **specs["Optimizer"],
    )
    scheduler = get_scheduler(optimizer, **specs["LearningRateSchedule"])

    history = {"epoch": 0}
    if not args.no_resume:
        checkpoint_path = ws.get_checkpoint_filename(expdir)
        if args.load_epoch is None and os.path.isfile(checkpoint_path):
            checkpoint = ws.load_checkpoint(expdir)
            ws.use_checkpoint(checkpoint, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, checkpoint["epoch"])
            logging.info(f"Loaded checkpoint from epoch={history['epoch']}.")
        elif args.load_epoch is not None and os.path.isfile(os.path.join(expdir, ws.HISTORY_FILE)):
            ws.load_experiment(expdir, args.load_epoch, model, None, optimizer, scheduler)
            history = ws.load_history(expdir, args.load_epoch)
            logging.info(f"Loaded checkpoint from epoch={history['epoch']}.")

    metric_names = ["loss", "loss_sdf", "loss_lat", "loss_pose", "loss_kl", "lr"]
    if valid_loader is not None:
        metric_names.extend(
            ["val_loss", "val_loss_sdf", "val_loss_lat", "val_loss_pose", "val_loss_kl"]
        )
    for key in metric_names:
        history.setdefault(key, [])

    n_epochs = specs["NumEpochs"]
    clampD = specs.get("ClampingDistance", None)
    recon_lambda = float(specs.get("ReconLossLambda", 1.0))
    latent_lambda = float(specs.get("PartLatentReconLambda", 1.0))
    pose_lambda = float(specs.get("PoseReconLambda", 0.5))
    kl_lambda = float(specs.get("KLLossLambda", 1e-4))
    log_frequency = int(specs.get("LogFrequency", 10))
    snapshot_frequency = int(specs.get("SnapshotFrequency", 200))

    for epoch in range(history["epoch"] + 1, n_epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model=model,
            frozen_partsdf=frozen_partsdf,
            dataloader=train_loader,
            device=device,
            loss_recon=loss_recon,
            recon_lambda=recon_lambda,
            latent_lambda=latent_lambda,
            pose_lambda=pose_lambda,
            kl_lambda=kl_lambda,
            clampD=clampD,
            optimizer=optimizer,
            use_occ=use_occ,
        )

        valid_metrics = None
        if valid_loader is not None:
            with torch.no_grad():
                valid_metrics = run_epoch(
                    model=model,
                    frozen_partsdf=frozen_partsdf,
                    dataloader=valid_loader,
                    device=device,
                    loss_recon=loss_recon,
                    recon_lambda=recon_lambda,
                    latent_lambda=latent_lambda,
                    pose_lambda=pose_lambda,
                    kl_lambda=kl_lambda,
                    clampD=clampD,
                    optimizer=None,
                    use_occ=use_occ,
                )

        history["epoch"] += 1
        history["loss"].append(train_metrics["loss"])
        history["loss_sdf"].append(train_metrics["loss_sdf"])
        history["loss_lat"].append(train_metrics["loss_lat"])
        history["loss_pose"].append(train_metrics["loss_pose"])
        history["loss_kl"].append(train_metrics["loss_kl"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if valid_metrics is not None:
            history["val_loss"].append(valid_metrics["loss"])
            history["val_loss_sdf"].append(valid_metrics["loss_sdf"])
            history["val_loss_lat"].append(valid_metrics["loss_lat"])
            history["val_loss_pose"].append(valid_metrics["loss_pose"])
            history["val_loss_kl"].append(valid_metrics["loss_kl"])

        if scheduler is not None:
            scheduler.step()

        if epoch % log_frequency == 0:
            ws.save_history(expdir, history)
            checkpoint = ws.build_checkpoint(epoch, model, None, optimizer, scheduler)
            ws.save_checkpoint(expdir, checkpoint)
            del checkpoint

        if epoch % snapshot_frequency == 0:
            ws.save_experiment(expdir, epoch, model, None, optimizer, scheduler)

        msg = (
            f"Epoch {epoch}/{n_epochs}: "
            f"loss={train_metrics['loss']:.6f} - "
            f"loss_sdf={train_metrics['loss_sdf']:.6f} - "
            f"loss_lat={train_metrics['loss_lat']:.6f} - "
            f"loss_pose={train_metrics['loss_pose']:.6f} - "
            f"loss_kl={train_metrics['loss_kl']:.6f}"
        )
        if valid_metrics is not None:
            msg += (
                f" - val={valid_metrics['loss']:.6f}"
                f" - val_sdf={valid_metrics['loss_sdf']:.6f}"
            )
        msg += f" ({time.time() - t0:.1f}s/epoch)"
        logging.info(msg)

    final_epoch = history["epoch"]
    checkpoint = ws.build_checkpoint(final_epoch, model, None, optimizer, scheduler)
    ws.save_checkpoint(expdir, checkpoint)
    ws.save_history(expdir, history)
    ws.save_experiment(expdir, final_epoch, model, None, optimizer, scheduler)

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(
        duration // 3600, (duration // 60) % 60, duration % 60
    )
    logging.info(f"End of training after {duration_msg}.")


if __name__ == "__main__":
    main(parser())
