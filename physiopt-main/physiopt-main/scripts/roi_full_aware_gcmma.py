import glob
import json
import os
import random
import sys
import zlib
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import polyscope.imgui as psim

from physiopt.vis.ui_config import UiConfig, GLOBAL_UI_CONFIG
from physiopt.vis.viewer_base_mma_gudingli import ViewerBase
from physiopt.utils.phys_utils import generate_hex
from physiopt.utils.grid_utils import filter_one_component
from physiopt.opt.forces import ForceConfig
from physiopt.opt.boundary import BoundaryCondType
from physiopt.opt.optimizer_state import OptimizerConfig
from physiopt.opt.field import occ_kernel

try:
    from pytorch.mma import asymp, concheck, gcmmasub, kktcheck, raaupdate

    HAS_MMA = True
except ImportError:
    HAS_MMA = False


GCMMA_EPSIMIN = 1e-7
GCMMA_RAA0EPS = 1e-6
GCMMA_RAAEPS = 1e-6
GCMMA_MAX_INNERIT = 15
GCMMA_MOVE = 1e-3


class ViewerPartAwareVae(ViewerBase):
    def init_network(self, args) -> OptimizerConfig:
        self.global_volume_val = None
        self.global_volume_constraint_val = None
        self.seed = int(args.seed)
        self.deterministic_eval = bool(args.deterministic_eval)
        self._set_reproducible_seed_(self.seed)

        self.partsdf_root = self._resolve_partsdf_root_(args.partsdf_root)
        if str(self.partsdf_root) not in sys.path:
            sys.path.insert(0, str(self.partsdf_root))

        from src import workspace as ws  # noqa: E402
        from src.model import get_model  # noqa: E402

        self.ws = ws
        self.get_model = get_model

        self.expdir = self._resolve_experiment_dir_(args.exp)
        self.specs = self.ws.load_specs(str(self.expdir))
        self.use_mean_latent = bool(self.specs.get("UseMeanLatentAtEval", True))

        self.partsdf_expdir = self._resolve_partsdf_experiment_dir_(
            self.specs["PartSdfExperimentDir"]
        )
        self.partsdf_specs = self.ws.load_specs(str(self.partsdf_expdir))
        self.data_root = self._resolve_data_root_()
        self.surface_samples_per_scene = int(
            self.specs.get("SurfaceSamplesPerScene", 2048)
        )
        self.pose_param_dir = self.specs.get("PartsPoseDir", "parts/parameters")

        self._init_frozen_partsdf_()
        self._init_part_aware_vae_()

        self.instances = self._load_instances_(args.split)
        if len(self.instances) == 0:
            raise RuntimeError("No instances found in the requested split.")

        self.current_instance_index = min(
            max(int(args.shape_idx), 0), len(self.instances) - 1
        )
        if args.instance is not None:
            if args.instance not in self.instances:
                raise ValueError(
                    f'Instance "{args.instance}" is not present in split "{args.split}".'
                )
            self.current_instance_index = self.instances.index(args.instance)

        self.instance_id = self.instances[self.current_instance_index]
        self.latent = self._infer_instance_latent_(self.instance_id)
        self.latent = self.latent.detach().clone().cuda().requires_grad_(True)

        default_gravity_force = ForceConfig(external_force=[0.0, -500.0, 0.0])
        default_seat_force = ForceConfig(external_force=[0.0, -1000.0, 0.0])
        default_back_force = ForceConfig(external_force=[0.0, 0.0, 1000.0])

        config = OptimizerConfig(
            lr=1e-2,
            boundary_cond=BoundaryCondType.BOTTOM_Y,
            init_forces=[default_gravity_force, default_seat_force, default_back_force],
            res=32,
        )
        config.latent_optimizer = "mma"
        config.latent_clip = None
        config.mma_latent_shape_reg = 0.0
        config.volume_fraction_target = float(args.roi_volume_fraction_target)
        config.mma_volume_roi_margin = 3
        config.freeze_force_region_in_space = True
        config.fixed_force_y_layers = 2

        self._bootstrap_config = config
        self._set_optimize_mode_(args.optimize_mode)
        return config

    def _init_frozen_partsdf_(self) -> None:
        parts_cfg = self.partsdf_specs["Parts"]
        self.n_parts = int(parts_cfg["NumParts"])
        self.part_latent_dim = int(parts_cfg["LatentDim"])
        self.pose_dim = 10 if bool(parts_cfg.get("UsePoses", False)) else 0
        self.use_occ = (
            self.partsdf_specs.get("ImplicitField", "SDF").lower()
            in ["occ", "occupancy"]
        )

        self.frozen_partsdf = self.get_model(
            self.partsdf_specs.get("Network", "PartSDF-PartSDF"),
            **self.partsdf_specs.get("NetworkSpecs", {}),
            n_parts=self.n_parts,
            part_dim=self.part_latent_dim,
            use_occ=self.use_occ,
        ).cuda()

        partsdf_checkpoint = self._load_partsdf_training_checkpoint_()
        self.frozen_partsdf.load_state_dict(partsdf_checkpoint["model_state_dict"])
        self.frozen_partsdf.eval()
        for param in self.frozen_partsdf.parameters():
            param.requires_grad_(False)

    def _init_part_aware_vae_(self) -> None:
        network_specs = dict(self.specs.get("NetworkSpecs", {}))
        self.global_latent_dim = int(network_specs.get("global_latent_dim", 64))
        self.part_latent_code_dim = int(network_specs.get("part_latent_code_dim", 16))

        self.model = self.get_model(
            self.specs.get("Network", "PartAwareVAE"),
            **network_specs,
            n_parts=self.n_parts,
            part_latent_dim=self.part_latent_dim,
            pose_dim=self.pose_dim,
        ).cuda()
        self._load_part_aware_vae_weights_()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def shape_selection_gui(self) -> bool:
        clicked, self.current_instance_index = psim.InputInt(
            "shape_idx",
            self.current_instance_index,
            step=1,
        )
        self.current_instance_index = min(
            max(self.current_instance_index, 0), len(self.instances) - 1
        )
        psim.Text(f"instance={self.instance_id}")
        psim.Text(f"latent_norm={float(self.latent.detach().norm()):.4f}")
        psim.Text(f"roi target = {float(self.config.volume_fraction_target):.4f}")
        if self.volume_ratio_val is not None:
            psim.Text(f"roi volume ratio = {float(self.volume_ratio_val):.4f}")
        if self.global_volume_val is not None:
            psim.Text(f"global volume fraction = {float(self.global_volume_val):.4f}")
        if clicked:
            self.get_latent(self.current_instance_index)
        return clicked

    @torch.no_grad()
    def get_latent(self, t: int = 0):
        self.current_instance_index = min(max(int(t), 0), len(self.instances) - 1)
        self.instance_id = self.instances[self.current_instance_index]
        self.latent = self._infer_instance_latent_(self.instance_id)
        self._project_latent_()
        self._store_reference_latent_()
        self._clear_mma_reference_scalars_()
        self.latent.requires_grad_(True)

    @torch.no_grad()
    def _store_reference_latent_(self) -> None:
        self.initial_latent = self.latent.detach().clone().cuda()

    @torch.no_grad()
    def _clear_mma_reference_scalars_(self) -> None:
        self.initial_compliance_ref = None
        self.initial_volume_ref = None
        self.volume_ratio_val = None
        self.global_volume_val = None
        self.global_volume_constraint_val = None
        self.volume_constraint_val = None
        self.fixed_volume_initial_value = None
        self.fixed_volume_roi_nodes = None
        self.fixed_volume_roi_elements = None

    @staticmethod
    def _instance_seed_(instance_id: str, seed: int) -> int:
        return int((zlib.crc32(instance_id.encode("utf-8")) + int(seed)) % (2**32))

    @staticmethod
    def _set_reproducible_seed_(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _rescale_model_input_(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    def _pack_latent_state_(self, z_g: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_g.reshape(-1), z_p.reshape(-1)], dim=0).detach().clone().cuda()

    def _unpack_latent_state_(
        self, state_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_size = self.global_latent_dim
        z_g = state_vector[:global_size].view(self.global_latent_dim)
        z_p = state_vector[global_size:].view(self.n_parts, self.part_latent_code_dim)
        return z_g, z_p

    def _prepare_partsdf_inputs_(
        self, pred_pose: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.pose_dim <= 0:
            return None, None, None
        rotation = pred_pose[..., :4]
        translation = pred_pose[..., 4:7]
        scale = pred_pose[..., 7:10]
        return rotation.unsqueeze(0).unsqueeze(0), translation.unsqueeze(0).unsqueeze(
            0
        ), scale.unsqueeze(0).unsqueeze(0)

    def _decode_sdf_(
        self, x: torch.Tensor, state_vector: torch.Tensor | None = None
    ) -> torch.Tensor:
        practical_state = self.latent if state_vector is None else state_vector
        z_g, z_p = self._unpack_latent_state_(practical_state)
        pred_part_latent, pred_pose = self.model.decode(
            z_g.unsqueeze(0), z_p.unsqueeze(0)
        )
        xyz = self._rescale_model_input_(x).unsqueeze(0)
        part_latent = pred_part_latent.unsqueeze(1)

        if self.pose_dim > 0:
            rotation, translation, scale = self._prepare_partsdf_inputs_(
                pred_pose.squeeze(0)
            )
            sdf = self.frozen_partsdf(
                part_latent,
                xyz,
                R=rotation,
                t=translation,
                s=scale,
            )
        else:
            sdf = self.frozen_partsdf(part_latent, xyz)
        return sdf.reshape(-1)

    def extract_solid(self):
        full_nodes, full_elements = self._get_full_grid_()
        sdf = self._decode_sdf_(full_nodes)
        if not torch.isfinite(sdf).all():
            raise RuntimeError("Decoded SDF contains non-finite values.")

        full_occ_nodes = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
        self.full_occ = full_occ_nodes[full_elements].mean(1)
        self.full_nodes = full_nodes
        self.full_elements = full_elements

        nodewise_mask = sdf <= self._get_sdf_filter_value_()
        elementwise_mask = torch.any(nodewise_mask[full_elements], dim=1)
        if self.config.one_cc_only:
            elementwise_mask = filter_one_component(
                elementwise_mask, self.config.res - 1
            )
        if not torch.any(elementwise_mask):
            raise RuntimeError("Part-aware VAE optimization produced an empty solid.")

        selected_elements = full_elements[elementwise_mask]
        selected_node_indices, selected_elements = torch.unique(
            selected_elements, return_inverse=True
        )
        self.elements = selected_elements
        self.nodes = full_nodes[selected_node_indices]

        coarse_sdf = sdf[selected_node_indices]
        coarse_occ_nodes = occ_kernel(
            coarse_sdf, self.config.res, self.config.occ_sdf_beta
        )
        self.coarse_occ = coarse_occ_nodes[self.elements].mean(1)
        self.coarse_coords = (
            (self.nodes[self.elements].mean(1) + 0.5) * (self.config.res - 1)
        ).long()

    def init_optimizer(self):
        self.latent.requires_grad_(True)
        self._init_mma_state()

    def backward_step(self):
        def f_compliance(state_vector: torch.Tensor):
            sdf = self._decode_sdf_(self.nodes, state_vector=state_vector)
            occ = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
            occ = occ[self.elements].mean(1)
            if self.config.alpha_min > 0.0:
                occ = torch.clip(occ, self.config.alpha_min)
            return occ

        def f_volume(state_vector: torch.Tensor):
            return self._compute_roi_occ_(state_vector)

        state_for_grad = self.latent.detach().clone().requires_grad_(True)
        active_mask = self._get_active_mask_(dtype=torch.float32)
        latent_reg_weight = float(getattr(self.config, "mma_latent_shape_reg", 0.0))

        grad_f0 = torch.autograd.functional.vjp(
            f_compliance, state_for_grad, self.sensitivity_density_c
        )[1]
        grad_f1 = torch.autograd.functional.vjp(
            f_volume, state_for_grad, self.sensitivity_density_v
        )[1]

        grad_f0 = torch.nan_to_num(grad_f0, nan=0.0) * active_mask
        grad_f1 = torch.nan_to_num(grad_f1, nan=0.0) * active_mask

        if latent_reg_weight > 0.0 and hasattr(self, "initial_latent"):
            grad_f0 = grad_f0 + 2.0 * latent_reg_weight * (
                state_for_grad - self.initial_latent
            ) * active_mask

        if self.initial_compliance_ref is None:
            self.initial_compliance_ref = max(float(self.compliance_val), 1e-12)
        compliance_ref = self.initial_compliance_ref

        if self.initial_volume_ref is None:
            self.initial_volume_ref = max(float(self.fixed_volume_initial_value), 1e-12)
        volume_ref = self.initial_volume_ref
        volume_target = float(getattr(self.config, "volume_fraction_target", 1.0))

        self.latent.grad = None
        self.f0val_mma = torch.tensor(
            [[self.compliance_val / compliance_ref]],
            dtype=torch.float32,
            device=self.device,
        )
        self.fval_mma = torch.tensor(
            [[self.volume_val / volume_ref - volume_target]],
            dtype=torch.float32,
            device=self.device,
        )
        self.mma_df0dx = (grad_f0 / compliance_ref).view(self.mma_n, 1).to(torch.float64)
        self.mma_dfdx = (grad_f1 / volume_ref).view(1, -1).to(torch.float64)

    @torch.no_grad()
    def _clear_simulation_state_(self) -> None:
        for name in [
            "nodes",
            "elements",
            "solid",
            "u",
            "f_int",
            "f_ext",
            "sigma",
            "coarse_occ",
            "full_occ",
        ]:
            if hasattr(self, name):
                try:
                    delattr(self, name)
                except AttributeError:
                    pass
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _update_density_metrics_from_current_state_(self) -> None:
        k0 = self.solid.k0()
        u_j = self.u[self.elements].reshape(self.solid.n_elem, -1)
        w_k = torch.einsum("...i, ...ij, ...j", u_j, k0, u_j)

        self.compliance_val = w_k.sum().item()
        self.sensitivity_density_c = (
            -self.config.p_exponent
            * self.coarse_occ ** (self.config.p_exponent - 1)
            * w_k
        )
        self._update_roi_volume_metrics_()

    @torch.no_grad()
    def _evaluate_mma_candidate_(
        self, xmma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.latent = (
            xmma.view_as(self.latent).to(torch.float32).detach().clone().requires_grad_(True)
        )
        self._project_latent_()
        self._clear_simulation_state_()
        self.prepare_solid()
        self.u, self.f_int, self.f_ext, self.sigma, _, _ = self.solid.solve(max_iter=1)
        self._update_density_metrics_from_current_state_()

        compliance_ref = max(float(self.initial_compliance_ref), 1e-12)
        volume_ref = max(float(self.initial_volume_ref), 1e-12)
        volume_target = float(getattr(self.config, "volume_fraction_target", 1.0))
        f0valnew = torch.tensor(
            [[self.compliance_val / compliance_ref]],
            dtype=torch.float64,
            device=self.device,
        )
        fvalnew = torch.tensor(
            [[self.volume_val / volume_ref - volume_target]],
            dtype=torch.float64,
            device=self.device,
        )
        return f0valnew, fvalnew

    def update_mma_step(self) -> None:
        if not HAS_MMA:
            raise ImportError(
                "pytorch.mma is not available. Install it to run GCMMA optimization."
            )

        self.mma_outeriter += 1
        epsimin = GCMMA_EPSIMIN
        f0val_mma_64 = self.f0val_mma.to(torch.float64)
        fval_mma_64 = self.fval_mma.to(torch.float64)

        self.mma_low, self.mma_upp, self.mma_raa0, self.mma_raa = asymp(
            self.mma_outeriter,
            self.mma_n,
            self.mma_xval,
            self.mma_xold1,
            self.mma_xold2,
            self.mma_xmin,
            self.mma_xmax,
            self.mma_low,
            self.mma_upp,
            self.mma_raa0,
            self.mma_raa,
            GCMMA_RAA0EPS,
            GCMMA_RAAEPS,
            self.mma_df0dx,
            self.mma_dfdx,
            asyinit=1e-3,
        )

        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
            self.mma_m,
            self.mma_n,
            self.mma_outeriter,
            epsimin,
            self.mma_xval,
            self.mma_xmin,
            self.mma_xmax,
            self.mma_low,
            self.mma_upp,
            self.mma_raa0,
            self.mma_raa,
            f0val_mma_64,
            self.mma_df0dx,
            fval_mma_64,
            self.mma_dfdx,
            self.mma_a0,
            self.mma_a,
            self.mma_c,
            self.mma_d,
            move=GCMMA_MOVE,
        )
        f0valnew, fvalnew = self._evaluate_mma_candidate_(xmma)
        conserv = concheck(
            self.mma_m,
            epsimin,
            f0app,
            f0valnew,
            fapp,
            fvalnew,
        )

        innerit = 0
        while conserv == 0 and innerit < GCMMA_MAX_INNERIT:
            innerit += 1
            self.mma_raa0, self.mma_raa = raaupdate(
                xmma,
                self.mma_xval,
                self.mma_xmin,
                self.mma_xmax,
                self.mma_low,
                self.mma_upp,
                f0valnew,
                fvalnew,
                f0app,
                fapp,
                self.mma_raa0,
                self.mma_raa,
                raa0eps=GCMMA_RAA0EPS,
                raaeps=GCMMA_RAAEPS,
                epsimin=epsimin,
            )
            xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                self.mma_m,
                self.mma_n,
                self.mma_outeriter,
                epsimin,
                self.mma_xval,
                self.mma_xmin,
                self.mma_xmax,
                self.mma_low,
                self.mma_upp,
                self.mma_raa0,
                self.mma_raa,
                f0val_mma_64,
                self.mma_df0dx,
                fval_mma_64,
                self.mma_dfdx,
                self.mma_a0,
                self.mma_a,
                self.mma_c,
                self.mma_d,
                move=GCMMA_MOVE,
            )
            f0valnew, fvalnew = self._evaluate_mma_candidate_(xmma)
            conserv = concheck(
                self.mma_m,
                epsimin,
                f0app,
                f0valnew,
                fapp,
                fvalnew,
            )

        residu, kktnorm, residumax = kktcheck(
            self.mma_m,
            self.mma_n,
            xmma,
            ymma,
            zmma,
            lam,
            xsi,
            eta,
            mu,
            zet,
            s,
            self.mma_xmin,
            self.mma_xmax,
            self.mma_df0dx,
            fval_mma_64,
            self.mma_dfdx,
            self.mma_a0,
            self.mma_a,
            self.mma_c,
            self.mma_d,
        )
        self.debug_gcmma_kktnorm = float(kktnorm)
        self.debug_gcmma_residumax = float(residumax)
        self.debug_gcmma_innerit = int(innerit)
        self.debug_gcmma_conserv = int(conserv)

        self.mma_xold2 = self.mma_xold1.clone()
        self.mma_xold1 = self.mma_xval.clone()
        self.mma_xval = xmma.clone()
        self.latent = (
            self.mma_xval.view_as(self.latent).to(torch.float32).detach().clone().requires_grad_(True)
        )
        self._project_latent_()

    def get_field(self, x: torch.Tensor):
        return self._decode_sdf_(x)

    def field_to_occ(self, x: torch.Tensor) -> torch.Tensor:
        return occ_kernel(x, self.config.res, self.config.occ_sdf_beta)

    @property
    def mc_isovalue(self) -> float:
        return 0.0

    @property
    def mc_needed(self) -> bool:
        return True

    def set_replay_state(self, opt_state) -> bool:
        if opt_state.latent is None:
            return False
        self.latent = opt_state.latent.detach().clone().cuda().requires_grad_(True)
        self._project_latent_()
        self._store_reference_latent_()
        self._clear_mma_reference_scalars_()
        return True

    @torch.no_grad()
    def _pre_optimize(self, keep_latent: bool = False):
        super()._pre_optimize(keep_latent=keep_latent)
        self._update_fixed_volume_roi_()

    def pre_init_trajectory(
        self, keep_latent: bool = True, replace_current: bool = False
    ) -> None:
        if keep_latent and self.current_state.latent is not None:
            self.latent = self.current_state.latent.detach().clone().cuda().requires_grad_(True)
            self._project_latent_()
            self._store_reference_latent_()
        else:
            self.get_latent(self.current_instance_index)
        self._clear_mma_reference_scalars_()

    @torch.no_grad()
    def _project_latent_(self) -> None:
        z_g, z_p = self._unpack_latent_state_(self.latent)
        config = self._get_effective_config_()
        max_norm = getattr(config, "latent_clip", None) if config is not None else None
        if max_norm is not None:
            global_norm = z_g.norm()
            if global_norm > max_norm:
                z_g = z_g * (float(max_norm) / (global_norm + 1e-8))
            part_norm = z_p.norm(dim=-1, keepdim=True)
            scale = torch.clamp(float(max_norm) / (part_norm + 1e-8), max=1.0)
            z_p = z_p * scale
        self.latent.copy_(self._pack_latent_state_(z_g, z_p))

    def _get_effective_config_(self) -> OptimizerConfig | None:
        try:
            return self.config
        except Exception:
            return getattr(self, "_bootstrap_config", None)

    def _get_sdf_filter_value_(self) -> float:
        return float(getattr(self.config, "sdf_filter_scale", 1.0)) / self.config.res

    def _get_volume_display_value_(self) -> float:
        if self.volume_ratio_val is None:
            return super()._get_volume_display_value_()
        return float(self.volume_ratio_val)

    def _get_global_volume_display_value_(self) -> float:
        if self.global_volume_val is None:
            return float("nan")
        return float(self.global_volume_val)

    @torch.no_grad()
    def _get_full_grid_(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "full_grid_nodes") or self.full_grid_nodes is None:
            self.full_grid_nodes, self.full_grid_elements = generate_hex(self.config.res)
        return self.full_grid_nodes, self.full_grid_elements

    @torch.no_grad()
    def _extract_element_mask_from_sdf_(self, sdf: torch.Tensor) -> torch.Tensor:
        _, full_elements = self._get_full_grid_()
        nodewise_mask = sdf <= self._get_sdf_filter_value_()
        elementwise_mask = torch.any(nodewise_mask[full_elements], dim=1)
        if self.config.one_cc_only:
            elementwise_mask = filter_one_component(
                elementwise_mask, self.config.res - 1
            )
        return elementwise_mask

    @torch.no_grad()
    def _dilate_element_mask_(self, elementwise_mask: torch.Tensor) -> torch.Tensor:
        margin = max(int(getattr(self.config, "mma_volume_roi_margin", 0)), 0)
        if margin <= 0:
            return elementwise_mask

        grid_res = self.config.res - 1
        roi_volume = elementwise_mask.view(grid_res, grid_res, grid_res).float()
        roi_volume = roi_volume[None, None]
        kernel_size = 2 * margin + 1
        dilated = F.max_pool3d(
            roi_volume,
            kernel_size=kernel_size,
            stride=1,
            padding=margin,
        )
        return dilated[0, 0].bool().reshape(-1)

    @torch.no_grad()
    def _update_fixed_volume_roi_(self) -> None:
        full_nodes, full_elements = self._get_full_grid_()
        initial_sdf = self._decode_sdf_(full_nodes)
        initial_element_mask = self._extract_element_mask_from_sdf_(initial_sdf)
        roi_element_mask = self._dilate_element_mask_(initial_element_mask)
        if not torch.any(roi_element_mask):
            roi_element_mask = initial_element_mask
        if not torch.any(roi_element_mask):
            raise RuntimeError("Failed to build a fixed ROI for the volume constraint.")

        roi_elements = full_elements[roi_element_mask]
        roi_node_indices, roi_elements = torch.unique(
            roi_elements, return_inverse=True
        )
        self.fixed_volume_roi_nodes = full_nodes[roi_node_indices]
        self.fixed_volume_roi_elements = roi_elements

        self.init_volume = self._compute_roi_occ_().detach()
        self.fixed_volume_initial_value = max(float(self.init_volume.sum().item()), 1e-12)
        self.initial_volume_ref = self.fixed_volume_initial_value
        self._update_roi_volume_metrics_()

    def _compute_roi_occ_(self, state_vector: torch.Tensor | None = None) -> torch.Tensor:
        if self.fixed_volume_roi_nodes is None or self.fixed_volume_roi_elements is None:
            raise RuntimeError("Fixed ROI volume constraint has not been initialized.")
        sdf = self._decode_sdf_(self.fixed_volume_roi_nodes, state_vector=state_vector)
        roi_occ_nodes = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
        return roi_occ_nodes[self.fixed_volume_roi_elements].mean(1)

    @torch.no_grad()
    def _update_global_volume_metrics_(self) -> None:
        if not hasattr(self, "full_occ") or self.full_occ is None:
            self.global_volume_val = None
            self.global_volume_constraint_val = None
            return
        self.global_volume_val = float(self.full_occ.mean().item())
        self.global_volume_constraint_val = None

    @torch.no_grad()
    def _update_roi_volume_metrics_(self) -> None:
        self._update_global_volume_metrics_()
        roi_occ = self._compute_roi_occ_()
        self.volume_val = roi_occ.sum().item()
        self.sensitivity_density_v = torch.ones_like(roi_occ)
        if self.fixed_volume_initial_value is None:
            self.volume_ratio_val = None
        else:
            self.volume_ratio_val = self.volume_val / max(
                float(self.fixed_volume_initial_value), 1e-12
            )
        self.volume_constraint_val = (
            float(self.volume_ratio_val) - float(self.config.volume_fraction_target)
            if self.volume_ratio_val is not None
            else None
        )

    @torch.no_grad()
    def _update_volume_constraint_metrics_(self) -> None:
        self._update_roi_volume_metrics_()

    def _init_mma_state(self) -> None:
        self.mma_n = self.latent.numel()
        self.mma_m = 1
        dtype, device = torch.float64, self.latent.device
        self.mma_xval = self.latent.detach().to(dtype).view(self.mma_n, 1).clone()
        self.mma_xold1 = self.mma_xval.clone()
        self.mma_xold2 = self.mma_xval.clone()
        self.mma_xmin, self.mma_xmax = self._build_mma_bounds_(dtype=dtype, device=device)
        self.mma_low = self.mma_xmin.clone()
        self.mma_upp = self.mma_xmax.clone()
        self.mma_raa0 = 1e-4
        self.mma_raa = 1e-4 * torch.ones((self.mma_m, 1), dtype=dtype, device=device)
        self.mma_a0 = 0.0
        self.mma_a = torch.zeros((self.mma_m, 1), dtype=dtype, device=device)
        self.mma_c = 1e4 * torch.ones((self.mma_m, 1), dtype=dtype, device=device)
        self.mma_d = torch.zeros((self.mma_m, 1), dtype=dtype, device=device)
        self.mma_outeriter = 0

    def _build_mma_bounds_(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xval = self.latent.detach().to(dtype).view(self.mma_n, 1)
        xmin = -3.0 * torch.ones((self.mma_n, 1), dtype=dtype, device=device)
        xmax = 3.0 * torch.ones((self.mma_n, 1), dtype=dtype, device=device)
        inactive_mask = ~self._get_active_mask_(dtype=torch.bool).view(-1, 1)
        xmin[inactive_mask] = xval[inactive_mask] - 1e-12
        xmax[inactive_mask] = xval[inactive_mask] + 1e-12
        return xmin, xmax

    def _set_optimize_mode_(self, mode: str) -> None:
        self.optimize_global = mode in ["global", "both"]
        self.optimize_parts = mode in ["parts", "both"]
        self.optimize_mode = mode

    def _get_active_mask_(self, dtype: torch.dtype) -> torch.Tensor:
        global_size = self.global_latent_dim
        part_size = self.n_parts * self.part_latent_code_dim
        mask = torch.zeros(global_size + part_size, dtype=torch.bool, device=self.device)
        if self.optimize_global:
            mask[:global_size] = True
        if self.optimize_parts:
            mask[global_size:] = True
        return mask.to(dtype=dtype)

    def _resolve_partsdf_root_(self, explicit_root: str | None) -> Path:
        candidates = []
        if explicit_root:
            candidates.append(Path(explicit_root).expanduser().resolve())
        script_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                script_dir.parents[1] / "PartSDF-main",
                script_dir.parents[2] / "PartSDF-main",
                Path("c:/Users/dell/Downloads/PartSDF-main/PartSDF-main"),
            ]
        )
        for candidate in candidates:
            if (candidate / "src" / "model" / "__init__.py").is_file():
                return candidate
        raise FileNotFoundError(
            "Could not locate PartSDF-main. Pass --partsdf_root explicitly."
        )

    def _resolve_experiment_dir_(self, exp_arg: str) -> Path:
        expdir = Path(exp_arg).expanduser().resolve()
        if (expdir / self.ws.SPECS_FILE).is_file():
            return expdir
        nested = [
            p for p in expdir.iterdir() if p.is_dir() and (p / self.ws.SPECS_FILE).is_file()
        ]
        if len(nested) == 1:
            return nested[0]
        raise FileNotFoundError(
            f'Could not find a unique experiment directory with "{self.ws.SPECS_FILE}" under {expdir}.'
        )

    def _resolve_partsdf_experiment_dir_(self, exp_value: str) -> Path:
        exp_path = Path(exp_value)
        candidates = []
        if exp_path.is_absolute():
            candidates.append(exp_path)
        else:
            candidates.extend(
                [
                    self.partsdf_root / exp_path,
                    self.expdir / exp_path,
                    self.expdir.parent / exp_path,
                    self.partsdf_root.parent / exp_path,
                ]
            )
        for candidate in candidates:
            if (candidate / self.ws.SPECS_FILE).is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f'Could not resolve PartSdfExperimentDir "{exp_value}" from {self.expdir}.'
        )

    def _resolve_data_root_(self) -> Path:
        data_source = Path(self.specs["DataSource"])
        if data_source.is_absolute():
            return data_source
        candidates = [
            self.partsdf_root / data_source,
            self.expdir / data_source,
            self.expdir.parent / data_source,
            self.partsdf_root.parent / data_source,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f'Unable to resolve DataSource "{self.specs["DataSource"]}" from {self.expdir}.'
        )

    def _load_part_aware_vae_weights_(self) -> None:
        checkpoint_path = self.expdir / self.ws.CHECKPOINT_FILE
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.loaded_epoch = int(checkpoint["epoch"])
            print(
                f"Loading Part-aware VAE checkpoint: {checkpoint_path} "
                f"(epoch={self.loaded_epoch})"
            )
            return

        model_paths = glob.glob(str(self.expdir / self.ws.MODEL_DIR / "model_*.pth"))
        if not model_paths:
            raise FileNotFoundError(
                f"No checkpoint or model parameters found under {self.expdir}."
            )
        model_paths = sorted(model_paths, key=lambda p: int(Path(p).stem.split("_")[-1]))
        model_path = Path(model_paths[-1])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.loaded_epoch = int(model_path.stem.split("_")[-1])
        print(
            f"Loading Part-aware VAE model: {model_path} (epoch={self.loaded_epoch})"
        )

    def _load_partsdf_training_checkpoint_(self) -> dict:
        checkpoint_path = self.partsdf_expdir / self.ws.CHECKPOINT_FILE
        if checkpoint_path.is_file():
            return torch.load(checkpoint_path, map_location="cpu")

        history = self.ws.load_history(str(self.partsdf_expdir))
        epoch = int(history["epoch"])
        model_state_dict = torch.load(
            self.partsdf_expdir / "model" / f"model_{epoch}.pth", map_location="cpu"
        )
        return {"epoch": epoch, "model_state_dict": model_state_dict}

    def _load_instances_(self, split_name: str) -> list[str]:
        split_map = {
            "train": self.specs["TrainSplit"],
            "valid": self.specs.get("ValidSplit", None),
            "test": self.specs.get("TestSplit", None),
        }
        split_value = split_map[split_name]
        if split_value is None:
            raise ValueError(f'Split "{split_name}" is not defined in experiment specs.')
        split_path = Path(split_value)
        if split_path.is_absolute():
            resolved_split_path = split_path
        else:
            candidates = [
                self.partsdf_root / split_path,
                self.expdir / split_path,
                self.expdir.parent / split_path,
                self.partsdf_root.parent / split_path,
            ]
            resolved_split_path = next((p for p in candidates if p.is_file()), None)
            if resolved_split_path is None:
                raise FileNotFoundError(
                    f'Could not resolve split file "{split_value}" from {self.partsdf_root}.'
                )
        with open(resolved_split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_surface_sample_dir_(self, instance_id: str) -> Path:
        return self.data_root / self.specs["SamplesDir"] / instance_id

    def _resolve_surface_file_(self, instance_id: str) -> Path:
        sample_dir = self._get_surface_sample_dir_(instance_id)
        preferred = self.specs.get("SurfaceSamplesFile", "surface.npy")
        candidates = [sample_dir / preferred, sample_dir / "surface.npy"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f'Could not find a surface sample file for "{instance_id}" under {sample_dir}.'
        )

    def _load_surface_points_(self, instance_id: str) -> np.ndarray:
        surface_path = self._resolve_surface_file_(instance_id)
        surface = np.load(surface_path)
        surface = np.asarray(surface, dtype=np.float32)
        if surface.ndim != 2 or surface.shape[1] < 3:
            raise ValueError(
                f'Surface samples for "{instance_id}" must have shape (N, >=3), got {surface.shape}.'
            )
        if surface.shape[0] > self.surface_samples_per_scene:
            rng = np.random.default_rng(self._instance_seed_(instance_id, self.seed))
            sample_idx = rng.permutation(surface.shape[0])[: self.surface_samples_per_scene]
            surface = surface[sample_idx]
        surface = surface[:, :3]
        return surface

    def _load_input_pose_(self, instance_id: str) -> np.ndarray:
        pose_dir = self.data_root / self.pose_param_dir / instance_id
        quaternions = np.load(pose_dir / "quaternions.npy").astype(np.float32)
        translations = np.load(pose_dir / "translations.npy").astype(np.float32)
        scales = np.load(pose_dir / "scales.npy").astype(np.float32)
        quaternions = np.nan_to_num(quaternions, nan=0.0, posinf=0.0, neginf=0.0)
        translations = np.nan_to_num(translations, nan=0.0, posinf=0.0, neginf=0.0)
        scales = np.nan_to_num(scales, nan=1.0, posinf=1.0, neginf=1.0)
        quat_norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        valid_quat = quat_norm[..., 0] > 1e-8
        normalized_quat = np.zeros_like(quaternions, dtype=np.float32)
        normalized_quat[..., 0] = 1.0
        normalized_quat[valid_quat] = quaternions[valid_quat] / quat_norm[valid_quat, :]
        scales = np.clip(scales, 1e-3, None)
        return np.concatenate([normalized_quat, translations, scales], axis=-1)

    @torch.no_grad()
    def _infer_instance_latent_(self, instance_id: str) -> torch.Tensor:
        surface = self._load_surface_points_(instance_id)
        input_pose = self._load_input_pose_(instance_id)
        surface_tensor = torch.from_numpy(surface).float().unsqueeze(0).cuda()
        pose_tensor = torch.from_numpy(input_pose).float().unsqueeze(0).cuda()
        if self.deterministic_eval:
            instance_seed = self._instance_seed_(instance_id, self.seed)
            torch.manual_seed(instance_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(instance_seed)
        z_g, z_p = self.model.encode(
            surface_tensor,
            pose_tensor,
            sample=False if self.deterministic_eval else (not self.use_mean_latent),
            return_dist=False,
        )
        return self._pack_latent_state_(z_g.squeeze(0), z_p.squeeze(0))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="/mnt/d/Groupdata/GH/physiopt/PartSDF-main/experiments/chair_part_vae",
        help="Path to the Part-aware VAE experiment directory.",
    )
    parser.add_argument(
        "--partsdf_root",
        type=str,
        default="/mnt/d/Groupdata/GH/physiopt/PartSDF-main",
        help="Path to the PartSDF-main project root containing src/ and data/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test"],
        default="train",
        help="Dataset split used to choose the initial instance.",
    )
    parser.add_argument(
        "--shape_idx",
        type=int,
        default=0,
        help="Index of the shape inside the chosen split.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional explicit instance id to override --shape_idx.",
    )
    parser.add_argument(
        "--optimize_mode",
        type=str,
        default="both",
        choices=["global", "parts", "both"],
        help="Choose whether to optimize z_g, z_p, or both.",
    )
    parser.add_argument(
        "--roi_volume_fraction_target",
        type=float,
        default=0.86,
        help="ROI-relative volume target used by the inner GCMMA optimization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=319,
        help="Random seed used to make latent inference reproducible.",
    )
    parser.add_argument(
        "--deterministic_eval",
        action=BooleanOptionalAction,
        default=True,
        help="Use deterministic surface subsampling and mean-latent inference.",
    )
    parser.add_argument("--ui_config", type=str, default=None)
    args = parser.parse_args()

    if args.ui_config is not None:
        GLOBAL_UI_CONFIG = UiConfig.from_yaml_file(args.ui_config)

    ViewerPartAwareVae(args)
