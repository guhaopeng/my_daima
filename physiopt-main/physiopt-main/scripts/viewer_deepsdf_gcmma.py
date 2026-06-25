import os

from physiopt.vis.ui_config import UiConfig, GLOBAL_UI_CONFIG

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.


from argparse import ArgumentParser
import glob
import json

import torch
import polyscope.imgui as psim

from physiopt.vis.viewer_base_yuanshi import ViewerBase
from physiopt.utils.phys_utils import generate_hex
from physiopt.utils.grid_utils import filter_one_component
from physiopt.opt.forces import ForceConfig
from physiopt.opt.boundary import BoundaryCondType
from physiopt.opt.optimizer_state import OptimizerConfig, OptimizationState

from physiopt.models.deepsdf.deep_sdf_decoder import Decoder
import physiopt.models.deepsdf.dir_info as dinfo
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
GCMMA_MOVE = 0.001
GCMMA_BOUND = 3.0


class ViewerSdf(ViewerBase):

    def init_network(self, args) -> OptimizerConfig:

        # 1. Load specs
        specs_filename = os.path.join(args.exp, "specs.json")
        if not os.path.isfile(specs_filename):
            raise Exception(
                'The experiment directory does not include specifications file "specs.json"'
            )
        specs = json.load(open(specs_filename))
        latent_size = specs["CodeLength"]

        # 2. Load network
        decoder = Decoder(latent_size, **specs["NetworkSpecs"])
        decoder = torch.nn.DataParallel(decoder)
        decoder.eval()

        all_ckpts = glob.glob(
            os.path.join(args.exp, dinfo.model_params_subdir, "*.pth")
        )
        all_ckpts = [ckpt for ckpt in all_ckpts if not "latest" in ckpt]
        all_ckpts = sorted(
            all_ckpts, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        ckpt_path = all_ckpts[-1]
        epoch = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"Loading ckpt: {ckpt_path}")
        saved_model_state = torch.load(ckpt_path)
        decoder.load_state_dict(saved_model_state["model_state_dict"])
        self.model = decoder.module.cuda()

        # 3. Load latents
        latents_path = os.path.join(args.exp, dinfo.latent_codes_subdir, f"{epoch}.pth")
        self.latents = torch.load(latents_path)["latent_codes"]["weight"]

        self.current_latent_index: int = 0
        self.get_latent(self.current_latent_index)

        default_gravity_force = ForceConfig(external_force=[0.0, -500.0, 0.0])
        default_external_force = ForceConfig(external_force=[0.0, 0.0, 1000.0])

        config = OptimizerConfig(
            lr=1e-3,
            boundary_cond=BoundaryCondType.BOTTOM_Y,
            init_forces=[default_gravity_force, default_external_force],
            res=32,
        )
        config.latent_optimizer = "gcmma"
        config.volume_term = 0.0
        config.volume_fraction_target = 0.02
        self.target_volume_fraction = float(config.volume_fraction_target)
        return config

    @torch.no_grad()
    def _store_reference_latent_(self) -> None:
        self.initial_latent = self.latent.detach().clone().cuda()

    @torch.no_grad()
    def _clear_mma_reference_scalars_(self) -> None:
        self.initial_compliance_ref = None
        if (
            hasattr(self, "trajectory_handler")
            and self.trajectory_handler.current_idx in self.trajectory_handler.trajectories
        ):
            self.target_volume_fraction = float(self.config.volume_fraction_target)
        elif not hasattr(self, "target_volume_fraction"):
            self.target_volume_fraction = 0.02
        self.mma_kkt_norm = None
        self.mma_residual_max = None
        self.gcmma_innerit = None
        self.gcmma_conserv = None
        self.gcmma_raa0 = None

    def pre_init_trajectory(
        self, keep_latent: bool = True, replace_current: bool = False
    ) -> None:
        if keep_latent and self.current_state.latent is not None:
            self.latent = (
                self.current_state.latent.detach().clone().cuda().requires_grad_(True)
            )
            self._store_reference_latent_()
        else:
            self.get_latent(self.current_latent_index)
        self._clear_mma_reference_scalars_()

    def extract_solid(self):
        full_nodes, full_elements = generate_hex(self.config.res)
        model_input = self._query_to_input(full_nodes)
        sdf = self.model(model_input).flatten()

        full_occ_nodes = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
        self.full_occ = full_occ_nodes[full_elements].mean(1)
        self.full_nodes = full_nodes
        self.full_elements = full_elements

        sdf_filter_value = 1.0 / self.config.res
        nodewise_mask = sdf <= sdf_filter_value

        elementwise_mask = torch.any(nodewise_mask[full_elements], dim=1)
        if self.config.one_cc_only:
            elementwise_mask = filter_one_component(
                elementwise_mask, self.config.res - 1
            )

        selected_elements = full_elements[elementwise_mask]
        if selected_elements.numel() == 0:
            raise RuntimeError("DeepSDF extraction produced an empty solid.")
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
        if not HAS_MMA:
            raise ImportError("Cannot import `pytorch.mma.gcmmasub`.")
        self.latent.requires_grad_(True)
        self.optimizer = None
        self._store_reference_latent_()
        self._clear_mma_reference_scalars_()
        self._init_mma_state()

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
        xmin = -GCMMA_BOUND * torch.ones((self.mma_n, 1), dtype=dtype, device=device)
        xmax = GCMMA_BOUND * torch.ones((self.mma_n, 1), dtype=dtype, device=device)
        return xmin, xmax

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

        compliance = torch.sum(
            (self.coarse_occ ** self.config.p_exponent) * w_k
        )
        self.compliance_val = float(compliance.item())
        self.sensitivity_density_c = (
            -self.config.p_exponent
            * self.coarse_occ ** (self.config.p_exponent - 1)
            * w_k
        )
        self.volume_val = float(self.full_occ.mean().item())
        self.sensitivity_density_v = (
            torch.ones_like(self.full_occ) / max(int(self.full_occ.numel()), 1)
        )
        self.sensitivity_density = self.sensitivity_density_c

    def backward_step(self):

        def f_compliance(latent: torch.Tensor):
            model_input = self._query_to_input(self.nodes, latent=latent)
            sdf = self.model(model_input).flatten()
            occ = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
            occ = occ[self.elements].mean(1)
            if self.config.alpha_min > 0.0:
                occ = torch.clip(occ, self.config.alpha_min)
            return occ

        def f_volume(latent: torch.Tensor):
            model_input = self._query_to_input(self.full_nodes, latent=latent)
            sdf = self.model(model_input).flatten()
            occ_nodes = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)
            return occ_nodes[self.full_elements].mean(1)

        state_for_grad = self.latent.detach().clone().requires_grad_(True)
        grad_f0 = torch.autograd.functional.vjp(
            f_compliance, state_for_grad, self.sensitivity_density_c
        )[1]
        grad_f1 = torch.autograd.functional.vjp(
            f_volume, state_for_grad, self.sensitivity_density_v
        )[1]

        grad_f0 = torch.nan_to_num(grad_f0, nan=0.0)
        grad_f1 = torch.nan_to_num(grad_f1, nan=0.0)

        if self.initial_compliance_ref is None:
            self.initial_compliance_ref = max(float(self.compliance_val), 1e-12)
        compliance_ref = self.initial_compliance_ref

        self.latent.grad = None
        self.f0val_mma = torch.tensor(
            [[self.compliance_val / compliance_ref]],
            dtype=torch.float32,
            device=self.device,
        )
        self.fval_mma = torch.tensor(
            [[self.volume_val - self.target_volume_fraction]],
            dtype=torch.float32,
            device=self.device,
        )
        self.mma_df0dx = (grad_f0 / compliance_ref).view(self.mma_n, 1).to(torch.float64)
        self.mma_dfdx = grad_f1.view(1, -1).to(torch.float64)

    @torch.no_grad()
    def _evaluate_mma_candidate_(
        self, xmma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.latent = (
            xmma.view_as(self.latent).to(torch.float32).detach().clone().requires_grad_(True)
        )
        self._clear_simulation_state_()
        self.prepare_solid()
        self.u, self.f_int, self.f_ext, self.sigma, _, _ = self.solid.solve(max_iter=1)
        self._update_density_metrics_from_current_state_()

        compliance_ref = max(float(self.initial_compliance_ref), 1e-12)
        f0valnew = torch.tensor(
            [[self.compliance_val / compliance_ref]],
            dtype=torch.float64,
            device=self.device,
        )
        fvalnew = torch.tensor(
            [[self.volume_val - self.target_volume_fraction]],
            dtype=torch.float64,
            device=self.device,
        )
        return f0valnew, fvalnew

    def update_mma_step(self) -> None:
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
            asyinit=0.5,
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
                GCMMA_RAA0EPS,
                GCMMA_RAAEPS,
                epsimin,
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

        _, kktnorm, residumax = kktcheck(
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
            fvalnew,
            self.mma_dfdx,
            self.mma_a0,
            self.mma_a,
            self.mma_c,
            self.mma_d,
        )
        self.mma_kkt_norm = float(kktnorm)
        self.mma_residual_max = float(residumax)
        self.gcmma_innerit = innerit
        self.gcmma_conserv = int(conserv)
        self.gcmma_raa0 = float(self.mma_raa0)

        self.mma_xold2 = self.mma_xold1.clone()
        self.mma_xold1 = self.mma_xval.clone()
        self.mma_xval = xmma.clone()
        self.latent = (
            self.mma_xval.view_as(self.latent)
            .to(torch.float32)
            .detach()
            .clone()
            .requires_grad_(True)
        )

    def training_step(self):
        loss_dict = {}
        if self.i_step > self.config.num_iters:
            print("WARNING: this should never happen!")
            return False, {}

        if self.i_step > 0:
            self._clear_simulation_state_()

        self.micro_timer.reset()

        with torch.no_grad():
            self.micro_timer.start("time_prepare_solid")
            self.prepare_solid()
            self.micro_timer.stop("time_prepare_solid")

            self.micro_timer.start("time_solve")
            self.u, self.f_int, self.f_ext, self.sigma, _, _ = self.solid.solve(
                max_iter=1
            )
            self.micro_timer.stop("time_solve")

            if self.config.autorescale and self.i_step == 0:
                max_rescale_iteration = 10
                for i_rescale in range(max_rescale_iteration):
                    if (
                        torch.abs(self.u).mean()
                        <= self.config.autorescale_u_mean_target
                    ):
                        break
                    print(
                        f"Displacement is too big ({torch.abs(self.u).mean()})! Rescaling forces by {self.config.autorescale_u_factor} ({i_rescale}/{max_rescale_iteration})"
                    )
                    for force in self.config.forces:
                        force.magnitude *= self.config.autorescale_u_factor

                    self.prepare_solid()
                    self.u, self.f_int, self.f_ext, self.sigma, _, _ = self.solid.solve(
                        max_iter=1
                    )

            if (
                self.config.autorescale_u_mean_halt > 0.0
                and torch.abs(self.u).mean() > self.config.autorescale_u_mean_halt
            ):
                self.alert_handler.trigger("Displacement is too big!")
                return False, loss_dict

        self.micro_timer.start("time_backward")
        self._update_density_metrics_from_current_state_()

        loss_dict["compliance"] = self.compliance_val
        loss_dict["volume"] = self.volume_val
        loss_dict["volume_constraint"] = (
            self.volume_val - self.target_volume_fraction
        )
        loss_dict["displacement"] = torch.abs(self.u).mean().item()
        loss_dict["n_voxels"] = len(self.elements)
        loss_dict["total"] = self.compliance_val

        self.backward_step()

        self.micro_timer.stop("time_backward")
        self.micro_timer.start("time_post_solve")
        self.post_solve_step()
        self.micro_timer.stop("time_post_solve")

        loss_dict |= self.micro_timer.collect()

        if self.i_step >= self.config.num_iters:
            self.update_results()
            self.show_current()
            return False, loss_dict

        self.update_mma_step()
        self.update_results()
        self.show_current()
        self.trajectory_handler.current_trajectory.add(OptimizationState())
        return True, loss_dict

    def get_field(self, x: torch.Tensor):
        model_input = self._query_to_input(x, latent=self.latent)
        sdf = self.model(model_input).flatten()
        return sdf

    def field_to_occ(self, x: torch.Tensor) -> torch.Tensor:
        return occ_kernel(x, self.config.res, self.config.occ_sdf_beta)

    @property
    def mc_isovalue(self) -> float:
        return 0.0

    @property
    def mc_needed(self) -> bool:
        return True

    def set_replay_state(self, opt_state: OptimizationState) -> bool:
        if opt_state.latent is None:
            return False
        self.latent = (
            opt_state.latent.detach().clone().cuda().requires_grad_(True)
        )
        self._store_reference_latent_()
        self._clear_mma_reference_scalars_()
        self._init_mma_state()
        return True

    def shape_selection_gui(self) -> bool:
        clicked, self.current_latent_index = psim.InputInt(
            "shape_idx",
            self.current_latent_index,
            step=1,
        )
        self.current_latent_index = min(
            max(self.current_latent_index, 0), self.latents.shape[0] - 1
        )
        if clicked:
            self.get_latent(self.current_latent_index)
        psim.Text(f"target volume fraction = {self.target_volume_fraction:.4f}")
        if self.mma_kkt_norm is not None:
            psim.Text(f"gcmma kkt = {self.mma_kkt_norm:.3e}")
            psim.Text(f"gcmma residu max = {self.mma_residual_max:.3e}")
            psim.Text(f"gcmma innerit = {int(self.gcmma_innerit)}")
            psim.Text(f"gcmma conserv = {int(self.gcmma_conserv)}")
        return clicked

    @torch.no_grad()
    def get_latent(self, t: int = 0):
        self.current_latent_index = min(max(int(t), 0), self.latents.shape[0] - 1)
        self.latent = (
            self.latents[self.current_latent_index]
            .detach()
            .clone()
            .cuda()
            .requires_grad_(True)
        )
        self._store_reference_latent_()
        if (
            hasattr(self, "trajectory_handler")
            and self.trajectory_handler.current_idx in self.trajectory_handler.trajectories
        ):
            self._clear_mma_reference_scalars_()

    def _rescale_model_input(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    def _query_to_input(
        self, x: torch.Tensor, latent: torch.Tensor | None = None
    ) -> torch.Tensor:
        practical_latent = self.latent if latent is None else latent
        return torch.cat(
            [
                practical_latent[None, :].repeat(x.shape[0], 1),
                self._rescale_model_input(x),
            ],
            dim=1,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--ui_config", type=str, default=None)
    args = parser.parse_args()

    if args.ui_config is not None:
        GLOBAL_UI_CONFIG = UiConfig.from_yaml_file(args.ui_config)

    ViewerSdf(args)
