import os

from physiopt.vis.ui_config import UiConfig

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.


from argparse import ArgumentParser
import glob
import json

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim


from physiopt.vis.viewer_base import ViewerBase
from physiopt.utils.phys_utils import generate_hex
from physiopt.utils.grid_utils import filter_one_component
from physiopt.opt.forces import ForceConfig
from physiopt.opt.boundary import BoundaryCondType
from physiopt.opt.optimizer_state import OptimizerConfig, OptimizationState

from physiopt.models.deepsdf.deep_sdf_decoder import Decoder
import physiopt.models.deepsdf.dir_info as dinfo
from physiopt.opt.field import occ_kernel


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
        # Re-order
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

        DEFAULT_GRAVITY_FORCE = ForceConfig(external_force=[0.0, -500.0, 0.0])
        DEFAULT_EXTERNAL_FORCE = ForceConfig(external_force=[0.0, 0.0, 1000.0])

        # 4. Specific config
        return OptimizerConfig(
            lr=1e-3,
            # alpha_min=0.01,
            boundary_cond=BoundaryCondType.BOTTOM_Y,
            init_forces=[DEFAULT_GRAVITY_FORCE, DEFAULT_EXTERNAL_FORCE],
            res=32,
        )

    def extract_solid(self):
        full_nodes, full_elements = generate_hex(self.config.res)
        model_input = self._query_to_input(full_nodes)
        sdf = self.model(model_input).flatten()

        # Keep everythin below that value
        SDF_FILTER_VALUE = 1.0 / self.config.res
        nodewise_mask = sdf <= SDF_FILTER_VALUE

        elementwise_mask = torch.any(nodewise_mask[full_elements], dim=1)
        if self.config.one_cc_only:
            elementwise_mask = filter_one_component(
                elementwise_mask, self.config.res - 1
            )

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
        """
        Init the optimizer with whatever is needed!
        """
        self.latent.requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.latent], self.config.lr)

    def backward_step(self):

        def f(latent: torch.Tensor):

            model_input = self._query_to_input(self.nodes, latent=latent)
            sdf = self.model(model_input).flatten()
            occ = occ_kernel(sdf, self.config.res, self.config.occ_sdf_beta)

            occ = occ[self.elements].mean(1)

            # Threshold on element occupancy
            if self.config.alpha_min > 0.0:
                occ = torch.clip(occ, self.config.alpha_min)

            return occ

        grad_latent = torch.autograd.functional.vjp(
            f,
            self.latent.clone().requires_grad_(True),
            self.sensitivity_density,
        )[1]

        # Set manually the gradient (will be seen by the optimizer i.e., Adam)
        self.latent.grad = grad_latent

    def get_field(self, x: torch.Tensor):
        model_input = self._query_to_input(x, latent=self.latent)
        sdf = self.model(model_input).flatten()
        return sdf

    def field_to_occ(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts the field into occupancy
        """
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
        self.latent = opt_state.latent
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

        return clicked

    # =========================
    # DeepSDF specifics
    # =========================

    @torch.no_grad()
    def get_latent(self, t: int = 0):
        self.latent: torch.Tensor = self.latents[t].cuda()

    def _rescale_model_input(self, x: torch.Tensor) -> torch.Tensor:
        # DeepSDF samples from [-1, 1]
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
