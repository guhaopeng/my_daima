import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.


from argparse import ArgumentParser

import torch
import numpy as np
import imageio
from PIL import Image

import polyscope as ps
import polyscope.imgui as psim

from ps_utils.ui import save_popup, get_next_save_factory

from tqdm import tqdm

from trellis.pipelines import load_submodel
from trellis.renderers.gaussian_render import GaussianRenderer
from trellis.models.structured_latent_vae import SLatMeshDecoder
from trellis.models.structured_latent_flow import SLatFlowModel
from trellis.modules import sparse as sp
from trellis.modules.sparse.basic import SlatPayload
from trellis.utils import render_utils
from trellis.modules.sparse.basic import save_slat_conds
from trellis.pipelines import samplers
from trellis.models.model_data import MODEL_NAMES, SLAT_INFO
from trellis.representations.mesh import MeshExtractResult
from trellis.utils.postprocessing_utils import to_glb

from physiopt.vis.ui_config import UiConfig, GLOBAL_UI_CONFIG
from physiopt.vis.viewer_base import ViewerBase
from physiopt.utils.phys_utils import generate_hex_at_coords

from physiopt.opt.optimizer_state import OptimizerConfig, OptimizationState
from physiopt.opt.optimizer_utils import update_tensors
from physiopt.utils.cube_processing import add_voxels_with_sdf

SLAT_SAVE_FOLDER = "results/slat"
GAUSSIAN_SAVE_FOLDER = "results/gaussians"
TEXTURED_MESH_SAVE_FOLDER = "results/textured_mesh"
SPLATS_RENDER_FOLDER = "results/splats_render"


class ViewerTrellis(ViewerBase):

    # ====================================
    # REQUIRED
    # ====================================

    def init_network(self, args) -> OptimizerConfig:
        """
        Initialize everything you need (e.g., network, renderer, etc)
        Returns an optimizer config
        """
        # 0. Additional save locations
        self._get_next_slat_path = get_next_save_factory(SLAT_SAVE_FOLDER, None)
        self._get_next_gaussian_path = get_next_save_factory(
            GAUSSIAN_SAVE_FOLDER, "ply"
        )
        self._get_next_textured_mesh_path = get_next_save_factory(
            TEXTURED_MESH_SAVE_FOLDER, "glb"
        )
        self._get_splats_render_path = get_next_save_factory(SPLATS_RENDER_FOLDER, None)
        self.slat_save_path = self._get_next_slat_path()
        self.gaussian_save_path = self._get_next_gaussian_path()
        self.texture_mesh_save_path = self._get_next_textured_mesh_path()
        self.splats_render_path = self._get_splats_render_path()

        # 1. Network
        self.gaussian_decoder = load_submodel(*MODEL_NAMES["gaussian_decoder"]).to(
            "cuda"
        )
        self.mesh_decoder: SLatMeshDecoder = load_submodel(
            *MODEL_NAMES["mesh_decoder"]
        ).to("cuda")
        self.slat_sampler: samplers.FlowEulerGuidanceIntervalSampler = getattr(
            samplers, SLAT_INFO["slat_sampler"]["name"]
        )(**SLAT_INFO["slat_sampler"]["args"])

        self.init_slat_payload = SlatPayload.from_path(args.input)

        self.post_update_cond()

        # 2. Renderer
        kwargs = {}
        self.renderer = GaussianRenderer()
        self.renderer.rendering_options.resolution = kwargs.get("resolution", 1080)
        self.renderer.rendering_options.near = kwargs.get("near", 0.01)
        self.renderer.rendering_options.far = kwargs.get("far", 1.6)
        self.renderer.rendering_options.bg_color = kwargs.get("bg_color", (1, 1, 1))
        self.renderer.rendering_options.ssaa = kwargs.get("ssaa", 1)
        self.renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
        self.renderer.pipe.use_mip_gaussian = True

        # 3. Config
        init_config = OptimizerConfig()
        return init_config

    def post_update_cond(self) -> None:
        """
        Called right after cond is updated to ensure the proper model is current loaded
        """
        new_image_mode = self.init_slat_payload.image_mode
        image_mode_changed = (
            not hasattr(self, "image_mode") or new_image_mode != self.image_mode
        )
        self.image_mode = new_image_mode
        if image_mode_changed:
            self.load_flow_model()

    def load_flow_model(self) -> None:
        print("######################################")
        print(f"Loading flow model: {'image_mode' if self.image_mode else 'text_mode'}")
        print("######################################")
        if hasattr(self, "flow_model"):
            # First delete to release VRAM!
            del self.flow_model
        self.flow_model: SLatFlowModel = load_submodel(
            *MODEL_NAMES[
                (
                    "slat_flow_model_text"
                    if not self.image_mode
                    else "slat_flow_model_image"
                )
            ],
        ).to("cuda")

    def extract_solid(self) -> None:
        """
        Extracts a solid from nodes. Make sure that the following are properly updated after this step!
        `self.coarse_coords, self.coarse_occ, self.fine_coords, self.fine_sdf, self.nodes, self.elements`
        """
        self.coarse_coords, self.coarse_occ, self.fine_coords, self.fine_sdf = (
            self.mesh_decoder.to_coarse_occ(
                self.slat, 256 // self.config.res, beta=self.config.occ_sdf_beta
            )[0]
        )

        self.nodes, self.elements = generate_hex_at_coords(
            self.coarse_coords, self.config.res
        )

    def init_optimizer(self):
        self.slat.feats.requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.slat.feats], self.config.lr)

        # Disable gradients on the decoder
        for param in self.mesh_decoder.parameters():
            param.requires_grad = False

        self.current_trajectory.cond_payload.cond = self.init_slat_payload.cond.clone()
        self.current_trajectory.cond_payload.neg_cond = (
            self.init_slat_payload.neg_cond.clone()
        )
        self.current_trajectory.cond_payload.z_s = self.init_slat_payload.z_s.clone()

    def backward_step(self):
        """
        Called after a solve and using the per-voxel density sensitivities.
        """

        def f(feats: torch.Tensor):

            tmp_slat = sp.SparseTensor(feats, self.slat.coords)

            _, occ, _, _ = self.mesh_decoder.to_coarse_occ(
                tmp_slat, 256 // self.config.res
            )[0]

            if self.config.alpha_min > 0.0:
                occ = torch.clip(occ, self.config.alpha_min)

            return occ

        grad_latent = torch.autograd.functional.vjp(
            f,
            self.slat.feats.clone().requires_grad_(True),
            self.sensitivity_density,
        )[1]

        # Set manually the gradient (will be seen by the optimizer i.e., Adam)
        self.slat.feats.grad = grad_latent

    def post_solve_step(self):

        if self.config.inpainting_interval > 0:
            # Always inpaint at the end!
            if self.i_step >= self.config.num_iters:
                self.inpaint()
            elif self.i_step % self.config.inpainting_interval == 0 and self.i_step > 0:
                self.inpaint()

    @property
    def mc_needed(self) -> bool:
        """
        Specifies whether MC is needed after each optimization step
        """
        return False

    def set_replay_state(self, opt_state: OptimizationState) -> bool:
        if opt_state.slat is None:
            return False
        self.slat = opt_state.slat
        return True

    # ====================================
    # OPTIONAL
    # ====================================

    @torch.no_grad()
    def shape_selection_gui(self) -> bool:
        """
        Additional latent/shape control. Return true if trajectory needs to be updated
        """
        # =========================
        # Export
        # =========================

        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            if psim.TreeNode("Export:"):

                # 1. Save Slat
                clicked, self.slat_save_path = save_popup(
                    "save slat", self.slat_save_path, save_label="Save Slat"
                )
                if clicked:
                    try:
                        self.save_slat()
                        self.slat_save_path = self._get_next_slat_path()
                    except Exception as e:
                        print(f"Failed to save slat: {e}")

                psim.SameLine()

                # 2. Save Gaussian
                clicked, self.gaussian_save_path = save_popup(
                    "save gaussian", self.gaussian_save_path, save_label="Save Gaussian"
                )
                if clicked and self.current_state.splats is not None:
                    try:
                        self.current_state.splats.save_ply(self.gaussian_save_path)
                        self.gaussian_save_path = self._get_next_gaussian_path()
                    except Exception as e:
                        print(f"Failed to save gaussians: {e}")

                psim.SameLine()

                # 3. Save Textured Mesh
                clicked, self.texture_mesh_save_path = save_popup(
                    "save textured_mesh",
                    self.texture_mesh_save_path,
                    save_label="Save TexturedMesh",
                )
                if (
                    clicked
                    and self.current_state.splats is not None
                    and self.current_state.mesh_vertices is not None
                ):
                    try:
                        mesh = MeshExtractResult(
                            torch.from_numpy(self.current_state.mesh_vertices),
                            torch.from_numpy(self.current_state.mesh_faces),
                        )
                        glb = to_glb(
                            self.current_state.splats,
                            mesh,
                            # Optional parameters
                            simplify=0.95,  # Ratio of triangles to remove in the simplification process
                            texture_size=1024,  # Size of the texture used for the GLB
                            y_up=False,
                        )
                        glb.export(self.texture_mesh_save_path)
                        self.texture_mesh_save_path = (
                            self._get_next_textured_mesh_path()
                        )
                    except Exception as e:
                        print(f"Failed to save textured mesh: {e}")

                # 4. Save all splats render
                clicked, self.splats_render_path = save_popup(
                    "save splats render",
                    self.splats_render_path,
                    save_label="Save Splats Render",
                )
                if clicked:
                    os.makedirs(self.splats_render_path, exist_ok=True)
                    for i_step in tqdm(range(self.current_trajectory.size)):
                        self.set_i_step(i_step)
                        self.show_current()

                        rendered_image = self._render_current()
                        img = Image.fromarray(
                            (
                                rendered_image.cpu().numpy().clip(0.0, 1.0) * 255.0
                            ).astype(np.uint8)
                        )
                        img.save(
                            os.path.join(
                                self.splats_render_path, f"render_{i_step:04d}.png"
                            )
                        )
                    self.splats_render_path = self._get_splats_render_path()

                psim.TreePop()

        # =========================
        # Inpainting
        # =========================

        psim.Separator()

        if psim.TreeNode("Inpainting:"):

            psim.BeginDisabled(self.current_trajectory.size > 1)

            clicked, self.config.inpainting_interval = psim.SliderInt(
                "inpainting_interval",
                self.config.inpainting_interval,
                v_min=0,
                v_max=40,
            )
            clicked, self.config.inpainting_critical_ratio = psim.SliderFloat(
                "inpainting_critical_ratio",
                self.config.inpainting_critical_ratio,
                v_min=0.0,
                v_max=0.2,
            )
            clicked, self.config.inpainting_reset_adam = psim.Checkbox(
                "inpainting_reset_adam", self.config.inpainting_reset_adam
            )

            psim.EndDisabled()

            psim.TreePop()

        return False

    def additional_ps_drop_callback(self, input: str, extension: str) -> None:
        """
        Handle any other files... (e.g., pt with TRELLIS)
        """
        if extension == ".pt":
            try:
                self.init_slat_payload = SlatPayload.from_path(input)
                self.post_update_cond()
                self.init_trajectory(
                    config=self.config,
                    keep_latent=False,
                )
            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
            return True
        else:
            return False

    def _render_current(self):
        cam_params = ps.get_view_camera_parameters()

        def fix_view_matrix(view_matrix: torch.Tensor, diag) -> torch.Tensor:
            """
            Fix the camera view by flipping the z-axis in the view matrix.

            Parameters:
                view_matrix (torch.Tensor): A 4x4 tensor representing the original view matrix.

            Returns:
                torch.Tensor: The modified view matrix with the z-axis flipped.
            """
            # Create a diagonal matrix that flips the z axis. The diagonal values are:
            # [1.0, 1.0, -1.0, 1.0]
            flip_z = torch.diag(
                torch.tensor(
                    diag,
                    dtype=view_matrix.dtype,
                    device=view_matrix.device,
                )
            )

            # Multiply the view matrix by the flip_z matrix.
            # The multiplication order matters; multiplying on the right adjusts the coordinates.
            fixed_view_matrix = view_matrix @ flip_z
            return fixed_view_matrix

        view_mat = fix_view_matrix(
            torch.from_numpy(cam_params.get_view_mat()).float().cuda(), self.diag
        )

        rgb = self.renderer.render_ps(
            self.display_splats,
            cam_params.get_fov_vertical_deg(),
            self.window_size[0],
            self.window_size[1],
            view_mat,
            mean_offset=torch.tensor(GLOBAL_UI_CONFIG.pos_gaussians)[None, :]
            .float()
            .cuda(),
        )["color"].permute((1, 2, 0))

        rendered_image = torch.cat(
            [
                rgb,
                torch.ones(
                    (self.buffer_size[1], self.buffer_size[0], 1), device=self.device
                ),
            ],
            dim=-1,
        )

        return rendered_image

    # Always disable gradients!
    @torch.no_grad()
    def draw(self) -> None:

        # Handle window resize
        if ps.get_window_size() != self.window_size:
            self.update_render_sizes()
            self.init_render_buffer()

        if GLOBAL_UI_CONFIG.display_gaussians:

            rendered_image = self._render_current()

            self.render_buffer.update_data_from_device(rendered_image)
        # if GLOBAL_STATE.use_depth:
        #     self.render_buffer_depth.update_data_from_device(
        #         render_pkg["depth"].squeeze(-1)
        #     )
        # else:
        #     self.render_buffer_depth.update_data_from_device(
        #         MAX_DEPTH
        #         * torch.ones(
        #             (rendered_image.shape[0], rendered_image.shape[1]),
        #             device="cuda",
        #         )
        #     )

    def pre_init_trajectory(
        self, keep_latent: bool = True, replace_current: bool = False
    ) -> None:
        """
        Called before the trakectory is created in case something is needed!
        e.g., tracking latents
        """
        if keep_latent and not self.current_state.slat is None:
            self.slat = sp.SparseTensor(
                self.current_state.slat.feats.detach().clone(),
                self.current_state.slat.coords.detach().clone(),
            )
        else:
            self.slat = sp.SparseTensor(
                self.init_slat_payload.slat.feats.clone(),
                self.init_slat_payload.slat.coords.clone(),
            )

        # Reset full_to_init_map
        # self.full_to_init_map = None

    @property
    def up_dir(self) -> str:
        return "z_up"

    # =========================
    # TRELLIS specifics
    # =========================

    @torch.no_grad()
    def inpaint(self):
        # ==============
        # INPAINT
        # NB: enable pruning, right now we're just bulking!
        # ==============

        # 1. Compute which voxels to add when their SDF becomes close to the ratio mask
        self.to_add_cubes, self.to_add_cubes_slat, self.ratio_mask = (
            add_voxels_with_sdf(
                self.fine_coords,
                self.fine_sdf,
                critical_ratio=self.config.inpainting_critical_ratio,
            )
        )

        # 2. Update SparseTensor and inpaint
        n_old_tensors = self.slat.feats.shape[0]
        n_new_tensors = self.to_add_cubes.shape[0]

        # ps_voxels.add_scalar_quantity(
        #     "sdf",
        #     # self.fine_sdf.flatten().cpu().numpy(),
        #     self.ratio_mask.float().flatten().cpu().numpy(),
        #     defined_on="vertices",
        #     enabled=True,
        # )

        def pad_with_zeros(coords: torch.Tensor):
            return torch.cat(
                [
                    torch.zeros(
                        coords.shape[0], 1, dtype=coords.dtype, device=coords.device
                    ),
                    coords,
                ],
                dim=1,
            )

        all_coords = torch.cat(
            [self.slat.coords, pad_with_zeros(self.to_add_cubes_slat)], dim=0
        )
        # WARNING: we take renormalized feats here!
        std = torch.tensor(SLAT_INFO["slat_normalization"]["std"])[None].to(
            self.slat.device
        )
        mean = torch.tensor(SLAT_INFO["slat_normalization"]["mean"])[None].to(
            self.slat.device
        )
        all_feats = torch.cat(
            [
                (self.slat.feats - mean) / std,
                torch.zeros(
                    self.to_add_cubes_slat.shape[0],
                    self.slat.feats.shape[1],
                    device=self.slat.feats.device,
                    dtype=self.slat.feats.dtype,
                ),
            ]
        )
        inpainting_mask = torch.zeros(
            all_coords.shape[0], dtype=torch.bool, device=all_coords.device
        )
        inpainting_mask[: self.slat.coords.shape[0]] = True
        inpainting_mask = inpainting_mask[:, None]

        noise = sp.SparseTensor(
            feats=torch.randn(all_coords.shape[0], self.flow_model.in_channels).to(
                self.device
            ),
            coords=all_coords,
        )
        inpainted_slat = self.slat_sampler.inpaint(
            self.flow_model,
            noise,
            x_0_ref=sp.SparseTensor(all_feats, all_coords),
            mask=inpainting_mask,
            cond=self.current_trajectory.cond_payload.cond.to("cuda"),
            neg_cond=self.current_trajectory.cond_payload.neg_cond.to("cuda"),
            **SLAT_INFO["slat_sampler"]["params"],
            verbose=True,
        ).samples
        self.slat: sp.SparseTensor = inpainted_slat * std + mean

        # Update learnable parameters
        new_slat = update_tensors(
            self.optimizer,
            old_tensors=self.slat.feats[:n_old_tensors],
            new_tensors=self.slat.feats[n_old_tensors:],
            reset_avg=self.config.inpainting_reset_adam,
        )
        # NB: we need to recreate a SparseTensor otherwise, gradients aren't tracked properly anymore
        self.slat = sp.SparseTensor(new_slat, self.slat.coords)

        # 3. Record
        self.current_state.to_add_cubes = self.to_add_cubes.detach().cpu().numpy()
        self.current_state.to_add_cubes_slat = (
            self.to_add_cubes_slat.detach().cpu().numpy()
        )
        self.current_state.ratio_mask = self.ratio_mask.detach().cpu().numpy()

    def save_slat(self):
        os.makedirs(self.slat_save_path, exist_ok=True)

        # ==============
        # Gaussians
        # ==============

        video = render_utils.render_video(self.current_state.splats)["color"]
        imageio.mimsave(
            os.path.join(self.slat_save_path, f"sample_gs_{0:02d}.mp4"),
            video,
            fps=30,
        )

        self.current_state.splats.save_ply(
            os.path.join(self.slat_save_path, f"sample_{0:02d}.ply")
        )

        # ==============
        # Slats
        # ==============

        slat: sp.SparseTensor = self.current_state.slat
        cond: torch.Tensor = self.current_trajectory.cond_payload.cond
        neg_cond: torch.Tensor = self.current_trajectory.cond_payload.neg_cond
        z_s: torch.Tensor = self.current_trajectory.cond_payload.z_s
        save_slat_conds(
            os.path.join(self.slat_save_path, f"slat_{0:02d}.pt"),
            slat,
            cond,
            neg_cond,
            z_s,
        )


if __name__ == "__main__":  #
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--pipeline", type=str, default="JeffreyXiang/TRELLIS-image-large"
    )
    parser.add_argument("--ui_config", type=str, default=None)
    args = parser.parse_args()

    if args.ui_config is not None:
        GLOBAL_UI_CONFIG = UiConfig.from_yaml_file(args.ui_config)

    ViewerTrellis(args)
