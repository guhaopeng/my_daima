import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.


from argparse import ArgumentParser, Namespace
import time
from PIL import Image
import glob
import imageio

import torch
import numpy as np


import polyscope as ps
import polyscope.imgui as psim

from physiopt.vis.ui_utils import KEY_HANDLER, Thumbnail

def save_popup(label: str, path: str, save_label: str = "Save"):
    import polyscope.imgui as psim
    psim.PushID(label)
    clicked = psim.Button(save_label)
    psim.SameLine()
    changed, new_path = psim.InputText("##path", path)
    # Check if Enter is pressed while input is active
    try:
        if psim.IsItemFocused() and psim.IsKeyPressed(psim.ImGuiKey_Enter):
            clicked = True
    except AttributeError:
        pass
    psim.PopID()
    return clicked, new_path


from trellis.pipelines import TrellisTextTo3DPipeline, TrellisImageTo3DPipeline
from trellis.renderers.gaussian_render import GaussianRenderer
from trellis.utils import render_utils
from trellis.modules import sparse as sp
from trellis.modules.sparse.basic import save_slat_conds

MAX_DEPTH = 10.0


class Viewer:

    def __init__(self, args) -> None:

        self.diag = [1.0] * 4
        self.device = "cuda"
        self.image_mode = bool(args.image)
        self.seed = 42
        self.text = "a soft modern lamps"
        #"a chair that looks like an octopus but only has three legs"
        #"a chair that looks like an octopus" #avocado 
        #"An octopus-shaped chair with only three legs"
        self.thumbnail: Thumbnail = None
        self.save_path = "out/tmp"

        # Load a pipeline from a model folder or a Hugging Face model hub.
        if self.image_mode:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
        else:
            self.pipeline = TrellisTextTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-text-xlarge"
            )
        self.pipeline.cuda()

        self.outputs = None

        # -----------------------
        # Init renderer
        # -----------------------

        kwargs = {}
        self.renderer = GaussianRenderer()
        self.renderer.rendering_options.resolution = kwargs.get("resolution", 1080)
        self.renderer.rendering_options.near = kwargs.get("near", 0.01)
        self.renderer.rendering_options.far = kwargs.get("far", 1.6)
        self.renderer.rendering_options.bg_color = kwargs.get("bg_color", (1, 1, 1))
        self.renderer.rendering_options.ssaa = kwargs.get("ssaa", 1)
        self.renderer.pipe.kernel_size = kwargs.get("kernel_size", 0.1)
        self.renderer.pipe.use_mip_gaussian = True

        # -----------------------
        # Init polyscope
        # -----------------------

        ps.init()
        self.ps_init()

        # -----------------------
        # Start polyscope
        # -----------------------

        ps.set_user_callback(self.ps_callback)
        if hasattr(ps, "set_drop_callback"):
            ps.set_drop_callback(self.ps_drop_callback)
        else:
            print("[WARNING] Polyscope 'set_drop_callback' not found. Drag-and-drop will be disabled.")
            print("Please ensure you have the correct version of Polyscope installed (via ps-py-plus).")
            print("You can still use the 'Image Path' input in the GUI to load images manually.")
        ps.show()

    def ps_init(self) -> None:
        """
        Initialize Polyscope
        """
        ps.set_ground_plane_mode("none")
        ps.set_max_fps(120)
        ps.set_window_size(1080, 1080)
        # Anti-aliasing
        ps.set_SSAA_factor(4)
        # Uncomment to prevent polyscope from changing scales (including Gizmo!)
        # ps.set_automatically_compute_scene_extents(False)
        ps.set_up_dir("z_up")

        # ps_plane = ps.add_scene_slice_plane()
        # ps_plane.set_draw_plane(False)
        # ps_plane.set_draw_widget(True)

        self.update_render_sizes()
        self.init_render_buffer()

        self.last_time = time.time()

    def init_render_buffer(self):
        # print(
        #     f"Initialized render_buffer with shape: {(self.buffer_size[1], self.buffer_size[0], 4)}"
        # )
        self.render_buffer_quantity = ps.add_raw_color_alpha_render_image_quantity(
            "render_buffer",
            MAX_DEPTH
            * np.ones((self.buffer_size[1], self.buffer_size[0]), dtype=float),
            np.ones((self.buffer_size[1], self.buffer_size[0], 4), dtype=float),
            enabled=True,
            allow_fullscreen_compositing=True,
        )

        self.render_buffer = ps.get_quantity_buffer("render_buffer", "colors")
        self.render_buffer_depth = ps.get_quantity_buffer("render_buffer", "depths")

    def update_render_sizes(self):
        self.window_size = ps.get_window_size()
        self.buffer_size = (
            int(self.window_size[0]),
            int(self.window_size[1]),
        )

    # `ps_callback` is called every frame by polyscope
    def ps_callback(self) -> None:

        # Update fps count
        new_time = time.time()
        self.fps = 1.0 / (new_time - self.last_time)
        self.last_time = new_time

        # I usually put all my guy stuff in another function
        self.gui()

        # I usually draw things in a draw function (e.g., rendering buffer)
        self.draw()

    def gui(self) -> None:
        psim.Text(f"fps: {self.fps:.4f};")

        psim.BeginDisabled(self.image_mode and self.thumbnail is None)
        if psim.Button("Sample"):
            self.sample()
        psim.EndDisabled()

        if self.outputs is not None:
            psim.SameLine()
            clicked, self.save_path = save_popup("save", self.save_path)
            if clicked:
                self.save()

        _, self.seed = psim.InputInt("seed", self.seed, step=1)

        if self.image_mode:
            # Manual image loading fallback
            psim.Separator()
            psim.Text("Load Image (Fallback)")
            changed, self.manual_image_path = psim.InputText("##path_input", getattr(self, "manual_image_path", ""))
            psim.SameLine()
            if psim.Button("Load"):
                if os.path.exists(self.manual_image_path):
                    self.ps_drop_callback(self.manual_image_path)
                else:
                    print(f"File not found: {self.manual_image_path}")
            psim.Separator()

            if self.thumbnail is not None:
                self.thumbnail.gui()
        else:
            clicked, self.text = psim.InputText("text", self.text)

    # Always disable gradients!
    @torch.no_grad()
    def draw(self) -> None:

        # Handle window resize
        if ps.get_window_size() != self.window_size:
            self.update_render_sizes()
            self.init_render_buffer()

        if self.outputs is None:
            return

        cam_params = ps.get_view_camera_parameters()
        # fov = torch.tensor(cam_params.get_fov_vertical_deg())
        # intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov).to(self.device)

        # V_inv = np.linalg.inv(cam_params.get_view_mat())
        # camera_position = V_inv[:3, 3]  # Extract the translation part

        # extrinsics = utils3d.torch.extrinsics_look_at(
        #     torch.tensor(camera_position).float().cuda(),
        #     torch.tensor(cam_params.get_look_dir()).float().cuda(),
        #     torch.tensor(cam_params.get_up_dir()).float().cuda(),
        # )
        # rgb = self.renderer.render(self.display_splats, extrinsics, intrinsics)[
        #     "color"
        # ].permute((1, 2, 0))

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
            self.outputs["gaussian"][0],
            cam_params.get_fov_vertical_deg(),
            self.window_size[0],
            self.window_size[1],
            view_mat,
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

        rendered_image = rendered_image.cpu()
        self.render_buffer.update_data(rendered_image.reshape(-1, 4).cpu().numpy())

    @torch.no_grad()
    def sample(self):
        self.outputs = self.pipeline.run(
            self.thumbnail.image if self.image_mode else self.text,
            seed=self.seed,
            # Optional parameters
            # sparse_structure_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            # slat_sampler_params={
            #     "steps": 12,
            #     "cfg_strength": 7.5,
            # },
            formats=["gaussian", "slat"],
            num_samples=1,
        )
                # Calculate and print voxel count for 32 resolution
        if "slat" in self.outputs and self.outputs["slat"] is not None:
            slat = self.outputs["slat"][0] if isinstance(self.outputs["slat"],list) else self.outputs["slat"]
            try:
                # We need the mesh_decoder to decode the slat to coarse occ
                # It's usually part of the pipeline's model components
                if hasattr(self.pipeline, 'models') and 'slat_decoder_mesh' in self.pipeline.models:
                    mesh_decoder = self.pipeline.models['slat_decoder_mesh']
                    # Use resolution 32, which means division factor is 256 // 32 = 8
                    coarse_coords, coarse_occ, _, _ = mesh_decoder.to_coarse_occ(slat, 8)[0]
                    # Filter based on same logic as extract_solid / optimization
                    valid_voxels = (coarse_occ > 1e-3).sum().item()
                    slat_elements = slat.shape[0] if hasattr(slat, 'shape') else (slat.coords.shape[0] if hasattr(slat, 'coords') else "Unknown")
                    print(f"\n======================================")
                    print(f"Generated 3D Model Info:")
                    print(f" - SLAT elements: {slat_elements}")
                    print(f" - Estimated voxels (at res=32): {valid_voxels}")
                    print(f"======================================\n")
            except Exception as e:
                print(f"Could not calculate voxel count: {e}")


    @torch.no_grad()
    def save(self):
        os.makedirs(self.save_path, exist_ok=True)

        if self.image_mode:
            self.thumbnail.image.save(os.path.join(self.save_path, "image.png"))
            with open(os.path.join(self.save_path, f"info.txt"), "w") as f:
                f.write(f"Seed: {self.seed}\n")
        else:
            ps.screenshot(os.path.join(self.save_path, f"screenshot_{0:02d}.png"))
            with open(os.path.join(self.save_path, f"info.txt"), "w") as f:
                f.write(f"Prompt: {self.text}\n")
                f.write(f"Seed: {self.seed}\n")

        # ==============
        # Gaussians
        # ==============

        video = render_utils.render_video(self.outputs["gaussian"][0])["color"]
        imageio.mimsave(
            os.path.join(self.save_path, f"sample_gs_{0:02d}.mp4"),
            video,
            fps=30,
        )

        self.outputs["gaussian"][0].save_ply(
            os.path.join(self.save_path, f"sample_{0:02d}.ply")
        )

        # ==============
        # Slats
        # ==============

        slat: sp.SparseTensor = self.outputs["slat"][0]
        cond: sp.SparseTensor = self.outputs["cond"]
        neg_cond: sp.SparseTensor = self.outputs["neg_cond"]
        save_slat_conds(
            os.path.join(self.save_path, f"slat_{0:02d}.pt"),
            slat,
            cond,
            neg_cond,
            self.outputs["z_s"] if "z_s" in self.outputs else None,
        )

    def ps_drop_callback(self, input: str) -> None:

        if not self.image_mode:
            raise ValueError("Drag-n-drop is only enabled for images!")

        extension = os.path.splitext(input)[1]
        if extension in {".png", ".jpg", ".jpeg"}:
            try:
                self.thumbnail = Thumbnail.from_path(input)
                self.sample()
            except Exception as e:
                # handle the exception
                print("Could not import:", input)
                print("Error:\n", e)
        else:
            print("Only image files are accepted!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()

    if args.image and args.text:
        raise ValueError("Only one mode allowed!")

    if not args.image and not args.text:
        raise ValueError("One mode required!")

    Viewer(args)
