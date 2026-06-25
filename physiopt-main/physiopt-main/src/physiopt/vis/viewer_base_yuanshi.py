import os
import sys
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

import time
from copy import deepcopy
import glob

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim
# from pytorch.Stress_3D_Sensitivity_Comp import Stress_3D_Sensitivity_Comp

# Monkey-patch polyscope.SurfaceMesh to include set_hover_callback if missing
# This is required because VoxelSet uses it, but it might be missing in some polyscope versions
if not hasattr(ps.SurfaceMesh, "set_hover_callback"):
    def set_hover_callback(self, callback):
        pass # No-op if not supported
    ps.SurfaceMesh.set_hover_callback = set_hover_callback

# Monkey-patch psim.GetIO to support KeysDown for legacy libs (like ps_utils)
try:
    # Check if KeysDown is missing
    _test_io = psim.GetIO()
    if not hasattr(_test_io, "KeysDown"):
        _original_GetIO = psim.GetIO
        
        class KeysDownProxy:
            def __getitem__(self, key):
                # Handle legacy int keys by converting to ImGuiKey if possible
                if isinstance(key, int):
                    # Try to map common keys or cast
                    # Note: Direct casting might work if the enum values match, 
                    # but ImGuiKey values are often high (>=512).
                    # If key is small (0-512), it might be a legacy key code.
                    # Polyscope/ImGui 1.87+ changed this.
                    # For safety, try to cast to ImGuiKey if valid, or ignore.
                    try:
                        return psim.IsKeyDown(psim.ImGuiKey(key))
                    except (ValueError, TypeError):
                        # Fallback for keys that don't map directly
                        return False
                return psim.IsKeyDown(key)
                
        class IOWrapper:
            def __init__(self, original_io):
                self.original_io = original_io
            
            def __getattr__(self, name):
                return getattr(self.original_io, name)
                
            @property
            def KeysDown(self):
                return KeysDownProxy()
                
        def GetIO_Patched():
            return IOWrapper(_original_GetIO())
            
        psim.GetIO = GetIO_Patched
except Exception as e:
    print(f"Warning: Failed to patch psim.GetIO: {e}")

from physiopt.vis.ui_utils import (
    state_button,
    exp_slider,
    scientific_slider,
    KEY_HANDLER,
    get_enum_maps,
    get_list_map,
    get_next_save_factory,
    parse_int_list,
    AlertHandler,
    choice_combo,
)

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

from physiopt.vis.voxel_set import VoxelSet

from torchfem.sparse import sparse_solve
from torchfem.materials import IsotropicElasticity3D
from torch_scatter import scatter
from tqdm import tqdm

from physiopt.vis.ui_config import GLOBAL_UI_CONFIG

from physiopt.utils.phys_utils import get_von_mises_stress
from physiopt.opt.solid_grid import SolidGrid
from physiopt.opt.boundary import BoundaryCondData, BoundaryCondType
from physiopt.opt.trajectory import TrajectoryHandler, Trajectory
from physiopt.opt.optimizer_state import OptimizerConfig, OptimizationState
from physiopt.utils.chamfer_distance import find_nearest_point
from physiopt.utils.grid_utils import grid_to_flat_idx, index_of_a_in_b
from physiopt.utils.timer import MicroTimer

MAX_DEPTH = 10.0


TRAJ_SAVE_FOLDER = "results/physics"
CAM_SAVE_FOLDER = "results/cam"
MESH_SAVE_FOLDER = "results/deformed_meshes"
VOXEL_SAVE_FOLDER = "results/voxels"
SLICES_SAVE_FOLDER = "results/slices"
ALL_RENDERS_FOLDER = "results/all_renders"

BOUNDARY_COND_MAP, BOUNDARY_COND_INVMAP, BOUNDARY_COND_NAMES, _ = get_enum_maps(
    BoundaryCondType
)
ALLOWED_OPT_RES = [16, 32, 64]
ALLOWED_OPT_RES_MAP, ALLOWED_OPT_RES_INVMAP = get_list_map(ALLOWED_OPT_RES)

ALLOWED_MC_RES = [32, 64, 96, 128, 256]
ALLOWED_MC_RES_MAP, ALLOWED_MC_RES_INVMAP = get_list_map(ALLOWED_MC_RES)

ALLOWED_UPSCALE_FACTORS = [2, 4, 8]
ALLOWED_UPSCALE_FACTORS_MAP, ALLOWED_UPSCALE_FACTORS_INVMAP = get_list_map(
    ALLOWED_UPSCALE_FACTORS
)

DEFORMATION_ALERT_MESSAGE = "Deformation is too big! Halting optimization..."


class ViewerBase:

    # ====================================
    # REQUIRED
    # ====================================

    def init_network(self, args) -> OptimizerConfig:
        """
        Initialize everything you need (e.g., network, renderer, etc)
        Returns an optimizer config
        """
        raise NotImplementedError()

    def extract_solid(self) -> None:
        """
        Extracts a solid from nodes. Make sure that the following are properly updated after this step!
        `self.coarse_coords, self.coarse_occ, self.fine_coords, self.fine_sdf, self.nodes, self.elements`
        """
        raise NotImplementedError()

    def init_optimizer(self) -> None:
        """
        Init the optimizer with whatever is needed!
        """
        raise NotImplementedError()

    def backward_step(self):
        """
        Called after a solve and using the per-voxel density sensitivities.
        """
        raise NotImplementedError()

    def post_solve_step(self):
        """
        Called after a backward pass (e.g., inpainting)
        """
        pass

    def get_field(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for mesh reconstruction if a field needs to be extracted: set mc_needed!
        NB: `x` is always assumed within [-1, 1]. Remap points accordingly!
        """
        raise NotImplementedError()

    def field_to_occ(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts the field into occupancy
        """
        raise NotImplementedError()

    @property
    def mc_isovalue(self) -> float:
        """
        The isovalue used for Marching Cubes extraction
        """
        raise NotImplementedError()

    @property
    def mc_needed(self) -> bool:
        """
        Specifies whether MC is needed after each optimization step
        """
        raise NotImplementedError()

    def set_replay_state(self, opt_state: OptimizationState) -> bool:
        """
        Given a state, initializes a replay state. Returns false if this wasn't possible.
        """
        raise NotImplementedError()

    # ====================================
    # OPTIONAL
    # ====================================

    def shape_selection_gui(self) -> bool:
        """
        Additional latent/shape control. Return true if trajectory needs to be updated
        """
        return False

    def additional_ps_drop_callback(self, input: str, extension: str) -> bool:
        """
        Handle any other files... (e.g., pt with TRELLIS)
        """
        return False

    # Always disable gradients!
    @torch.no_grad()
    def draw(self) -> None:
        """
        Draw anything in the render_buffer (e.g., Gaussian Splats)
        """
        pass

    def pre_init_trajectory(
        self, keep_latent: bool = True, replace_current: bool = False
    ) -> None:
        """
        Called before the trakectory is created in case something is needed!
        e.g., tracking latents
        """
        pass

    @property
    def up_dir(self) -> str:
        return "y_up"

    # ====================================
    # VARIABLES (used in simulation)
    # ====================================

    nodes: torch.Tensor
    elements: torch.Tensor
    solid: SolidGrid
    u: torch.Tensor
    f_int: torch.Tensor
    f_ext: torch.Tensor
    sigma: torch.Tensor
    coarse_coords: torch.Tensor
    coarse_occ: torch.Tensor

    # ====================================
    # ROUTINES
    # ====================================

    def __init__(self, args) -> None:

        self.device = "cuda"
        # Reset optimization
        self.optimizing = False
        self.diag = [1.0] * 4
        self.show_u_warning: bool = False
        self.mc_resolution = 96

        # Initialize save paths
        self._get_next_mesh_path = get_next_save_factory(MESH_SAVE_FOLDER, "npz")
        self._get_next_cam_path = get_next_save_factory(CAM_SAVE_FOLDER, "cam")
        self._get_next_save_path = get_next_save_factory(TRAJ_SAVE_FOLDER, "traj")
        self._get_next_voxels_path = get_next_save_factory(VOXEL_SAVE_FOLDER, "npz")
        self._get_next_all_renders_path = get_next_save_factory(
            ALL_RENDERS_FOLDER, None
        )
        self.traj_save_path = self._get_next_save_path()
        self.camera_path = self._get_next_cam_path()
        self.mesh_path = self._get_next_mesh_path()
        self.voxels_path = self._get_next_voxels_path()
        self.all_renders_path = self._get_next_all_renders_path()
        self.region_config_path = os.path.join(TRAJ_SAVE_FOLDER, "fixed_regions.json")

        # Initialize trajectory handler
        self.trajectory_handler = TrajectoryHandler()

        # Initialize micro timer
        self.micro_timer = MicroTimer()

        # Initialize variables for upscaling
        self.upscale_factor = 2
        self.replay_indices = "[]"

        # Initialize alert handler
        self.alert_handler = AlertHandler(background_color=(1.0, 0.0, 0.0, 1.0))

        # -----------------------
        # Init other variables
        # -----------------------

        self.u = None
        self.coarse_coords, self.coarse_occ, self.fine_coords, self.fine_sdf = (
            None,
            None,
            None,
            None,
        )
        self.slat = None
        self.voxel_set = None
        self.full_to_init_map = None
        self.fixed_force_coords = None
        self.fixed_support_coords = None
        self.region_config_locked = False

        # -----------------------
        # Init polyscope
        # -----------------------

        ps.init()
        self.ps_init()

        init_config = self.init_network(args)
        self.init_trajectory(init_config, keep_latent=False)

        # -----------------------
        # Start polyscope
        # -----------------------

        ps.set_user_callback(self.ps_callback)
        if hasattr(ps, "set_drop_callback"):
            ps.set_drop_callback(self.ps_drop_callback)
        ps.show()

    # =========================
    # NB: I'm lazy :D
    # =========================
    @property
    def config(self) -> OptimizerConfig:
        return self.trajectory_handler.current_trajectory.optimizer_config

    @property
    def i_step(self) -> int:
        return self.trajectory_handler.current_trajectory.i_step

    def set_i_step(self, i_step) -> None:
        self.trajectory_handler.current_trajectory.i_step = i_step
        self.trajectory_handler.current_trajectory.post_update_i_step()

    @property
    def current_state(self) -> OptimizationState:
        return self.trajectory_handler.current_trajectory.current_state

    @property
    def current_trajectory(self) -> Trajectory:
        return self.trajectory_handler.current_trajectory

    @property
    def losses(self):
        return self.trajectory_handler.current_trajectory.losses

    # =========================
    # INITIALIZERS
    # =========================

    def init_trajectory(
        self,
        config: OptimizerConfig,
        keep_latent: bool = True,  # Whether or not to keep latent
        replace_current: bool = False,  # Just to know whether to add new trajectories or not
    ):

        # Make sure to make a copy of the config!
        config = deepcopy(config)
        # Get forces from init_forces (because forces can change with auto_rescale)
        config.forces = deepcopy(config.init_forces)

        # =============================
        # Initialization stuff
        # =============================

        self.pre_init_trajectory(keep_latent, replace_current)
        
        # =============================
        # Create a new trajectory
        # =============================

        # Create the current state
        self.trajectory_handler.add(
            trajectory=Trajectory(OptimizationState(), config),
            replace_current=replace_current,
        )

        # Reset optimization
        self.optimizing = False

        # =============================
        # Optimizer
        # =============================

        self.init_optimizer()

        # =============================
        # Solid
        # =============================

        self._pre_optimize(keep_latent=keep_latent)

    @torch.no_grad()
    def _pre_optimize(self, keep_latent: bool = False):
        """Prior to optimization, we need to update the occupancy volume in case the user edited it in-between"""

        # Reset the full to init map, and the corresponding voxel set
        if (
            keep_latent
            and self.full_to_init_map is not None
            and self.voxel_set is not None
            and GLOBAL_UI_CONFIG.display_force_selection
        ):
            # NB: this assumes that there are no bugs and that we don't chnage
            # self.coarse_coords after the new extraction below
            selection_mask = None
            if self.fixed_force_coords is not None:
                selection_mask = self._build_fixed_force_selection_mask()
            elif self.voxel_set is not None:
                selection_mask = self.voxel_set.selection_mask[
                    self.full_to_init_map.cpu().numpy()
                ]
            self.voxel_set = VoxelSet(
                self.coarse_coords.cpu().numpy(),
                self.config.res,
                -0.5,
                0.5,
                offset=np.array(GLOBAL_UI_CONFIG.pos_force_selection),
                selection_mask=selection_mask,
            )
            self.full_to_init_map = torch.arange(self.coarse_coords.shape[0]).cuda()
        else:
            self.full_to_init_map = None
            self.voxel_set = None

        # Set the bottom manually here!
        self.prepare_solid(first_solid=True)
        self.init_volume = 0.02   #self.full_occ.mean().detach()
        if self.fixed_force_coords is not None:
            self._print_selected_force_voxels()
        if self.fixed_support_coords is not None:
            self._print_fixed_support_coords()

        # Show
        self.show_current()

    @torch.no_grad()
    def _update_forces(self):

        # 1. Global forces (e.g., gravity)
        self.solid.forces = (
            torch.from_numpy(self.config.forces[0].get_total_force()).float().cuda()
        )[None, :].repeat((self.solid.forces.shape[0], 1))

        # 2. Selected forces
        if len(self.config.forces) > 1:
            actual_selection_mask = self._get_force_selection_mask_for_current_coords_()
            if actual_selection_mask is None:
                actual_selection_mask = torch.zeros(
                    self.elements.shape[0], dtype=torch.int32, device="cuda"
                )

            force_mask = scatter(
                actual_selection_mask[:, None]
                .repeat(1, self.elements.shape[1])
                .flatten(),
                self.elements.flatten(),
                reduce="max",
            ).bool()
            self.solid.forces += force_mask[:, None] * (
                torch.from_numpy(self.config.forces[1].get_total_force()).float().cuda()
            )[None, :].repeat((self.solid.forces.shape[0], 1))

            self.current_state.force_mask = force_mask.detach().bool().cpu().numpy()

        # 3. Update state
        self.current_state.forces = self.solid.forces.detach().float().cpu().numpy()
    
    @torch.no_grad()
    def _capture_fixed_force_voxels(self):
        if self.voxel_set is None or self.coarse_coords is None:
            return

        selection_mask = np.asarray(self.voxel_set.selection_mask, dtype=bool)
        if selection_mask.size != self.coarse_coords.shape[0]:
            print(
                "[force selection] selection size does not match current coarse grid:"
                f" mask={selection_mask.size}, coords={self.coarse_coords.shape[0]}"
            )
            return

        selected_coords = self.coarse_coords[selection_mask]
        self.fixed_force_coords = (
            selected_coords.detach().cpu().numpy().astype(np.int32).copy()
        )

    @torch.no_grad()
    def _node_positions_to_grid_coords(self, nodes: torch.Tensor) -> np.ndarray:
        if nodes is None or nodes.numel() == 0:
            return np.zeros((0, 3), dtype=np.int32)

        coords = torch.round((nodes + 0.5) * float(self.config.res - 1)).to(torch.int32)
        return coords.detach().cpu().numpy()

    @torch.no_grad()
    def _capture_fixed_support_coords(self, support_mask: torch.Tensor | None = None):
        if self.nodes is None:
            return

        practical_mask = support_mask
        if practical_mask is None:
            if self.current_state.bottom is not None:
                practical_mask = torch.from_numpy(
                    self.current_state.bottom.astype(bool)
                ).to(self.nodes.device)
            elif self.boundary_cond_data is not None:
                practical_mask = self.boundary_cond_data.check(
                    self.nodes,
                    self.config.boundary_cond,
                    threshold=self.config.bottom_fixed_threshold,
                )

        if practical_mask is None:
            return

        support_nodes = self.nodes[practical_mask]
        self.fixed_support_coords = self._node_positions_to_grid_coords(support_nodes)

    @torch.no_grad()
    def _build_fixed_support_node_mask(self):
        if self.fixed_support_coords is None or self.nodes is None:
            return None

        support_coords = np.asarray(self.fixed_support_coords, dtype=np.int32).reshape(-1, 3)
        if support_coords.size == 0:
            return torch.zeros(self.nodes.shape[0], dtype=torch.bool, device=self.nodes.device)

        node_coords = self._node_positions_to_grid_coords(self.nodes)
        support_set = {tuple(v.tolist()) for v in support_coords}
        mask = np.array([tuple(v.tolist()) in support_set for v in node_coords], dtype=bool)
        return torch.from_numpy(mask).to(self.nodes.device)

    @torch.no_grad()
    def _print_fixed_support_coords(self):
        if self.fixed_support_coords is None:
            print("[support selection] no fixed support coords are stored yet.")
            return

        support_coords = np.asarray(self.fixed_support_coords, dtype=np.int32).reshape(-1, 3)
        print(f"[support selection] fixed {support_coords.shape[0]} support nodes:")
        print(support_coords.tolist())

    @torch.no_grad()
    def _save_region_config(self, file_path: str):
        if self.fixed_support_coords is None:
            self._capture_fixed_support_coords()
        if self.fixed_force_coords is None and self.voxel_set is not None:
            self._capture_fixed_force_voxels()

        payload = {
            "res": int(self.config.res),
            "boundary_cond": self.config.boundary_cond.value,
            "bottom_fixed_threshold": float(self.config.bottom_fixed_threshold),
            "force_coords": []
            if self.fixed_force_coords is None
            else np.asarray(self.fixed_force_coords, dtype=np.int32).reshape(-1, 3).tolist(),
            "support_coords": []
            if self.fixed_support_coords is None
            else np.asarray(self.fixed_support_coords, dtype=np.int32).reshape(-1, 3).tolist(),
        }

        parent = os.path.dirname(file_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[region io] saved fixed regions to: {file_path}")
        print(
            "[region io] counts:"
            f" force={len(payload['force_coords'])}, support={len(payload['support_coords'])}"
        )

    @torch.no_grad()
    def _load_region_config(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        force_coords = np.asarray(payload.get("force_coords", []), dtype=np.int32)
        support_coords = np.asarray(payload.get("support_coords", []), dtype=np.int32)

        self.fixed_force_coords = (
            None if force_coords.size == 0 else force_coords.reshape(-1, 3).copy()
        )
        self.fixed_support_coords = (
            None if support_coords.size == 0 else support_coords.reshape(-1, 3).copy()
        )
        self.region_config_locked = True

        print(f"[region io] loaded fixed regions from: {file_path}")
        if self.fixed_force_coords is not None:
            print(
                f"[region io] loaded {self.fixed_force_coords.shape[0]} fixed force voxels."
            )
        else:
            print("[region io] loaded 0 fixed force voxels.")
        if self.fixed_support_coords is not None:
            print(
                f"[region io] loaded {self.fixed_support_coords.shape[0]} fixed support nodes."
            )
        else:
            print("[region io] loaded 0 fixed support nodes.")

        self.prepare_solid(first_solid=False)
        self._print_selected_force_voxels()
        self._print_fixed_support_coords()
        self.show_current()

    @torch.no_grad()
    def _build_fixed_force_selection_mask(self):
        if self.fixed_force_coords is None or self.coarse_coords is None:
            return None

        current_coords = self.coarse_coords.detach().cpu().numpy().astype(np.int32)
        fixed_coords = np.asarray(self.fixed_force_coords, dtype=np.int32).reshape(-1, 3)
        if fixed_coords.size == 0:
            return np.zeros(current_coords.shape[0], dtype=bool)

        fixed_set = {tuple(v.tolist()) for v in fixed_coords}
        return np.array(
            [tuple(v.tolist()) in fixed_set for v in current_coords], dtype=bool
        )

    @torch.no_grad()
    def _get_force_selection_mask_for_current_coords_(self):
        if self.coarse_coords is None:
            return None

        selection_mask_np = None
        if self.fixed_force_coords is not None:
            selection_mask_np = self._build_fixed_force_selection_mask()
            if self.voxel_set is not None and selection_mask_np is not None:
                self.voxel_set.selection_mask = selection_mask_np.copy()
        elif self.voxel_set is not None:
            selection_mask_np = np.asarray(self.voxel_set.selection_mask, dtype=bool).reshape(-1)
            if selection_mask_np.size != self.coarse_coords.shape[0]:
                if self.full_to_init_map is not None:
                    map_np = self.full_to_init_map.detach().cpu().numpy().astype(np.int64)
                    if (
                        map_np.size == self.coarse_coords.shape[0]
                        and selection_mask_np.size > 0
                        and map_np.max(initial=-1) < selection_mask_np.size
                    ):
                        selection_mask_np = selection_mask_np[map_np]
                if selection_mask_np.size != self.coarse_coords.shape[0]:
                    print(
                        "[force selection] selection size does not match current coarse grid:"
                        f" mask={selection_mask_np.size}, coords={self.coarse_coords.shape[0]}"
                    )
                    selection_mask_np = np.zeros(self.coarse_coords.shape[0], dtype=bool)

        if selection_mask_np is None:
            return None

        if selection_mask_np.size != self.elements.shape[0]:
            print(
                "[force selection] selection size does not match current element grid:"
                f" mask={selection_mask_np.size}, elements={self.elements.shape[0]}"
            )
            selection_mask_np = np.zeros(self.elements.shape[0], dtype=bool)

        return torch.from_numpy(selection_mask_np.astype(np.int32)).cuda()

    @torch.no_grad()
    def _apply_fixed_force_region_selection(self):
        if (
            self.fixed_force_coords is None
            or self.voxel_set is None
            or self.coarse_coords is None
        ):
            return

        mask = self._build_fixed_force_selection_mask()
        self.voxel_set.selection_mask = mask
        self.voxel_set.update_selection_buffer()

    @torch.no_grad()
    def _print_selected_force_voxels(self):
        if self.voxel_set is None or self.coarse_coords is None:
            print("[force selection] no voxel set is available yet.")
            return

        selection_mask = np.asarray(self.voxel_set.selection_mask, dtype=bool)
        if selection_mask.size != self.coarse_coords.shape[0]:
            print(
                "[force selection] selection size does not match current coarse grid:"
                f" mask={selection_mask.size}, coords={self.coarse_coords.shape[0]}"
            )
            return

        selected_coords = self.coarse_coords[selection_mask]
        selected_coords_np = selected_coords.detach().cpu().numpy().astype(np.int32)
        if self.fixed_force_coords is not None:
            fixed_coords = np.asarray(self.fixed_force_coords, dtype=np.int32)
            print(f"[force selection] fixed {fixed_coords.shape[0]} coarse voxels:")
            print(fixed_coords.tolist())
            print(
                "[force selection] matched current "
                f"{selected_coords_np.shape[0]} coarse voxels:"
            )
            print(selected_coords_np.tolist())
        else:
            print(
                f"[force selection] selected {selected_coords_np.shape[0]} coarse voxels:"
            )
            print(selected_coords_np.tolist())

    def _post_solid_extraction(self):
        # 2. Update the full to init coords (i.e., mapping from new coords to init coords in order to splat forces)
        # NB: this is on coarse coords!
        # NB: this control flow only works if we only ADD.
        # WARNING: the following logic is tricky (s)
        if self.full_to_init_map is None:
            # The full to init map, keeps a map w.r.t. the initial set of voxels
            # to splat forces on newly added voxels (only via NN mapping)
            self.full_to_init_map = torch.arange(self.coarse_coords.shape[0]).cuda()
        elif (
            self.prev_coarse_coords is not None
            and self.prev_coarse_coords.shape[0] != self.coarse_coords.shape[0]
        ):
            flat_coarse_coords = grid_to_flat_idx(self.coarse_coords, self.config.res)
            flat_prev_coarse_coords = grid_to_flat_idx(
                self.prev_coarse_coords, self.config.res
            )

            # Check who's new
            old_mask = torch.isin(
                flat_coarse_coords,
                flat_prev_coarse_coords,
            )
            new_mask = ~old_mask
            # Find the indices to the old ones in the new ones and remap accordingly
            old_in_new_indices = index_of_a_in_b(
                flat_coarse_coords[old_mask], flat_prev_coarse_coords
            )
            new_full_to_init_map = torch.zeros(
                self.coarse_coords.shape[0], dtype=torch.long, device="cuda"
            )
            new_full_to_init_map[old_mask] = self.full_to_init_map[old_in_new_indices]
            # For the new ones, do NN
            closest_to_new = find_nearest_point(
                self.coarse_coords[new_mask],
                self.prev_coarse_coords,
            )
            # And read the previous mapping + propagate
            new_full_to_init_map[new_mask] = self.full_to_init_map[closest_to_new]
            # Update the map
            self.full_to_init_map = new_full_to_init_map

    @torch.no_grad()
    def prepare_solid(self, first_solid: bool = False) -> SolidGrid:

        self.prev_coarse_coords = (
            self.coarse_coords if self.coarse_coords is not None else None
        )
        self.extract_solid()
        self._post_solid_extraction()

        # After extracting, build the force volume
        if self.voxel_set is None and GLOBAL_UI_CONFIG.display_force_selection:
            self.voxel_set = VoxelSet(
                self.coarse_coords.cpu().numpy(),
                self.config.res,
                -0.5,
                0.5,
                offset=np.array(GLOBAL_UI_CONFIG.pos_force_selection),
            )
        self._apply_fixed_force_region_selection()

        assert torch.all(self.coarse_coords >= 0)
        assert torch.all(self.coarse_coords < self.config.res)

        # Threshold by alpha_min (name stays consistent with occflexi)
        if self.config.alpha_min > 0.0:
            self.coarse_occ = torch.clip(self.coarse_occ, self.config.alpha_min)

        # =====================
        # Materials
        # =====================

        vectorized_material = self.config.material.get_fem_material().vectorize(
            self.elements.shape[0]
        )

        vectorized_material.C *= (
            self.coarse_occ[:, None, None] ** self.config.p_exponent
        )
        vectorized_material.rho *= self.coarse_occ

        # =====================
        # Solid
        # =====================

        self.solid = SolidGrid(
            self.nodes,
            self.elements,
            vectorized_material,
        )

        # =====================
        # Boundary Conditions
        # =====================

        # NB: this is done differently from before
        if first_solid:
            self.boundary_cond_data = BoundaryCondData(self.nodes)

        if self.fixed_support_coords is not None:
            bottom = self._build_fixed_support_node_mask()
        else:
            bottom = self.boundary_cond_data.check(
                self.nodes,
                self.config.boundary_cond,
                threshold=self.config.bottom_fixed_threshold,
            )

        self.solid.constraints[bottom, :] = True

        # =====================
        # Forces
        # =====================

        self._update_forces()

        # =================
        # Record
        # In the same order as above
        # =================

        # Coordinates / Occ / Sdf
        self.current_state.coarse_coords = self.coarse_coords.detach().cpu().numpy()
        self.current_state.coarse_occ = self.coarse_occ.detach().cpu().numpy()

        # WARNING: this is extremely HARDCODED for TRELLIS, but only it should need it.
        if not self.mc_needed:
            self.current_state.slat = self.slat.cuda()
            self.current_state.fine_coords = self.fine_coords.detach().cpu().numpy()
            self.current_state.fine_sdf = self.fine_sdf.detach().cpu().numpy()

        # Before solve
        self.current_state.nodes = self.nodes.detach().cpu().numpy()
        self.current_state.elements = self.elements.detach().cpu().numpy()
        self.current_state.bottom = bottom.detach().float().cpu().numpy()
        assert isinstance(vectorized_material, IsotropicElasticity3D)
        self.current_state.E = vectorized_material.E.detach().float().cpu().numpy()
        self.current_state.nu = vectorized_material.nu.detach().float().cpu().numpy()
        self.current_state.rho = vectorized_material.rho.detach().float().cpu().numpy()
        self.current_state.forces = self.solid.forces.detach().float().cpu().numpy()

    @torch.no_grad()
    def update_current_mesh(self):
        import mcubes

        limit = np.array([-0.5] * 3), np.array([0.5] * 3)
        resolution = self.mc_resolution
        mins = limit[0]
        maxs = limit[1]
        # Prepare grid coords
        xs = np.linspace(mins[0], maxs[0], resolution)
        ys = np.linspace(mins[1], maxs[1], resolution)
        zs = np.linspace(mins[2], maxs[2], resolution)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)  # (R,R,R,3)
        pts = grid.reshape(-1, 3).astype(np.float32)

        # Evaluate SDF in chunks
        sdf_vals = []
        batch = 100_000
        for i in range(0, pts.shape[0], batch):
            chunk = torch.from_numpy(pts[i : i + batch]).to(self.device)
            with torch.no_grad():
                sdf_chunk = self.get_field(chunk).flatten().detach().cpu().numpy()
            sdf_vals.append(sdf_chunk)
        sdf = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)

        # Marching cubes (mcubes)
        verts, faces = mcubes.marching_cubes(sdf, self.mc_isovalue)
        scale = (maxs - mins) / (resolution - 1)  # = [1/(R-1),...]
        verts = verts * scale + mins
        self.current_state.mesh_vertices = verts.astype(np.float32)
        self.current_state.mesh_faces = faces

        # Also mak
        if self.current_state.u is not None and GLOBAL_UI_CONFIG.display_deformed_mesh:
            deformed_mesh = self.current_state.get_deformed_mesh(
                self.current_state.mesh_vertices,
                self.current_state.mesh_faces,
                config=self.config,
                mc_based=self.mc_needed,
            )
            self.current_state.deformed_vertices = deformed_mesh.deformed_vertices
            self.current_state.deformed_u = deformed_mesh.u

    @torch.no_grad()
    def show_current(self):

        if self.mc_needed:
            if self.current_state.mesh_vertices is None:
                self.update_current_mesh()

        # WARNING: this is extremely HARDCODED for TRELLIS, but only it should need it.
        else:
            if self.current_state.slat is not None:
                if self.current_state.splats is None:
                    # Decoder splats and meshes (cache!)
                    self.current_state.splats = self.gaussian_decoder(
                        self.current_state.slat.cuda()
                    )[0]

                if self.current_state.mesh_vertices is None:
                    mesh = self.mesh_decoder(self.current_state.slat.cuda())[0]
                    self.current_state.mesh_vertices = (
                        mesh.vertices.detach().cpu().numpy()
                    )
                    self.current_state.mesh_faces = mesh.faces.detach().cpu().numpy()

            # Get the current_state splats to display them
            self.display_splats = self.current_state.splats.to(self.device)

        # If DeformedMesh visualization is enabled, display
        if (
            self.current_state.deformed_vertices is None
            and GLOBAL_UI_CONFIG.display_deformed_mesh
            and self.current_state.mesh_vertices is not None
        ):
            deformed_mesh = self.current_state.get_deformed_mesh(
                self.current_state.mesh_vertices,
                self.current_state.mesh_faces,
                config=self.config,
                mc_based=self.mc_needed,
            )
            self.current_state.deformed_vertices = deformed_mesh.deformed_vertices
            self.current_state.deformed_u = deformed_mesh.u

        # self.current_state.show(self.current_trajectory._ranges)
        self.current_state.show(self.current_trajectory._ranges)

        # Hide the voxelset if it exists
        if self.voxel_set is not None:
            if self.current_trajectory.size > 1:
                self.voxel_set.set_enabled(False)

    def ps_init(self) -> None:
        """
        Initialize Polyscope
        """
        ps.set_ground_plane_mode("none")
        ps.set_max_fps(120)
        ps.set_window_size(GLOBAL_UI_CONFIG.width, GLOBAL_UI_CONFIG.height)
        # Anti-aliasing
        ps.set_SSAA_factor(4)
        # Uncomment to prevent polyscope from changing scales (including Gizmo!)
        # ps.set_automatically_compute_scene_extents(False)
        ps.set_up_dir(self.up_dir)
        ps.set_background_color([1.0, 1.0, 1.0])

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

        # Step
        self.step()

        # I usually draw things in a draw function (e.g., rendering buffer)
        self.draw()

        # Step the global key handler
        KEY_HANDLER.step()

    def gui(self) -> None:
        psim.Text(f"fps: {self.fps:.4f};")

        # if psim.Button("Update Forces"):
        #     self._update_forces()

        psim.BeginDisabled(self.optimizing)
        clicked, self.optimizing = state_button(
            self.optimizing, "Stop##dense_optimizer", "Optimize##dense_optimizer"
        )
        if clicked and self.optimizing:
            if self.voxel_set is not None and not self.region_config_locked:
                self._capture_fixed_force_voxels()
                self._apply_fixed_force_region_selection()
            self._print_selected_force_voxels()
            if not self.region_config_locked:
                self._capture_fixed_support_coords()
            self._print_fixed_support_coords()
            self._update_forces()
            self.show_current()
        psim.EndDisabled()
        # if clicked and self.optimizing:
        #     # Always call pre_optimize to take into account any change in the latent!
        #     self._pre_optimize()
        psim.SameLine()
        if psim.Button("Reset/New##physics_optimizer"):
            # TODO: this is unsafe!
            if hasattr(self, "flow_model"):
                if (
                    self.current_trajectory.cond_payload.cond is not None
                    and self.current_trajectory.cond_payload.neg_cond is not None
                    and self.current_trajectory.cond_payload.z_s is not None
                ):
                    self.init_slat_payload.cond = (
                        self.current_trajectory.cond_payload.cond
                    )
                    self.init_slat_payload.neg_cond = (
                        self.current_trajectory.cond_payload.neg_cond
                    )
                    self.init_slat_payload.z_s = (
                        self.current_trajectory.cond_payload.z_s
                    )

                    self.post_update_cond()
                else:
                    self.alert_handler.trigger(
                        "WARNING: you are trying to create a new trajectory without a proper conditioning signal!"
                    )

            self.init_trajectory(
                config=self.config,
                keep_latent=True,
                replace_current=(self.trajectory_handler.current_trajectory.size == 1),
            )
        psim.SameLine()
        clicked, self.traj_save_path = save_popup(
            "traj_dense_optimizer",
            self.traj_save_path,
            "Save Trajectories##physics_optimizer",
        )
        if clicked:
            self.trajectory_handler.save(self.traj_save_path)
            self.traj_save_path = self._get_next_save_path()

        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            psim.SameLine()
            clicked, self.all_renders_path = save_popup(
                "all_renders",
                self.all_renders_path,
                "Save All Renders##physics_optimizer",
            )
            if clicked:
                self.save_all_renders()
                self.all_renders_path = self._get_next_all_renders_path()

        # =========================
        # DEFORMED MESH
        # =========================

        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            # Mesh
            # psim.BeginDisabled(self.current_state.u is None)
            if psim.Button("Show Mesh##dense_optimizer") or KEY_HANDLER("m"):
                if self.current_state.mesh_vertices is not None:
                    deformed_mesh = self.current_state.get_deformed_mesh(
                        self.current_state.mesh_vertices,
                        self.current_state.mesh_faces,
                        config=self.config,
                        mc_based=self.mc_needed,
                    )
                    deformed_mesh.show(self.current_trajectory._ranges)
                else:
                    self.alert_handler.trigger("WARNING: Mesh is not initialized yet!")
            psim.SameLine()
            clicked, self.mesh_path = save_popup(
                "mesh_dense_optimizer",
                self.mesh_path,
                "Save Mesh##dense_optimizer",
            )
            if clicked:
                if self.current_state.mesh_vertices is not None:
                    deformed_mesh = self.current_state.get_deformed_mesh(
                        self.current_state.mesh_vertices,
                        self.current_state.mesh_faces,
                        config=self.config,
                        mc_based=self.mc_needed,
                        complete_mode=True,
                    )
                    deformed_mesh.save(self.mesh_path)
                    deformed_mesh.show(self.current_trajectory._ranges)
                    self.mesh_path = self._get_next_mesh_path()
                else:
                    self.alert_handler.trigger("WARNING: Mesh is not initialized yet!")

            # Voxels
            psim.SameLine()
            if psim.Button("Show Voxels##dense_optimizer"):
                voxel_mesh = self.current_state.get_voxel_mesh(
                    config=self.config,
                )
                voxel_mesh.show()

            psim.SameLine()
            clicked, self.voxels_path = save_popup(
                "voxels_dense_optimizer",
                self.voxels_path,
                "Save Voxels##dense_optimizer",
            )
            if clicked:
                voxel_mesh = self.current_state.get_voxel_mesh(
                    config=self.config,
                )
                voxel_mesh.save(self.voxels_path)
                voxel_mesh.show()
                self.voxels_path = self._get_next_voxels_path()

        # =========================
        # Plot loss/metrics
        # =========================

        if "total" in self.losses:
            psim.PlotLines(
                f"total##dense_optimizer",
                np.array(self.losses["total"], dtype=np.float32),
                graph_size=(300, 100),
                overlay_text=f"total",
            )

        # psim.SetNextItemOpen(True, psim.ImGuiCond_Once)
        if psim.TreeNode("All metrics"):
            for k, v in self.losses.items():
                if k == "total":
                    continue
                psim.PlotLines(
                    f"{k}##dense_optimizer",
                    np.array(v, dtype=np.float32),
                    graph_size=(300, 100),
                    overlay_text=f"{k}",
                )
                psim.Text(f"{k}: max={np.max(v):2e}; min={np.min(v):2e};")
            psim.TreePop()

        # =========================
        # Trajectory Handler
        # =========================

        psim.Separator()

        # If trajectory handler was updated, update what is rendered!
        if self.trajectory_handler.gui():
            self.show_current()

        # =========================
        # Steps
        # =========================

        psim.Separator()

        if len(self.trajectory_handler.current_trajectory.states) > 1:

            psim.BeginDisabled(self.optimizing)

            clicked, i_step = psim.SliderInt(
                "step",
                self.i_step,
                v_min=0,
                v_max=len(self.trajectory_handler.current_trajectory.states) - 1,
            )
            if clicked:
                self.set_i_step(i_step)
                self.show_current()

            psim.EndDisabled()

        # =========================
        # Mesh resolution
        # =========================

        if self.mc_needed:
            psim.Separator()

            clicked, mc_resolution_idx = psim.SliderInt(
                "mc_resolution##dataset_loader",
                ALLOWED_MC_RES_MAP[self.mc_resolution],
                v_min=0,
                v_max=len(ALLOWED_MC_RES) - 1,
                format=f"{self.mc_resolution}",
            )
            if clicked:
                self.mc_resolution = ALLOWED_MC_RES_INVMAP[mc_resolution_idx]
                self.update_current_mesh()
                self.show_current()

        # =========================
        # Shape selection gui
        # =========================

        psim.Separator()

        clicked = self.shape_selection_gui()

        if clicked:
            self.init_trajectory(
                self.config,
                replace_current=self.current_trajectory.size == 1,
                keep_latent=False,
            )

        # =========================
        # SGD Options
        # =========================

        psim.Separator()

        if psim.TreeNode("SGD Options:"):

            psim.BeginDisabled(self.current_trajectory.size > 1)

            clicked, self.config.lr = exp_slider(
                "lr##dense_optimizer",
                self.config.lr,
                v_min_exp=-6,
                v_max_exp=0,
                v_min=1e-6,
                v_max=1.0,
            )
            if clicked:
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.config.lr

            clicked, self.config.num_iters = psim.SliderInt(
                "num_sgd_iters##physics_optimizer",
                self.config.num_iters,
                v_min=1,
                v_max=200,
            )

            psim.EndDisabled()

            psim.TreePop()

        # =========================
        # Physics Options
        # =========================
        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            psim.Separator()

            # psim.SetNextItemOpen(True, psim.ImGuiCond_Once)
            if psim.TreeNode("Physics Options:"):

                psim.BeginDisabled(self.current_trajectory.size > 1)

                update = False
                clicked, opt_resolution_idx = psim.SliderInt(
                    "opt_res##physics_optimizer",
                    ALLOWED_OPT_RES_MAP[self.config.res],
                    v_min=0,
                    v_max=len(ALLOWED_OPT_RES) - 1,
                    format=f"{self.config.res}",
                )
                if clicked:
                    self.config.res = ALLOWED_OPT_RES_INVMAP[opt_resolution_idx]
                    update |= True

                # Volume term
                clicked, self.config.volume_term = scientific_slider(
                    "volume_term##physics_optimizer",
                    self.config.volume_term,
                    v_max_exp=8,
                )
                update |= clicked

                # p_exponent
                clicked, self.config.p_exponent = psim.SliderFloat(
                    "p_exponen##physics_optimizert",
                    self.config.p_exponent,
                    v_min=1.0,
                    v_max=3.0,
                )
                update |= clicked
                clicked, self.config.occ_sdf_beta = exp_slider(
                    "occ_sdf_beta##physics_optimizer", self.config.occ_sdf_beta
                )
                update |= clicked
                clicked, self.config.alpha_min = exp_slider(
                    "alpha_min##physics_optimizer",
                    self.config.alpha_min,
                    v_max=1.0,
                    v_max_exp=1,
                )
                update |= clicked
                clicked, self.config.one_cc_only = psim.Checkbox(
                    "one_cc_only##physics_optimizer", self.config.one_cc_only
                )
                update |= clicked
                clicked, self.config.boundary_cond = choice_combo(
                    "boundary_cond##physics_optimizer",
                    self.config.boundary_cond,
                    BOUNDARY_COND_MAP,
                    BOUNDARY_COND_INVMAP,
                    BOUNDARY_COND_NAMES,
                )
                update |= clicked
                clicked, self.config.bottom_fixed_threshold = psim.SliderFloat(
                    "bottom_threshold##physics_optimizer",
                    self.config.bottom_fixed_threshold,
                    v_min=0.01,
                    v_max=0.5,
                )
                update |= clicked

                if update:
                    self._pre_optimize()

                psim.EndDisabled()

                psim.TreePop()

        # =========================
        # Materials
        # =========================

        psim.Separator()

        if psim.TreeNode("Materials"):

            # NB: Materials can only be edited before optimization!
            psim.BeginDisabled(self.current_trajectory.size > 1)

            if self.config.material.gui():
                self.prepare_solid()
                self.show_current()

            psim.EndDisabled()

            psim.TreePop()

        # =========================
        # Forces
        # NB: very similar to materials
        # =========================

        psim.Separator()

        if psim.TreeNode("Forces"):

            # NB: Forces can only be edited before optimization!
            psim.BeginDisabled(self.current_trajectory.size > 1)

            # Only update forces if not started yet
            if self.config.forces_gui() and self.current_trajectory.size == 1:
                self._update_forces()
                self.show_current()

            psim.EndDisabled()

            psim.TreePop()

        psim.Separator()

        if psim.TreeNode("Region Presets"):
            psim.TextUnformatted("Save/load fixed force and support regions")
            clicked, self.region_config_path = save_popup(
                "region_config_dense_optimizer",
                self.region_config_path,
                "Save Regions##physics_optimizer",
            )
            if clicked:
                self._save_region_config(self.region_config_path)
            if psim.Button("Load Regions##physics_optimizer"):
                if os.path.isfile(self.region_config_path):
                    self._load_region_config(self.region_config_path)
                else:
                    self.alert_handler.trigger(
                        f"WARNING: region config file not found: {self.region_config_path}"
                    )
            if self.fixed_force_coords is not None:
                psim.Text(
                    f"fixed force voxels: {int(np.asarray(self.fixed_force_coords).reshape(-1, 3).shape[0])}"
                )
            if self.fixed_support_coords is not None:
                psim.Text(
                    f"fixed support nodes: {int(np.asarray(self.fixed_support_coords).reshape(-1, 3).shape[0])}"
                )
            psim.TreePop()

        # =========================
        # UI
        # =========================
        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            psim.Separator()

            if psim.TreeNode("UI##physics_optimizer"):

                psim.BeginDisabled(self.current_trajectory.size > 1)

                _, self.config.autorescale_u_mean_halt = psim.SliderFloat(
                    "autorescale_u_mean_halt##physics_optimizer",
                    self.config.autorescale_u_mean_halt,
                    v_min=0.0,
                    v_max=0.5,
                )

                psim.EndDisabled()

                psim.TreePop()

        # =========================
        # REPLAY
        # =========================

        # DEV MODE only
        if GLOBAL_UI_CONFIG.dev_mode:

            psim.Separator()

            if psim.TreeNode("Replay##physics_optimizer"):

                clicked, upscale_factor_idx = psim.SliderInt(
                    "upscale_factor##physics_optimizer",
                    ALLOWED_UPSCALE_FACTORS_MAP[self.upscale_factor],
                    v_min=0,
                    v_max=len(ALLOWED_UPSCALE_FACTORS) - 1,
                    format=f"{self.upscale_factor}",
                )
                if clicked:
                    self.upscale_factor = ALLOWED_UPSCALE_FACTORS_INVMAP[
                        upscale_factor_idx
                    ]

                _, self.replay_indices = psim.InputText(
                    "replay_indices##physics_optimizer", self.replay_indices
                )

                # Up-res the current trajectory
                if psim.Button("Upres"):
                    new_config = deepcopy(self.config)
                    new_config.res = self.config.res * self.upscale_factor
                    ref_traj_idx = self.trajectory_handler.current_idx
                    self.init_trajectory(
                        config=new_config,
                        keep_latent=True,
                        replace_current=False,
                    )
                    # Filter all indices
                    additional_indices = parse_int_list(self.replay_indices)
                    additional_indices = [
                        idx
                        for idx in additional_indices
                        if idx >= 0
                        and idx
                        < len(self.trajectory_handler.trajectories[ref_traj_idx].states)
                    ]
                    target_indices = (
                        [0]
                        + additional_indices
                        + [
                            len(
                                self.trajectory_handler.trajectories[
                                    ref_traj_idx
                                ].states
                            )
                            - 1
                        ]
                    )
                    for i_state in tqdm(target_indices):
                        state = self.trajectory_handler.trajectories[
                            ref_traj_idx
                        ].states[i_state]
                        if not self.set_replay_state(state):
                            print(
                                f"Failed to retrieve latents from state {i_state} of this trajectory!"
                            )
                            break
                        self.prepare_solid(i_state == 0)
                        # Multiply forces down by the volume of the voxel
                        self.solid.forces *= (1.0 / self.upscale_factor) ** 3
                        self.u, self.f_int, self.f_ext, self.sigma, _, _ = (
                            self.solid.solve(max_iter=1)
                        )
                        self.update_results()
                        if (
                            i_state
                            < len(
                                self.trajectory_handler.trajectories[
                                    ref_traj_idx
                                ].states
                            )
                            - 1
                        ):
                            self.trajectory_handler.current_trajectory.add(
                                OptimizationState()
                            )

                    self.show_current()

                psim.TreePop()

        # =========================
        # VOXEL_SET
        # =========================
        if self.voxel_set is not None:
            # Make sure to step before to clear the selection
            if self.current_trajectory.size == 1:
                if self.voxel_set.gui():
                    self.region_config_locked = False
                    self._capture_fixed_force_voxels()
                    self._print_selected_force_voxels()
                    self._update_forces()
                    self.show_current()

        # =========================
        # Deformation Alert
        # =========================
        self.alert_handler.gui()

    # Updates polyscope with the results of a solve
    @torch.no_grad()
    def update_results(self):
        # ============
        # Latents + Representations
        # ============
        if self.mc_needed:
            self.update_current_mesh()

        # WARNING: this is extremely HARDCODED for TRELLIS, but only it should need it.
        else:
            self.current_state.slat = self.slat.clone().cuda()  # just in case
            # Decoder splats and meshes (cache!)
            self.current_state.splats = self.gaussian_decoder(self.current_state.slat)[
                0
            ]
            mesh = self.mesh_decoder(self.current_state.slat)[0]
            self.current_state.mesh_vertices = mesh.vertices.detach().cpu().numpy()
            self.current_state.mesh_faces = mesh.faces.detach().cpu().numpy()

        # If there is a latent, save it too
        if hasattr(self, "latent") and getattr(self, "latent") is not None:
            self.current_state.latent = self.latent.detach().clone()

        # ============
        # Solid
        # ============
        self.current_state.u = self.u.detach().cpu().numpy()

        mises = get_von_mises_stress(self.sigma)

        self.current_state.mises = mises.detach().cpu().numpy()
        self.current_state.sigma = self.sigma.detach().cpu().numpy()

        # ============
        # Sensitivity
        # ============

        self.current_state.sensitivity_density = (
            self.sensitivity_density.detach().cpu().numpy()
        )

    def save_all_renders(self):
        os.makedirs(self.all_renders_path, exist_ok=True)
        for i_step in tqdm(range(self.current_trajectory.size)):
            self.set_i_step(i_step)
            self.show_current()
            # Deformed meshes
            try:
                if self.current_state.mesh_vertices is not None:
                    deformed_mesh = self.current_state.get_deformed_mesh(
                        self.current_state.mesh_vertices,
                        self.current_state.mesh_faces,
                        config=self.config,
                        mc_based=self.mc_needed,
                    )
                    deformed_mesh.save(
                        os.path.join(
                            self.all_renders_path,
                            f"deformed_mesh_{i_step:03d}.npz",
                        )
                    )
            except Exception as e:
                print(e)

            # Gaussians
            try:
                if self.current_state.splats is not None:
                    self.current_state.splats.save_ply(
                        os.path.join(
                            self.all_renders_path,
                            f"gaussians_{i_step:03d}.ply",
                        )
                    )
            except Exception as e:
                print(e)

    # @torch.no_grad()
    def training_step(self):
        loss_dict = {}
        if self.i_step > self.config.num_iters:
            print("WARNING: this should never happen!")
            return False, {}

        # Clear memory
        if self.i_step > 0:
            del self.nodes
            del self.elements
            del self.solid
            del self.u
            del self.f_int
            del self.f_ext
            del self.sigma
            # del self.coarse_coords # Don't delete this for tracking added voxels
            del self.coarse_occ
            (
                self.nodes,
                self.elements,
                self.solid,
                self.u,
                self.f_int,
                self.f_ext,
                self.sigma,
                # self.coarse_coords,
                self.coarse_occ,
            ) = (None, None, None, None, None, None, None, None)
            torch.cuda.empty_cache()

        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad()

        # ================
        # SIMULATION
        # ================
        with torch.no_grad():
            self.micro_timer.reset()

            self.micro_timer.start("time_prepare_solid")
            self.prepare_solid()
            self.micro_timer.stop("time_prepare_solid")

            self.micro_timer.start("time_solve")
            self.u, self.f_int, self.f_ext, self.sigma, _, _ = self.solid.solve(
                max_iter=1
            )
            self.micro_timer.stop("time_solve")

            # If rescaling is enabled, scale down the force and rerun
            if self.config.autorescale and self.i_step == 0:
                MAX_RESCALE_ITERATION = 10
                for i_rescale in range(MAX_RESCALE_ITERATION):
                    if (
                        torch.abs(self.u).mean()
                        <= self.config.autorescale_u_mean_target
                    ):
                        break
                    print(
                        f"Displacement is too big ({torch.abs(self.u).mean()})! Rescaling forces by {self.config.autorescale_u_factor} ({i_rescale}/{MAX_RESCALE_ITERATION})"
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
                self.alert_handler.trigger(DEFORMATION_ALERT_MESSAGE)
                return False, loss_dict
                print("Displacement is too big!")

        # ================
        # MANUAL BACKPROP
        # NB: this allows to filter locally sensitivies (or not)
        # ================

        self.micro_timer.start("time_backward")


        # Compute analytical sensitivities
        k0 = self.solid.k0()  # Stiffness for zero strain
        u_j = self.u[self.elements].reshape(self.solid.n_elem, -1)
        w_k = torch.einsum("...i, ...ij, ...j", u_j, k0, u_j)

        compliance = torch.sum(
            (self.coarse_occ ** self.config.p_exponent) * w_k
        )

        loss_dict["total"] = w_k.mean().item() #
        loss_dict["displacement"] = torch.abs(self.u).mean().item()
        loss_dict["compliance"] = compliance.item()
        loss_dict["n_voxels"] = len(self.elements)

       # Compute sensitivity w.r.t., density (easy, no grad)
        self.sensitivity_density = (
            -self.config.p_exponent
            * self.coarse_occ ** (self.config.p_exponent - 1)
            * w_k
        )  
  
        # ==========================================
        # ADAM Setup:Compliance Minimization + Volume Regularization
        # ==========================================
        if self.config.volume_term > 0.0:
            self.sensitivity_volume = (
                self.config.volume_term
                * (1.0 / self.config.res) ** 3
                * torch.ones_like(self.sensitivity_density)
            )
            # Add a sign depending on higher/lower volume
            self.sensitivity_volume *= (
                1.0 if self.full_occ.mean() >= self.init_volume else -1.0
            )
            # print(self.sensitivity_volume)
            # print(self.sensitivity_volume.norm(p=2).item())
            self.sensitivity_density += self.sensitivity_volume

            loss_dict["volume"] = (self.full_occ).mean().item()
            loss_dict["total"] += loss_dict["volume"]

        

        self.backward_step()

        self.micro_timer.stop("time_backward")

        self.micro_timer.start("time_post_solve")

        self.post_solve_step()

        self.micro_timer.stop("time_post_solve")

        loss_dict |= self.micro_timer.collect()

        # NB: For the last step, we only do forward simulation to get the metrics but the
        # latent remains unchanged!
        if self.i_step >= self.config.num_iters:

            self.update_results()
            self.show_current()

            return False, loss_dict

        else:
            self.optimizer.step()  # yuan shi
            self.update_results()
            self.show_current()
            self.trajectory_handler.current_trajectory.add(OptimizationState())
            return True, loss_dict

    def step(self) -> None:
        strat_time = time.time()
        if self.optimizing:
            self.optimizing, loss_dict = self.training_step()
            end_time = time.time()
            print(f"Time (Step): {self.i_step:03d} | {end_time - strat_time:.4f}s | compliance: {loss_dict['compliance']:.5e} | volume: {loss_dict['volume']:.5e}")
            
            for k, v in loss_dict.items():
                self.losses[k].append(v)

    # ================================
    # SAVE & DROP CALLBACK
    # ================================

    def ps_drop_callback(self, input: str) -> None:
        extension = os.path.splitext(input)[1]
        if extension == ".traj":
            try:
                self.trajectory_handler.add_from_file(input)
                # Display!
                self.show_current()
            except Exception as e:
                # handle the exception
                print("Could not import from:", input)
                print("Error:\n", e)
        elif self.additional_ps_drop_callback(input, extension):
            pass
        else:
            print("Only .traj, .embed and .cam files are accepted!")

    # ================================
    # VISUALIZATION (SLICES)
    # ================================

    @torch.no_grad()
    def _get_grid_coords(self, res: int, device: str = "cuda"):
        mins = (-0.5, -0.5, -0.5)
        maxs = (0.5, 0.5, 0.5)
        xs = torch.linspace(mins[0], maxs[0], res, device=device)
        ys = torch.linspace(mins[1], maxs[1], res, device=device)
        zs = torch.linspace(mins[2], maxs[2], res, device=device)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), -1)  # (R,R,R,3)
        pts = grid.reshape(-1, 3)
        return pts
