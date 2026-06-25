import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from enum import Enum
from typing import Tuple

from physiopt.utils.ps_utils import (
    CUBE_VERTICES_NP,
    CUBE_TRIANGLES_NP,
)
from physiopt.vis.ui_utils import get_enum_maps, KEY_HANDLER

import torch


class BrushMode(Enum):
    ADD = "add"
    REMOVE = "remove"


BRUSH_MODE_MAP, BRUSH_MODE_INVMAP, BRUSH_MODE_NAMES, _ = get_enum_maps(BrushMode)


DEFAULT_HOVER_ADD_COLOR = np.array([0.0, 1.0, 0.0])
DEFAULT_HOVER_ERASE_COLOR = np.array([1.0, 0.0, 0.0])
DEFAULT_SELECTED_COLOR = np.array([0.0, 0.0, 1.0])
DEFAULT_BASE_COLOR = np.array([1.0, 1.0, 1.0])

DEFAULT_SELECTION_RADIUS = 4.0
SELECTION_RADIUS_SENTIVITY = 0.5
MIN_SELECTION_RADIUS = 0.001
MAX_SELECTION_RADIUS = 32.0
DEFAULT_BRUSH_MODE = BrushMode.ADD


# TODO: remove because this is a duplicate
@torch.no_grad()
def create_voxel_set_np(
    coords: np.ndarray,
    voxel_res: int,
    bbox_min: float,
    bbox_max: float,
    name: str = "voxel_set",
    offset: np.ndarray | None = None,
    enabled: bool = True,
) -> Tuple[ps.SurfaceMesh, np.ndarray, np.ndarray]:
    # self.voxels = coord_bbox_filter(self.voxels, self.res)

    vertex_offsets = np.repeat(coords, 8, axis=0)
    cube_vertices = (
        np.tile(CUBE_VERTICES_NP, (len(coords), 1)) - 0.5
    )  # Rescale voxels (each center with point cloud)
    vertices = cube_vertices + vertex_offsets
    vertices = (bbox_max - bbox_min) * (1.0 / float(voxel_res)) * vertices + bbox_min

    if offset is not None:
        vertices += offset

    # 8 for 8 vertices
    triangles_offsets = np.repeat(
        np.tile((8 * np.arange(len(coords)))[:, None], ((1, 3))),
        len(CUBE_TRIANGLES_NP),
        axis=0,
    )
    faces = np.tile(CUBE_TRIANGLES_NP, (len(coords), 1)) + triangles_offsets

    ps_voxels = ps.register_surface_mesh(
        name,
        vertices,
        faces,
        enabled=enabled,
    )
    ps_voxels.set_edge_width(0.0)

    return ps_voxels, vertices, faces


class VoxelSet:

    def __init__(
        self,
        coords: np.ndarray,
        voxel_res: int,
        bbox_min: float,
        bbox_max: float,
        name: str = "voxel_set",
        offset: np.ndarray | None = None,
        selection_mask: np.ndarray | None = None,
    ):

        self.square_brush = True
        self.selection_radius = DEFAULT_SELECTION_RADIUS
        self.brush_mode = DEFAULT_BRUSH_MODE
        self.square_brush = True
        self.last_selected_voxel_id = -1
        self.selection_changed = False
        self.name = name

        self.coords = coords
        self.selection_mask = self._normalize_selection_mask(selection_mask)

        self.ps_voxels, self.vertices, self.faces = create_voxel_set_np(
            coords=coords,
            voxel_res=voxel_res,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            name=name,
            offset=offset,
        )

        self.ps_voxels.add_color_quantity(
            name + "selection",
            np.tile(DEFAULT_BASE_COLOR[None, :], (len(self.faces), 1)),
            defined_on="faces",
            enabled=True,
        )

        self.selection_buffer = self.ps_voxels.get_quantity_buffer(
            name + "selection", "colors"
        )

        self.update_selection_buffer()

        # Precompute world coordinates for manual raycasting
        self.voxel_size = (bbox_max - bbox_min) / float(voxel_res)
        self.world_coords = self.voxel_size * coords + bbox_min
        if offset is not None:
            self.world_coords += offset

        self.ps_voxels.set_hover_callback(self.hover_callback)

    def _normalize_selection_mask(
        self, selection_mask: np.ndarray | None = None
    ) -> np.ndarray:
        expected_size = len(self.coords)
        if selection_mask is None:
            return np.zeros(expected_size, dtype=bool)

        mask = np.asarray(selection_mask, dtype=bool).reshape(-1)
        if mask.size == expected_size:
            return mask.copy()

        normalized_mask = np.zeros(expected_size, dtype=bool)
        copy_size = min(mask.size, expected_size)
        normalized_mask[:copy_size] = mask[:copy_size]
        return normalized_mask

    def update_selection_buffer(self, within_radius: np.ndarray | None = None):
        self.selection_mask = self._normalize_selection_mask(self.selection_mask)
        current_selection = np.tile(DEFAULT_BASE_COLOR[None, :], (len(self.coords), 1))
        current_selection[self.selection_mask] = DEFAULT_SELECTED_COLOR

        if within_radius is not None:
            within_radius = self._normalize_selection_mask(within_radius)
            current_selection[within_radius] = (
                DEFAULT_HOVER_ADD_COLOR
                if self.brush_mode == BrushMode.ADD
                else DEFAULT_HOVER_ERASE_COLOR
            )

        current_selection = np.repeat(current_selection, 12, axis=0)

        self.selection_buffer.update_data_from_host(current_selection)

    def _handle_hover(self, voxel_id: int):
        self.selection_mask = self._normalize_selection_mask(self.selection_mask)
        hovered_voxel = self.coords[voxel_id]

        if self.square_brush:
            within_radius = (np.abs(hovered_voxel[None, :] - self.coords)).max(
                1
            ) < self.selection_radius
        else:
            within_radius = ((hovered_voxel[None, :] - self.coords) ** 2).sum(
                1
            ) < self.selection_radius**2

        io = psim.GetIO()
        is_alt_down = io.KeyAlt
        try:
            if hasattr(psim, 'ImGuiKey_ModAlt'):
                is_alt_down = is_alt_down or psim.IsKeyDown(psim.ImGuiKey_ModAlt)
            if hasattr(psim, 'ImGuiKey_LeftAlt'):
                is_alt_down = is_alt_down or psim.IsKeyDown(psim.ImGuiKey_LeftAlt)
        except:
            pass

        if (
            psim.IsMouseDown(0)
            and is_alt_down
        ):
            if self.brush_mode == BrushMode.REMOVE:
                self.selection_mask &= ~within_radius
            else:
                self.selection_mask |= within_radius

            self.last_selected_voxel_id = voxel_id
            self.selection_changed = True

        self.update_selection_buffer(within_radius)

    def hover_callback(self, mesh_element: ps.MeshElement, index: int):
        if mesh_element == ps.MeshElement.VERTEX.value:
            voxel_id = index // 8
        elif mesh_element == ps.MeshElement.FACE.value:
            voxel_id = index // 12
        else:
            return
        self._handle_hover(voxel_id)

    def manual_hover_check(self):
        io = psim.GetIO()
        # if io.WantCaptureMouse:
        #    return

        try:
            # Try getting camera parameters from polyscope
            # io.MousePos might be a tuple in some imgui bindings
            mouse_pos_raw = io.MousePos
            if hasattr(mouse_pos_raw, 'x'):
                mouse_x = mouse_pos_raw.x
                mouse_y = mouse_pos_raw.y
            else:
                mouse_x, mouse_y = mouse_pos_raw
                
            try:
                ray_dir = ps.screen_coords_to_world_ray(mouse_x, mouse_y)
            except TypeError:
                ray_dir = ps.screen_coords_to_world_ray((mouse_x, mouse_y))
            
            cam_params = ps.get_view_camera_parameters()
            if hasattr(cam_params, 'get_position'):
                cam_pos = np.array(cam_params.get_position())
            elif hasattr(cam_params, 'root'):
                cam_pos = np.array(cam_params.root)
            elif hasattr(cam_params, '__getitem__'):
                # Try to access it as a tuple
                cam_pos = np.array(cam_params[2])
            else:
                # If we really can't get it, we can't do raycasting
                return

            # Simple intersection test with bounding spheres of voxels
            voxel_radius = float(self.voxel_size) * 0.866 # sqrt(3)/2 * size
            
            vecs = self.world_coords - cam_pos
            proj_lengths = np.dot(vecs, ray_dir)
            
            # Only consider voxels in front of camera
            valid_mask = proj_lengths > 0
            if np.any(valid_mask):
                proj_points = cam_pos + np.outer(proj_lengths[valid_mask], ray_dir)
                dists_sq = np.sum((self.world_coords[valid_mask] - proj_points)**2, axis=1)
                
                hit_mask = dists_sq < (voxel_radius**2)
                if np.any(hit_mask):
                    # Among the ones hit, find the one closest to the camera (smallest proj_length)
                    valid_proj_lengths = proj_lengths[valid_mask]
                    hit_proj_lengths = valid_proj_lengths[hit_mask]
                    closest_hit_idx_in_hits = np.argmin(hit_proj_lengths)
                    
                    # Map back to original indices
                    hit_indices = np.where(valid_mask)[0][hit_mask]
                    hit_idx = hit_indices[closest_hit_idx_in_hits]
                    
                    self._handle_hover(hit_idx)
                    return
        except Exception as e:
            print(f"manual_hover_check ray casting failed: {e}")
            pass

        # Fallback to screen space projection if ray casting fails
        try:
            params = ps.get_view_camera_parameters()
            if hasattr(params, 'get_view_matrix'):
                view_mat = np.array(params.get_view_matrix())
                proj_mat = np.array(params.get_projection_matrix())
            elif hasattr(params, 'E'):
                view_mat = np.array(params.E)
                proj_mat = np.array(params.fov) # Wait, fov is not a matrix.
            else:
                # If params is not subscriptable (like polyscope.core.CameraParameters)
                # We can't use it directly this way, try get_view_matrix
                pass
        except Exception as e:
            print(f"manual_hover_check fallback failed: {e}")
            return
            
        try:
            # Check if view_mat and proj_mat were defined
            view_mat
            proj_mat
        except NameError:
            return

        mouse_pos_raw = io.MousePos
        if hasattr(mouse_pos_raw, 'x'):
            mouse_x = mouse_pos_raw.x
            mouse_y = mouse_pos_raw.y
        else:
            mouse_x, mouse_y = mouse_pos_raw
        mouse_pos = np.array([mouse_x, mouse_y])
        win_w, win_h = ps.get_window_size()

        # Project coords to screen
        N = len(self.world_coords)
        coords_h = np.hstack([self.world_coords, np.ones((N, 1))])
        
        # very simple projection (only if the matrices are 4x4)
        if view_mat.shape == (4,4) and proj_mat.shape == (4,4):
            clip = coords_h @ (proj_mat @ view_mat).T
            
            w = clip[:, 3]
            mask = w > 0.1 # In front of camera
            
            if not np.any(mask):
                self.update_selection_buffer(None)
                return

            ndc = clip[mask, :2] / w[mask, None]
            screen_x = (ndc[:, 0] + 1) * 0.5 * win_w
            screen_y = (1 - ndc[:, 1]) * 0.5 * win_h
            screen = np.stack([screen_x, screen_y], axis=1)

            dists_sq = np.sum((screen - mouse_pos)**2, axis=1)
            min_idx = np.argmin(dists_sq)
            
            if dists_sq[min_idx] < 900: # 30px radius
                voxel_id = np.where(mask)[0][min_idx]
                self._handle_hover(voxel_id)
                return
        self.update_selection_buffer(None)

    def gui(self) -> bool:
        update = False

        psim.SeparatorText(f"{self.name}##voxel_set")

        io = psim.GetIO()

        is_alt_down = io.KeyAlt
        try:
            if hasattr(psim, 'ImGuiKey_ModAlt'):
                is_alt_down = is_alt_down or psim.IsKeyDown(psim.ImGuiKey_ModAlt)
            if hasattr(psim, 'ImGuiKey_LeftAlt'):
                is_alt_down = is_alt_down or psim.IsKeyDown(psim.ImGuiKey_LeftAlt)
        except:
            pass

        # Use wheel to increase/decrease brush radius
        if io.MouseWheel != 0 and is_alt_down:
            self.selection_radius += SELECTION_RADIUS_SENTIVITY * float(io.MouseWheel)
        self.selection_radius = max(
            MIN_SELECTION_RADIUS,
            min(self.selection_radius, MAX_SELECTION_RADIUS),
        )

        # Switch selection
        if KEY_HANDLER("s"):
            self.brush_mode = BRUSH_MODE_INVMAP[1 - BRUSH_MODE_MAP[self.brush_mode]]

        # Manual hover check (since set_hover_callback might be missing/broken)
        self.manual_hover_check()

        psim.Text(f"Radius: {self.selection_radius}")

        psim.SameLine()
        if psim.Button(f"Reset##voxel_set_{self.name}"):
            self.selection_mask = np.zeros_like(self.selection_mask)
            self.update_selection_buffer()
            update |= True

        psim.SameLine()
        if psim.Button(f"Invert##voxel_set_{self.name}"):
            self.selection_mask = ~self.selection_mask
            self.update_selection_buffer()
            update |= True

        # Notify parent if something changed
        update |= self.selection_changed
        self.selection_changed = False
        return update

    def step(self) -> bool:
        self.update_selection_buffer()

    def set_enabled(self, val: bool = True):
        self.ps_voxels.set_enabled(val)
