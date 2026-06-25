from typing import Tuple
import torch

import numpy as np


# I might have inverted axes but it doesn't matter for what we're doing!
def flat_to_grid_idx(indices: torch.Tensor, res: int):
    x = indices // res**2
    y = (indices - x * res**2) // res
    z = indices - x * res**2 - y * res
    assert torch.all((x >= 0) & (x < res))
    assert torch.all((y >= 0) & (y < res))
    assert torch.all((z >= 0) & (z < res))
    return torch.stack([x, y, z], dim=-1)


def grid_to_flat_idx(indices: torch.Tensor, res: int):
    assert indices.shape[-1] == 3
    return indices[..., 0] * res**2 + indices[..., 1] * res + indices[..., 2]


# Implements square-shaped structuring elements
def get_dilation_offsets_3d(radius: int):
    lin_range = torch.arange(
        -radius,
        radius + 1,
    )
    grid_x, grid_y, grid_z = torch.meshgrid(
        lin_range,
        lin_range,
        lin_range,
        indexing="ij",
    )
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    stroke_offsets = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    return stroke_offsets


# Filter coordinates that fall outside of the bbox
def coord_bbox_filter(coord: torch.Tensor, res: int, return_indices: bool = False):
    coord_idx = torch.argwhere(
        (torch.min(coord, dim=-1)[0] >= 0) & (torch.max(coord, dim=-1)[0] < res)
    ).squeeze(1)
    if return_indices:
        return coord[coord_idx], coord_idx
    return coord[coord_idx]


# Optimal way of doing this imply wavefront tracking with small steps (t, t-1, t-2) marking strategy
@torch.no_grad()
def dilate(
    coord: torch.Tensor,
    iterations: int,
    radius: int,
    res: int,
) -> torch.Tensor:
    """
    Performs standard 3D Morphological Dilation with a Square Structuring
    Element of radius `radius` for `iterations` iterations
    """
    assert len(coord.shape) == 2
    assert coord.shape[1] == 3
    dilation_offsets = get_dilation_offsets_3d(radius).long().cuda()
    all_coords = coord
    for _ in range(iterations):
        current_set_coord = all_coords.clone()
        for offset in dilation_offsets:
            offset_coords = current_set_coord + offset.unsqueeze(0)
            # First filter anything that is outside
            offset_coords = coord_bbox_filter(offset_coords, res)
            # Select unique offset coord
            all_coords = torch.unique(
                torch.cat([all_coords, offset_coords], dim=0), dim=0
            )

    return all_coords


@torch.no_grad()
def erode(
    coord: torch.Tensor,
    iterations: int,
    radius: int,
    res: int,
):
    if len(coord) == 0:
        return coord

    assert len(coord.shape) == 2
    assert coord.shape[1] == 3
    assert coord.min() >= 0 and coord.max() < res

    dilation_offsets = get_dilation_offsets_3d(radius).long().cuda()
    all_coords = coord
    for _ in range(iterations):
        erosion_test = torch.ones(
            (all_coords.shape[0]), dtype=torch.bool, device="cuda"
        )
        for offset in dilation_offsets:
            offset_coords = all_coords + offset.unsqueeze(0)
            # First filter anything that is outside
            offset_coords, indices = coord_bbox_filter(
                offset_coords, res, return_indices=True
            )
            flatten_offset_coords = grid_to_flat_idx(offset_coords, res)
            flatten_state_coords = grid_to_flat_idx(all_coords, res)
            tmp_test = torch.isin(flatten_offset_coords, flatten_state_coords)
            new_test = torch.ones_like(erosion_test)
            new_test[indices] = tmp_test
            erosion_test &= new_test

        all_coords = all_coords[erosion_test]

    return all_coords


def index_of_a_in_b(_a, b):
    # Trick because a needs to be made of unique elements...
    if len(torch.unique(_a)) != len(_a):
        a, invmap = torch.unique(_a, return_inverse=True)
    else:
        a = _a
        invmap = None
    b_indices = torch.where(torch.isin(b, a))[0]
    b_values = b[b_indices]
    result = b_indices[b_values.argsort()[a.argsort().argsort()]]
    if invmap is not None:
        result = result[invmap]
    return result


# ================================
# CONNECTED_COMPONENTS
# ================================
def filter_one_component(element_mask: torch.Tensor, res: int):

    import cupy as cp
    from torch.utils.dlpack import to_dlpack
    from torch.utils.dlpack import from_dlpack
    from cupyx.scipy.ndimage import label

    # assume `vol` is your (D,H,W) NumPy bool or 0/1 array
    vol_gpu = cp.from_dlpack(to_dlpack(element_mask.reshape(res, res, res))).astype(
        cp.uint8
    )
    # vol_gpu = cp.asarray(element_mask, dtype=cp.uint8)

    # define connectivity (3×3×3 all-ones → 26-conn; for 6-conn use a sparse structuring element)
    structure = cp.ones((3, 3, 3), dtype=cp.uint8)

    labels_gpu, num_labels = label(vol_gpu, structure=structure)
    labels_gpu = from_dlpack(labels_gpu.toDlpack())

    _, counts = torch.unique(labels_gpu, return_counts=True)
    sorted_labels = np.argsort(counts.detach().cpu().numpy())

    occupied_index = sorted_labels[-2]
    return (labels_gpu == occupied_index).flatten()


# ================================
# VOLUME -> MESH
# ================================
def get_indices_np(queries: np.ndarray, res: int, is_element_quantity: bool = False):
    if not is_element_quantity:
        node_indices = (queries + 0.5) * res
        node_indices_floored = np.floor(node_indices).astype(np.int64)
        assert node_indices_floored.min() >= 0 and node_indices_floored.max() < res + 1
    else:
        element_side_length = 1.0 / res
        node_indices = (queries + 0.5 - element_side_length * 0.5) * (res - 1)
        node_indices_floored = np.floor(node_indices).astype(np.int64)
        assert node_indices_floored.min() >= -1 and node_indices_floored.max() < res + 1

    # node_indices = np.clip(node_indices, 0, res + 1)
    # node_indices_floored = np.clip(node_indices_floored, 0, res)

    local_coords = node_indices - node_indices_floored.astype(np.float32)

    return node_indices_floored, local_coords


def get_indices(queries: torch.Tensor, res: int, is_element_quantity: bool = False):
    if not is_element_quantity:
        node_indices = (queries + 0.5) * res
        node_indices_floored = torch.floor(node_indices).long()
        assert node_indices_floored.min() >= 0 and node_indices_floored.max() < res + 1
    else:
        element_side_length = 1.0 / res
        node_indices = (queries + 0.5 - element_side_length * 0.5) * (res - 1)
        node_indices_floored = torch.floor(node_indices).long()
        assert node_indices_floored.min() >= -1 and node_indices_floored.max() < res + 1

    # node_indices = np.clip(node_indices, 0, res + 1)
    # node_indices_floored = np.clip(node_indices_floored, 0, res)

    local_coords = node_indices - node_indices_floored.float()

    return node_indices_floored, local_coords


def get_all_surroundings_np(indices: np.ndarray):
    OFFSETS = np.array(
        [
            np.array([0, 0, 0]).astype(np.int64),  # 0
            np.array([1, 0, 0]).astype(np.int64),  # 1
            np.array([0, 0, 1]).astype(np.int64),  # 2
            np.array([1, 0, 1]).astype(np.int64),  # 3
            np.array([0, 1, 0]).astype(np.int64),  # 4
            np.array([1, 1, 0]).astype(np.int64),  # 5
            np.array([0, 1, 1]).astype(np.int64),  # 6
            np.array([1, 1, 1]).astype(np.int64),  # 7
        ]
    ).astype(np.int64)
    surrounding_indices = indices[:, None, :] + OFFSETS[None, :, :]
    return surrounding_indices


def get_all_surroundings(indices: torch.Tensor):
    OFFSETS = (
        torch.tensor(
            [
                [0, 0, 0],  # 0
                [1, 0, 0],  # 1
                [0, 0, 1],  # 2
                [1, 0, 1],  # 3
                [0, 1, 0],  # 4
                [1, 1, 0],  # 5
                [0, 1, 1],  # 6
                [1, 1, 1],  # 7
            ],
            dtype=torch.long,
            device=indices.device
        )
    )
    surrounding_indices = indices[:, None, :] + OFFSETS[None, :, :]
    return surrounding_indices


def flat_to_grid_idx_np(indices: torch.Tensor, res: int):
    x = indices // res**2
    y = (indices - x * res**2) // res
    z = indices - x * res**2 - y * res
    assert np.all((x >= 0) & (x < res))
    assert np.all((y >= 0) & (y < res))
    assert np.all((z >= 0) & (z < res))
    return np.stack([x, y, z], axis=-1)


def grid_to_flat_idx_np(indices: torch.Tensor, res: int):
    return indices[..., 0] * res**2 + indices[..., 1] * res + indices[..., 2]


def trilinear_interpolation_np(
    local_coords: np.ndarray | torch.Tensor, corner_values: np.ndarray | torch.Tensor
):
    x = local_coords[:, 0][:, None]
    y = local_coords[:, 1][:, None]
    z = local_coords[:, 2][:, None]
    # Interpolate along x-axis at z=0 and z=1 planes
    c00 = corner_values[:, 0] * (1 - x) + corner_values[:, 1] * x
    c01 = corner_values[:, 2] * (1 - x) + corner_values[:, 3] * x
    c10 = corner_values[:, 4] * (1 - x) + corner_values[:, 5] * x
    c11 = corner_values[:, 6] * (1 - x) + corner_values[:, 7] * x

    # Interpolate along y-axis
    c0 = c00 * (1 - y) + c10 * y
    c1 = c01 * (1 - y) + c11 * y

    # Interpolate along z-axis
    result = c0 * (1 - z) + c1 * z
    return result


def index_of_a_in_b_np(_a, b):
    # Trick because a needs to be made of unique elements...
    if len(np.unique(_a)) != len(_a):
        a, invmap = np.unique(_a, return_inverse=True)
    else:
        a = _a
        invmap = None
    b_indices = np.where(np.isin(b, a))[0]
    b_values = b[b_indices]
    result = b_indices[b_values.argsort()[a.argsort().argsort()]]
    if invmap is not None:
        result = result[invmap]
    return result


# ================================
# VOXELS -> MESH
# ================================
def generate_mesh_from_voxel_coordinates(
    voxel_coords, voxel_occ: np.ndarray | bool | None
):
    """
    Generate a surface mesh from a set of voxel coordinates.

    Parameters:
        voxel_coords (np.ndarray): Array of shape (N, 3) with voxel coordinates.

    Returns:
        vertices (np.ndarray): Array of vertices (x, y, z).
        faces (np.ndarray): Array of triangular faces (indices of vertices).
    """
    voxel_coords = np.array(voxel_coords)
    voxels = set(map(tuple, voxel_coords))

    # Map from local face vertices to global mesh
    face_directions = {
        (1, 0, 0): [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],
        (-1, 0, 0): [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        (0, 1, 0): [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
        (0, -1, 0): [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
        (0, 0, 1): [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)],
        (0, 0, -1): [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
    }

    vertices = []
    faces = []
    if voxel_occ is not None:
        vox_indices = []
    vertex_map = {}
    vertex_counter = 0

    for i_voxel, voxel in enumerate(voxels):
        vx, vy, vz = voxel

        for direction, face in face_directions.items():
            neighbor = (vx + direction[0], vy + direction[1], vz + direction[2])

            # Only add the face if the neighbor voxel is missing
            if neighbor not in voxels:
                face_vertices = [(vx + dx, vy + dy, vz + dz) for dx, dy, dz in face]

                # Add vertices to the global list, ensuring uniqueness
                face_indices = []
                for vertex in face_vertices:
                    if vertex not in vertex_map:
                        vertex_map[vertex] = vertex_counter
                        vertices.append(vertex)
                        vertex_counter += 1
                    face_indices.append(vertex_map[vertex])

                # Add the face (two triangles)
                faces.append([face_indices[0], face_indices[1], face_indices[2]])
                faces.append([face_indices[0], face_indices[2], face_indices[3]])

                if voxel_occ is not None:
                    vox_indices.append(voxel)
                    vox_indices.append(voxel)

    vertices = np.array(vertices)
    faces = np.array(faces)
    if voxel_occ is not None:
        vox_indices = np.array(vox_indices)
        return vertices, faces, vox_indices
    return vertices, faces


def hex_to_tri_surface(
    nodes: np.ndarray, voxels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized extraction of the triangular surface mesh from a hexahedral mesh.

    Args:
      nodes:  (n_nodes, 3)  array of XYZ coordinates
      voxels: (n_voxels, 8) array of node‐indices for each hexahedron

    Returns:
      surf_nodes: same as `nodes`
      tris:       (n_triangles, 3) array of triangle node‐indices
    """
    # the 6 local faces of a hex, in your generate_hex ordering
    face_corners = np.array(
        [
            [0, 1, 2, 3],  # bottom  (k=0)
            [4, 5, 6, 7],  # top     (k=1)
            [0, 1, 5, 4],  # front   (j=0)
            [3, 2, 6, 7],  # back    (j=1)
            [0, 3, 7, 4],  # left    (i=0)
            [1, 2, 6, 5],  # right   (i=1)
        ],
        dtype=int,
    )

    # 1) gather all faces: shape (n_voxels, 6, 4)
    all_faces = voxels[:, face_corners]  # → (n_voxels,6,4)
    all_faces = all_faces.reshape(-1, 4)  # → (n_voxels*6,4)

    # 2) compute sorted keys for uniqueness: shape (n_voxels*6,4)
    face_keys = np.sort(all_faces, axis=1)

    # 3) find unique keys and their counts
    keys, inverse, counts = np.unique(
        face_keys, axis=0, return_inverse=True, return_counts=True
    )

    # 4) boundary faces: those entries in all_faces whose key has count==1
    is_boundary = counts[inverse] == 1
    boundary_quads = all_faces[is_boundary]  # → (n_boundary_quads,4)

    # 5) triangulate quads: (v0,v1,v2) and (v0,v2,v3)
    tris1 = boundary_quads[:, [0, 1, 2]]
    tris2 = boundary_quads[:, [0, 2, 3]]
    tris = np.vstack([tris1, tris2])

    return nodes, tris
