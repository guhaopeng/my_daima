import numpy as np
import polyscope as ps
import torch

# -----------------------------
# LISTS
# -----------------------------


CUBE_VERTICES_LIST = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
]

CUBE_EDGES_LIST = [
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 0),
    (5, 1),
    (7, 3),
    (6, 2),
]

CUBE_TRIANGLES_LIST = [
    (4, 5, 7),
    (7, 6, 4),
    (3, 1, 0),
    (0, 2, 3),
    (1, 7, 5),
    (7, 1, 3),
    (4, 6, 0),
    (0, 6, 2),
    (7, 3, 6),
    (6, 3, 2),
    (4, 0, 5),
    (5, 0, 1),
]


# -----------------------------
# NUMPY
# -----------------------------

CUBE_VERTICES_NP = np.array(CUBE_VERTICES_LIST)
CUBE_EDGES_NP = np.array(CUBE_EDGES_LIST)
CUBE_TRIANGLES_NP = np.array(CUBE_TRIANGLES_LIST)

# -----------------------------
# TORCH
# -----------------------------

# N = 8
CUBE_VERTICES = torch.tensor(
    CUBE_VERTICES_LIST,
    device="cuda",
)

# N = 2 * 6 = 12
CUBE_TRIANGLES = torch.tensor(
    CUBE_TRIANGLES_LIST,
    dtype=torch.int,
    device="cuda",
)

CUBE_FACE_NEIGHBOR_OFFSETS = torch.tensor(
    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, -1],
        [1, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    device="cuda",
)

CUBE_EDGES = torch.tensor(
    CUBE_EDGES_LIST,
    dtype=torch.int,
    device="cuda",
)

# -----------------------------
# PS
# -----------------------------


def create_bbox(
    bbox_min=np.array([-1.0, -1.0, -1.0]),
    bbox_max=np.array([1.0, 1.0, 1.0]),
    edge_radius: float = 0.005,
    suffix: str = "",
    enabled: bool = True,
):
    cube_vertices = (bbox_max - bbox_min) * CUBE_VERTICES_NP + bbox_min

    bbox = ps.register_curve_network(
        f"bbox{suffix}",
        cube_vertices,
        CUBE_EDGES_NP,
        enabled=enabled,
        radius=edge_radius,
    )

    return bbox


@torch.no_grad()
def create_voxel_set(
    coords: torch.Tensor,
    voxel_res: int,
    bbox_min: float,
    bbox_max: float,
    name: str = "voxel_set",
    feat: torch.Tensor | None = None,
    offset: torch.Tensor | None = None,
) -> None:
    # self.voxels = coord_bbox_filter(self.voxels, self.res)

    vertex_offsets = torch.repeat_interleave(coords, 8, dim=0)
    vertices = CUBE_VERTICES.repeat((len(coords), 1)) + vertex_offsets

    # 8 for 8 vertices
    triangles_offsets = torch.repeat_interleave(
        (8 * torch.arange(len(coords), device="cuda"))[:, None].repeat((1, 3)),
        len(CUBE_TRIANGLES),
        dim=0,
    )
    faces = CUBE_TRIANGLES.repeat((len(coords), 1)) + triangles_offsets

    # Convert to bbox and apply transform
    vertices = (bbox_max - bbox_min) * (1.0 / float(voxel_res)) * vertices + bbox_min

    if offset is not None:
        vertices += offset

    ps_voxels = ps.register_surface_mesh(
        name,
        vertices.cpu().numpy(),
        faces.cpu().numpy(),
        # enabled=enabled,
    )
    ps_voxels.set_edge_width(0.0)

    if feat is not None:
        if len(feat.shape) == 1:
            ps_voxels.add_scalar_quantity(
                "feat",
                torch.repeat_interleave(feat, 12, dim=0).cpu().numpy(),
                enabled=True,
                defined_on="faces",
            )
        else:
            assert feat.shape[1] == 3
            ps_voxels.add_color_quantity(
                "feat",
                torch.repeat_interleave(feat, 12, dim=0).cpu().numpy(),
                enabled=True,
                defined_on="faces",
            )

    return ps_voxels


@torch.no_grad()
def create_voxel_set_np(
    coords: np.ndarray,
    voxel_res: int,
    bbox_min: float,
    bbox_max: float,
    name: str = "voxel_set",
    feat: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> None:
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
        # enabled=enabled,
    )
    ps_voxels.set_edge_width(0.0)

    if feat is not None:
        if len(feat.shape) == 1:
            ps_voxels.add_scalar_quantity(
                "feat",
                np.repeat(feat, 12, axis=0),
                enabled=True,
                defined_on="faces",
            )
        else:
            assert feat.shape[1] == 3
            ps_voxels.add_color_quantity(
                "feat",
                np.repeat(feat, 12, axis=0),
                enabled=True,
                defined_on="faces",
            )

    return ps_voxels
