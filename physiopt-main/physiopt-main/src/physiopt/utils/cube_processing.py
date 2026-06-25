from typing import Tuple

import torch

from trellis.representations.mesh.cube2mesh import cube_corners

from physiopt.utils.phys_utils import grid_to_flat_idx


@torch.no_grad()
def add_voxels_with_sdf(
    fine_coords: torch.Tensor,
    fine_sdf: torch.Tensor,
    critical_ratio: float = 0.1,
    slat_res: int = 64,  # Resolution of the Structured Latents
    fine_res: int = 256,  # Resolution of the decoder
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute a ratio mask that will define which voxels to add
    # NB: the ratio is adjusted to the resolution to correlate with each cell's size
    ratio_mask = (fine_sdf >= 0.0) & (fine_sdf < (1.0 / fine_res * critical_ratio))

    # For all those that have a positive ratio mask,
    # Get the corresponding corner,
    to_add_corners = torch.argwhere(ratio_mask)
    # Get all the coords that there is to add
    cube_offsets = to_add_corners[:, 1]
    to_add_cubes = (
        fine_coords[to_add_corners[:, 0]][:, None, :]
        + (cube_corners.cuda() - 1)[None, ...].repeat((to_add_corners.shape[0], 1, 1))
        + cube_corners.cuda()[cube_offsets][:, None, :]
    )
    to_add_cubes = to_add_cubes.reshape(-1, 3)
    # Filter unique ones
    to_add_cubes = torch.unique(to_add_cubes, dim=0)

    # Filter within the grid
    def filter_in(coords: torch.Tensor, res: int = 256):
        mask = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < res)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < res)
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < res)
        )
        return coords[mask]

    to_add_cubes = filter_in(to_add_cubes, fine_res)
    # Only add what isn't already in
    new_mask = ~torch.isin(
        grid_to_flat_idx(to_add_cubes, fine_res),
        grid_to_flat_idx(fine_coords, fine_res),
    )
    to_add_cubes = to_add_cubes[new_mask]

    # Get corresponding cubes at 64 resolution (generator)
    factor = fine_res // slat_res
    to_add_cubes_slat = to_add_cubes // factor
    to_add_cubes_slat = torch.unique(to_add_cubes_slat, dim=0)

    return to_add_cubes, to_add_cubes_slat, ratio_mask
