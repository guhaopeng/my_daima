from typing import Tuple

import torch
from torch_scatter import scatter


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
    return indices[..., 0] * res**2 + indices[..., 1] * res + indices[..., 2]


# Generates the voxel volume given the limit bounds
# NB: we need this because hexaedra are wired in a specific way for positive Jacobians
def generate_hex(
    res: int = 64,
    limits=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
    device: str = "cuda",
    return_coords: bool = False,
):
    # Create nodes
    n1 = torch.linspace(limits[0][0], limits[0][1], res, device=device)
    n2 = torch.linspace(limits[1][0], limits[1][1], res, device=device)
    n3 = torch.linspace(limits[2][0], limits[2][1], res, device=device)
    n1, n2, n3 = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"))
    nodes = torch.stack([n1.ravel(), n2.ravel(), n3.ravel()], dim=1)

    # Create elements connecting nodes
    Is = torch.arange(res - 1, device=device)
    Js = torch.arange(res - 1, device=device)
    Ks = torch.arange(res - 1, device=device)

    Is, Js, Ks = torch.stack(torch.meshgrid(Is, Js, Ks, indexing="ij"))
    n0 = Is * res * res + Js * res + Ks  # [1, 1, 1]
    n1 = n0 + res * res  # [0, 1, 1]
    n2 = n1 + res  # [0, 0, 1]
    n3 = n0 + res
    n4 = n0 + 1
    n5 = n1 + 1
    n6 = n2 + 1
    n7 = n3 + 1

    elements = torch.stack([n0, n1, n2, n3, n4, n5, n6, n7], dim=-1).reshape(-1, 8)

    if return_coords:
        return nodes, elements, torch.stack([Is, Js, Ks], dim=-1).reshape(-1, 3)
    else:
        return nodes, elements


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


def generate_hex_at_coords(
    coords: torch.Tensor,
    res: int = 64,
    limits=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_nodes, grid_elements, grid_coords = generate_hex(
        res + 1, limits=limits, device=device, return_coords=True
    )
    flat_grid_coords = grid_to_flat_idx(grid_coords, res)
    flat_coords = grid_to_flat_idx(coords, res)
    input_to_grid = index_of_a_in_b(flat_coords, flat_grid_coords)

    selected_elements = grid_elements[input_to_grid]
    selected_node_indices, selected_elements = torch.unique(
        selected_elements, return_inverse=True
    )
    selected_nodes = grid_nodes[selected_node_indices]

    return selected_nodes, selected_elements


# Get Von-Mises Stress
# https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
def get_von_mises_stress(sigma: torch.Tensor):
    if sigma.shape[1] == 3:
        # 2D plane stress: sigma = [S11, S22, S12]
        return torch.sqrt(
            sigma[:, 0] ** 2
            - sigma[:, 0] * sigma[:, 1]
            + sigma[:, 1] ** 2
            + 3 * sigma[:, 2] ** 2
            + 1e-12
        )
    elif sigma.shape[1] == 6:
        # 3D stress: sigma = [S11, S22, S33, S12, S23, S13]
        S11 = sigma[:, 0]
        S22 = sigma[:, 1]
        S33 = sigma[:, 2]
        S12 = sigma[:, 3]
        S23 = sigma[:, 4]
        S13 = sigma[:, 5]
        return torch.sqrt(
            0.5 * ((S11 - S22) ** 2 + (S11 - S33) ** 2 + (S22 - S33) ** 2 + 6.0 * (S12 ** 2 + S23 ** 2 + S13 ** 2))
            + 1e-12
        )
    else:
        raise ValueError(f"Unsupported sigma shape for von Mises stress: {sigma.shape}")


def elements_to_nodes(
    elements: torch.Tensor,
    quantity: torch.Tensor,
    reduce: str = "max",
) -> torch.Tensor:
    """Transfers a quantity defined on elements to nodes"""
    assert len(quantity.shape) == 1

    dtype = quantity.dtype
    return scatter(
        quantity[:, None].repeat(1, elements.shape[1]).flatten(),
        elements.flatten(),
        reduce=reduce,
    ).to(dtype)


def nodes_to_elements(
    elements: torch.Tensor,
    quantity: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Transfers a quantity defined on nodes to elements"""
    assert len(quantity.shape) == 1

    dtype = quantity.dtype
    all_element_quantity = quantity[elements]

    if reduce == "mean":
        element_quantity = torch.mean(all_element_quantity, dim=-1)
    elif reduce == "max":
        element_quantity = torch.max(all_element_quantity, dim=-1).values
    else:
        raise ValueError()

    return element_quantity.to(dtype)
