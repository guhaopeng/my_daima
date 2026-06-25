from dataclasses import dataclass
from enum import Enum

import torch


class BoundaryCondType(Enum):

    BOTTOM_X = "bottom_x"
    TOP_X = "top_x"
    BOTTOM_Y = "bottom_y"
    TOP_Y = "top_y"
    BOTTOM_Z = "bottom_z"
    TOP_Z = "top_z"


BOUNDARY_TYPE_MAP = {x: i for i, x in enumerate(BoundaryCondType)}
BOUNDARY_TYPE_INVMAP = {i: x for i, x in enumerate(BoundaryCondType)}
BOUNDARY_TYPE_NAMEMAP = {x.value: x for x in BoundaryCondType}
BOUNDARY_NAMES = {x.value for x in BoundaryCondType}


@dataclass
class BoundaryCondData:
    """Utility class to hold boundary condition data for a given set of nodes."""

    bottom_x: torch.Tensor
    top_x: torch.Tensor
    bottom_y: torch.Tensor
    top_y: torch.Tensor
    bottom_z: torch.Tensor
    top_z: torch.Tensor

    def __init__(self, nodes: torch.Tensor):
        self.bottom_x = torch.min(nodes[:, 0])
        self.top_x = torch.max(nodes[:, 0])
        self.bottom_y = torch.min(nodes[:, 1])
        self.top_y = torch.max(nodes[:, 1])
        self.bottom_z = torch.min(nodes[:, 2])
        self.top_z = torch.max(nodes[:, 2])

    def check(
        self, nodes: torch.Tensor, boundary_cond: BoundaryCondType, threshold: float
    ) -> torch.Tensor:
        if boundary_cond == BoundaryCondType.BOTTOM_X:
            return torch.isclose(
                nodes[:, 0],
                self.bottom_x,
                atol=threshold,
            )
        elif boundary_cond == BoundaryCondType.TOP_X:
            return torch.isclose(
                nodes[:, 0],
                self.top_x,
                atol=threshold,
            )
        elif boundary_cond == BoundaryCondType.BOTTOM_Y:
            return torch.isclose(
                nodes[:, 1],
                self.bottom_y,
                atol=threshold,
            )
        elif boundary_cond == BoundaryCondType.TOP_Y:
            return torch.isclose(
                nodes[:, 1],
                self.top_y,
                atol=threshold,
            )
        elif boundary_cond == BoundaryCondType.BOTTOM_Z:
            return torch.isclose(
                nodes[:, 2],
                self.bottom_z,
                atol=threshold,
            )
        elif boundary_cond == BoundaryCondType.TOP_Z:
            return torch.isclose(
                nodes[:, 2],
                self.top_z,
                atol=threshold,
            )
        else:
            raise NotImplementedError()


def get_directional_boundary_conditions(
    nodes: torch.Tensor, direction: str = "bottom_z", threshold: float = 0.05
):
    """
    Will return a mask for nodes below the provided `threshold` in the given direction.
    """
    if direction not in BOUNDARY_NAMES:
        raise ValueError(
            f"{direction} is not a valid direction. Use any of: {BOUNDARY_NAMES}"
        )

    bdata = BoundaryCondData(nodes)
    bcond = BOUNDARY_TYPE_NAMEMAP[direction]
    return bdata.check(nodes, bcond, threshold)
