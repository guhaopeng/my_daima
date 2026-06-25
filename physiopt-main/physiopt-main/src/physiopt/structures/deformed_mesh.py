from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import polyscope as ps

# Monkey-patch numpy to add ComplexWarning which was removed in newer versions
# but is still used by deepdish
if not hasattr(np, 'ComplexWarning'):
    np.ComplexWarning = RuntimeWarning

import deepdish as dd

from physiopt.vis.ui_config import GLOBAL_UI_CONFIG

# Keys to transfer to mesh when possible
LIGHT_NODE_QUANTITIES = ["u"]
COMPLETE_NODE_QUANTITIES = [
    "u",
    "forces",
    "force_mask",
    "bottom",
    "interpolatable_node",
    "part_node_weights",
]
ELEMENT_QUANTITIES = [
    "colors",
    "sigma",
    "rho",
    "mises",
    "E",
    "nu",
    "interpolatable_element",
]

KEYS_TO_SERIALIZE_DEFORMED = [
    # Mesh
    "original_vertices",
    "deformed_vertices",
    "faces",
    "u",
    "bottom",
    "forces",
    "force_mask",
    "E",
    "nu",
    "rho",
    "colors",
    "mises",
    "sigma",
    "part_node_weights",
    # Voxels
    "voxel_nodes",
    "voxel_elements",
    "voxel_u",
    "voxel_forces",
]

INTEGER_QUANTITIES = {"faces", "voxel_elements"}


@dataclass
class DeformedMesh:
    """Datastructure to export results of the simulation (with additional state information)"""

    # =======
    # VOXELS
    # =======

    voxel_nodes: np.ndarray | None = None
    voxel_elements: np.ndarray | None = None
    voxel_u: np.ndarray | None = None
    voxel_forces: np.ndarray | None = None

    # =====
    # MESH
    # =====

    original_vertices: np.ndarray | None = None
    deformed_vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    # Per-node data
    u: np.ndarray | None = None
    forces: np.ndarray | None = None
    force_mask: np.ndarray | None = None
    bottom: np.ndarray | None = None
    part_node_weights: np.ndarray | None = None

    # Per-element data
    E: np.ndarray = None
    nu: np.ndarray = None
    rho: np.ndarray = None
    colors: np.ndarray = None
    mises: np.ndarray = None
    sigma: np.ndarray = None

    # Debug
    interpolatable_node: np.ndarray = None
    interpolatable_element: np.ndarray = None

    @torch.no_grad()
    def serialize(self, half_precision: bool = False) -> Dict[str, Any]:
        data = {}
        for k in KEYS_TO_SERIALIZE_DEFORMED:
            if hasattr(self, k) and getattr(self, k) is not None:
                v = getattr(self, k)
                # If it is a tensor, convert it to numpy
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                if half_precision and k not in INTEGER_QUANTITIES:
                    v = v.astype(np.float16)
                data[k] = v
        return data

    @torch.no_grad()
    def deserialize(data: Dict[str, Any]) -> DeformedMesh:
        parsed_data = {}
        for k in KEYS_TO_SERIALIZE_DEFORMED:
            if k in data:
                parsed_data[k] = data[k]
                if k not in INTEGER_QUANTITIES:
                    parsed_data[k] = parsed_data[k].astype(np.float32)

        return DeformedMesh(**parsed_data)

    @torch.no_grad()
    def show(self, ranges: Dict[str, Tuple[float, float]] = {}):
        actual_vertices = (
            self.deformed_vertices
            if self.deformed_vertices is not None
            else self.original_vertices
        )
        if (
            actual_vertices is not None
            and self.faces is not None
            and GLOBAL_UI_CONFIG.display_deformed_mesh
        ):
            deformed_mesh = ps.register_surface_mesh(
                "deformed_mesh",
                actual_vertices + np.array(GLOBAL_UI_CONFIG.pos_deformed_mesh)[None, :],
                self.faces,
            )

            if self.mises is not None:
                kwargs = {"vminmax": ranges["mises"]} if "mises" in ranges else {}
                deformed_mesh.add_scalar_quantity("mises stress", self.mises, **kwargs)

            if self.sigma is not None:
                kwargs = {"vminmax": ranges["sigma"]} if "sigma" in ranges else {}
                deformed_mesh.add_scalar_quantity(
                    "norm_stress",
                    np.linalg.norm(self.sigma, axis=-1),
                    **kwargs,
                )

            if self.u is not None:
                kwargs = {"cmap": "coolwarm", "enabled": True}
                if "u" in ranges:
                    kwargs |= {"vminmax": ranges["u"]}
                deformed_mesh.add_scalar_quantity(
                    "u_norm",
                    np.linalg.norm(self.u, axis=1),
                    **kwargs,
                )

            if self.bottom is not None:
                deformed_mesh.add_scalar_quantity(
                    "bottom",
                    self.bottom,
                )

            if self.force_mask is not None:
                deformed_mesh.add_scalar_quantity(
                    "force_mask",
                    self.force_mask,
                )

            if self.colors is not None:
                deformed_mesh.add_color_quantity(
                    "colors",
                    self.colors,
                )

            if self.E is not None:
                deformed_mesh.add_scalar_quantity(
                    "E",
                    self.E,
                )

            if self.nu is not None:
                deformed_mesh.add_scalar_quantity(
                    "nu",
                    self.nu,
                )

            if self.interpolatable_node is not None:
                deformed_mesh.add_scalar_quantity(
                    "interpolatable_node", self.interpolatable_node
                )

            if self.interpolatable_element is not None:
                deformed_mesh.add_scalar_quantity(
                    "interpolatable_element", self.interpolatable_element
                )

            if self.part_node_weights is not None:
                for i_part in range(self.part_node_weights.shape[-1]):
                    deformed_mesh.add_scalar_quantity(
                        f"part_node_weights_{i_part}", self.part_node_weights[:, i_part]
                    )

    def save(self, path: str, half_precision: bool = False, compress: bool = False):
        if compress:
            np.savez_compressed(path, **self.serialize(half_precision=half_precision))
        else:
            np.savez(path, **self.serialize(half_precision=half_precision))

    @staticmethod
    def load(path: str):
        return DeformedMesh.deserialize(np.load(path))
