from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from dataclass_wizard import YAMLWizard
from copy import deepcopy

import numpy as np
import torch
import trimesh

from trellis.modules import sparse as sp
from trellis.representations.gaussian import Gaussian

from physiopt.vis.ui_config import GLOBAL_UI_CONFIG
from physiopt.opt.materials import MaterialConfig
from physiopt.opt.forces import ForceConfig, DEFAULT_FORCES_Y, DEFAULT_FORCES_Z
from physiopt.opt.boundary import BoundaryCondType

from physiopt.structures.deformed_mesh import (
    DeformedMesh,
    ELEMENT_QUANTITIES,
    LIGHT_NODE_QUANTITIES,
    COMPLETE_NODE_QUANTITIES,
)
from physiopt.utils.grid_utils import (
    grid_to_flat_idx,
    get_all_surroundings,
    get_indices,
    trilinear_interpolation_np,
    index_of_a_in_b,
    hex_to_tri_surface,
)
from physiopt.utils.chamfer_distance import find_nearest_point


import polyscope as ps
import polyscope.imgui as psim


DEFAULT_BACK_FORCE_MAGNITUDE = 50.0
DEFAULT_SEAT_FORCE_MAGNITUDE = 100.0
DEFAULT_GLOBAL_FORCES = [ForceConfig(external_force=[0.0, 0.0, 0.0])]
DEFAULT_CHAIR_FORCES = [
    ForceConfig(),
    ForceConfig(external_force=[0.0, 0.0, -DEFAULT_BACK_FORCE_MAGNITUDE]),
    ForceConfig(external_force=[0.0, -DEFAULT_SEAT_FORCE_MAGNITUDE, 0.0]),
    ForceConfig(),
]


@dataclass
class OptimizerConfig(YAMLWizard, key_transform="SNAKE"):

    # Voxel resolution used for simulation
    res: int = 32

    # ==================================
    # BOUNDARY CONDITIONS
    # NOTE: these are mostly default boundary conditions. They are directional and threshold-based.
    # ==================================

    # Threshold used to define which part is fixed
    bottom_fixed_threshold = 0.05

    boundary_cond: BoundaryCondType = BoundaryCondType.BOTTOM_Z

    # ==================================
    # OPTIMIZATION
    # ==================================

    # Number of optimization steps
    num_iters: int = 50
    lr: float = 0.01
    seed: int = 319

    # ===================================
    # BACKWARD PASS
    # NOTE: Ideally, do not modify these!
    # ===================================

    # SIMP density exponent
    # NOTE: In practice, the learned implicit outputs are already sparse,
    # so we can simply use an exponent of 1.0.
    p_exponent: float = 1.0
    # Lambda value for the volume regularization/penalization term (see Section 4.2 of the paper)
    volume_term: float = 1e6
    # Beta value for the sigmoid kernel (Equation 2 in the paper)
    stress_q_penalty: float = 0.5   # Penalty exponent for stress evaluation
    stress_p_norm: float = 8.0      # P-norm exponent for aggregation of von-Mises stresses
    mma_latent_shape_reg: float = 0.0
    sdf_filter_scale: float = 1.0

    # Radius of the hard latent-distance constraint used by latent-space MMA:
    # ||z - z0||^2 / radius^2 - 1 <= 0
    mma_latent_constraint_radius: float = 0.0

    # Target soft-volume ratio used by latent-space MMA:
    # V_current / V_initial <= volume_fraction_target
    volume_fraction_target: float = 1.0

    # Number of voxel layers used to dilate the initial active region when
    # building the fixed ROI for the soft-volume constraint.
    mma_volume_roi_margin: int = 3
    # Freeze the user-selected extra force region in absolute grid space so the
    # loading position no longer drifts with the evolving shape.
    freeze_force_region_in_space: bool = False

    # Number of coarse-voxel layers used to dilate the fixed force ROI after
    # the user selects it.
    fixed_force_roi_margin: int = 1

    # Keep material in the fixed force ROI so the load remains applied on a
    # persistent non-design region.
    preserve_force_region: bool = False
    
        # If > 0, collapse the frozen force region to at most this many coarse
    # voxel layers along the y axis while preserving the selected x/z extent.
    fixed_force_y_layers: int = 0


   
    occ_sdf_beta: float = (
        2.0  # Specifies how the SDF is converted to occupancy via a scaled sigmoid
    )
    # Inspired by Section 5 of "Microstructures to Control Elasticity in 3D Printing"
    # Density threshold (\rho_{min} in the paper)
    alpha_min: float = 1e-2

    # =========================
    # IM-Net + Deep SDF
    # =========================
    # Only keep one connected component when extracting the mesh
    # NOTE: this prevents small floating adversarial artifacts
    one_cc_only: bool = True

    # ==================================
    # MATERIAL
    # ==================================

    material: MaterialConfig = field(default_factory=lambda: MaterialConfig())

    # ==================================
    # FORCES
    # ==================================

    # Forces
    # NOTE: init_forces are the user-specified forces regardless of rescaling (see below)
    init_forces: List[ForceConfig] = field(
        default_factory=lambda: [DEFAULT_FORCES_Y, DEFAULT_FORCES_Z]
    )
    forces: List[ForceConfig] = field(
        default_factory=lambda: [DEFAULT_FORCES_Y, DEFAULT_FORCES_Z]
    )

    # ==================================
    # INPAINTING INTERVAL
    # ==================================
    # Every `inpainting_interval` steps, we reproject the latents on the data manifold by performing a small inpainting step.
    # The critical ratio is the threshold for the SDF to consider adding a neighboring voxel.
    # `inpainting_reset_adam` specifies whether to reset the Adam optimizer after inpainting.
    inpainting_interval: int = 25
    inpainting_critical_ratio: float = 0.1
    inpainting_reset_adam: bool = False

    # ==================================
    # AUTOMATIC FORCE RESCALING
    # ==================================
    # Autorescale the forces if the displacement is too big.
    # `autorescale_u_mean_target` is the maximum target displacement.
    # `autorescale_u_factor` is the factor by which to scale the forces.
    # `autorescale_u_mean_halt` is the threshold for the displacement to trigger an error.
    autorescale: bool = True
    autorescale_u_mean_halt: float = 0.5
    autorescale_u_mean_target: float = 0.1
    autorescale_u_factor: float = 0.6

    def forces_gui(self):
        if isinstance(self.forces, list):
            update = False
            if psim.Button("Add Extra Force"):
                self.forces.append(ForceConfig(external_force=[0.0, 0.0, 0.0]))
                update = True
            if len(self.forces) > 1:
                psim.SameLine()
                if psim.Button("Remove Last Extra Force"):
                    self.forces.pop()
                    update = True
            for i, force in enumerate(self.forces):
                psim.SeparatorText(
                    f"Global Force (i.e., gravity)" if i == 0 else f" Extra Force {i}"
                )
                update |= force.gui(i)
            return update
        else:
            raise ValueError()


# Specifies which keys in OptimizationState should be serialized with Optimization.serialize(...)
KEYS_TO_SERIALIZE = [
    "latent",
    "slat",
    "mesh_vertices",
    "mesh_faces",
    # NB: splats are deserialized on the fly (by the decoders!) = saves memory
    # Coordinates / Occ / Sdf
    "coarse_coords",
    "coarse_occ",
    # "fine_coords",
    # "fine_sdf",
    # Before solve
    "nodes",
    "elements",
    "bottom",
    "E",  # per-element
    "nu",  # per-element
    "rho",  # per-element
    "forces",
    "force_mask",
    # After solve
    "u",
    "mises",
    "sigma",
    "colors",  # per-element
    # Inpainting Specific
    # "to_add_cubes",
    # "to_add_cubes_slat",
    # "ratio_mask",
    # Sensitivities
    "sensitivity_density",
]

# Specifies which keys need to move back to the GPU when the state is made "current". Otherwise, things stay on the CPU (i.e., numpy)
# NOTE: this is mostly to avoid OOM
KEYS_TO_MOVE_ACROSS_DEVICES = ["latent", "slat", "splats"]


DEFAULT_AFTER_U_NORM_KWARGS = {"cmap": "coolwarm", "enabled": True}


@dataclass
class OptimizationState:
    """
    OptimizationState stores most variables at a given iteration step of the optimization.
    """

    # =====================
    # TRELLIS
    # =====================
    slat: sp.SparseTensor = None
    # NOTE: splats are not saved! (because too heavy)
    splats: Gaussian = None

    # ======================
    # DeepSDF / IM-Net / SDF
    # ======================
    latent: torch.Tensor = None

    # =====================
    # MESH
    # =====================
    # Current mesh
    mesh_vertices: np.ndarray = None
    mesh_faces: np.ndarray = None
    # Deformed vertices / deformations `u` after static Ku=f solve
    # NOTE: deformations are propagated from the voxel volume to the mesh vertices with interpolation. Simulation isn't done on the mesh itself!
    deformed_vertices: np.ndarray = None
    deformed_u: np.ndarray = None

    # =====================
    # VOXEL VOLUME
    # =====================
    # Coordinates and occupancy/density of the voxel volume
    coarse_coords: np.ndarray = None
    coarse_occ: np.ndarray = None
    # TRELLIS-only (output of the Flexicube decoder)
    # NOTE: these are only necessary to enable voxel growth via inpainting
    fine_coords: np.ndarray = None
    fine_sdf: np.ndarray = None

    # =====================
    # SOLVER INPUTS
    # =====================
    nodes: np.ndarray = None
    elements: np.ndarray = None
    bottom: np.ndarray = None  # BOUNDARY CONDITIONS: sorry for the confusing name
    E: np.ndarray = None
    nu: np.ndarray = None
    rho: np.ndarray = None  # MATERIAL DENSITY
    forces: np.ndarray = None  # NODAL forces
    force_mask: np.ndarray = (
        None  # Specifies which nodes received extra forces other than gravity
    )

    # =====================
    # SOLVER OUTPUTS
    # =====================
    u: np.ndarray = None  # NODAL deformations
    mises: np.ndarray = None  # ELEMENT von-mises stress
    sigma: np.ndarray = None  # ELEMENT stress

    # Sensitivities
    sensitivity_density: np.ndarray = None

    # =====================
    # INPAINTING
    # =====================
    to_add_cubes: np.ndarray = None
    to_add_cubes_slat: np.ndarray = None
    ratio_mask: np.ndarray = None

    # Non-saved
    # Computed when loading states

    _first_show: bool = False

    @torch.no_grad()
    def show(self, ranges: Dict[str, Tuple[float, float]] = {}):
        """
        Displays the current optimization state.
        """

        if self.nodes is not None and self.elements is not None:

            if GLOBAL_UI_CONFIG.display_ref_voxels:
                original_cubes = ps.register_volume_mesh(
                    "cubes",
                    (self.nodes + np.array(GLOBAL_UI_CONFIG.pos_ref_voxels)[None, :]).astype(np.float32),
                    hexes=self.elements.astype(np.int32),
                )
                original_cubes.add_scalar_quantity(
                    "occ", self.coarse_occ, defined_on="cells", enabled=True
                )
                original_cubes.add_scalar_quantity(
                    "bottom",
                    self.bottom,
                    defined_on="vertices",
                    # enabled=True
                )

            if self.forces is not None and GLOBAL_UI_CONFIG.display_forces:
                ps_pc_forces = ps.register_point_cloud(
                    "pc_forces",
                    self.nodes + np.array(GLOBAL_UI_CONFIG.pos_forces)[None, :],
                    radius=0,
                )
                ps_pc_forces.add_vector_quantity(
                    "forces", self.forces, enabled=True, radius=0.1
                )

            if self.u is not None and GLOBAL_UI_CONFIG.display_deformed_voxels:
                deformed_cubes = ps.register_volume_mesh(
                    "deformed_cubes",
                    ((self.nodes + self.u)
                    + np.array(GLOBAL_UI_CONFIG.pos_deformed_voxels)[None, :]).astype(np.float32),
                    hexes=self.elements.astype(np.int32),
                )

                kwargs = {"vminmax": ranges["mises"]} if "mises" in ranges else {}
                deformed_cubes.add_scalar_quantity(
                    "mises stress",
                    self.mises,
                    defined_on="cells",
                )

                kwargs = {"vminmax": ranges["sigma"]} if "sigma" in ranges else {}
                deformed_cubes.add_scalar_quantity(
                    "norm_stress",
                    np.linalg.norm(self.sigma, axis=-1),
                    defined_on="cells",
                    **kwargs,
                )

                kwargs = DEFAULT_AFTER_U_NORM_KWARGS if self._first_show else {}
                if "u" in ranges:
                    kwargs |= {"vminmax": ranges["u"]}
                deformed_cubes.add_scalar_quantity(
                    "u_norm",
                    np.linalg.norm(self.u, axis=1),
                    defined_on="vertices",
                    **kwargs,
                )

        if (
            self.mesh_vertices is not None
            and self.mesh_faces is not None
            and GLOBAL_UI_CONFIG.display_ref_mesh
        ):
            ps.register_surface_mesh(
                "mesh",
                (self.mesh_vertices + np.array(GLOBAL_UI_CONFIG.pos_ref_mesh)[None, :]).astype(np.float32),
                self.mesh_faces.astype(np.int32),
            )

        if (
            self.deformed_vertices is not None
            and self.mesh_faces is not None
            and GLOBAL_UI_CONFIG.display_deformed_mesh
        ):
            deformed_mesh = ps.register_surface_mesh(
                "deformed_mesh",
                (self.deformed_vertices
                + np.array(GLOBAL_UI_CONFIG.pos_deformed_mesh)[None, :]).astype(np.float32),
                self.mesh_faces.astype(np.int32),
            )

            if self.deformed_u is not None:
                kwargs = {"cmap": "coolwarm", "enabled": True}
                if "u" in ranges:
                    kwargs |= {"vminmax": ranges["u"]}
                deformed_mesh.add_scalar_quantity(
                    "u_norm",
                    np.linalg.norm(self.deformed_u, axis=1),
                    **kwargs,
                )

        self._first_show = True

    @torch.no_grad()
    def serialize(self) -> Dict[str, Any]:
        data = {}
        # No need to save vertices/faces if we have slats
        practical_keys_to_serialize = (
            set(KEYS_TO_SERIALIZE) - set({"mesh_vertices", "mesh_faces"})
            if hasattr(self, "slat") and getattr(self, "slat")
            else KEYS_TO_SERIALIZE
        )
        for k in practical_keys_to_serialize:
            if hasattr(self, k) and getattr(self, k) is not None:
                v = getattr(self, k)
                # If it is a tensor, convert it to numpy
                if isinstance(v, sp.SparseTensor):
                    v = v.serialize()
                elif isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                data[k] = v
        return data

    @torch.no_grad()
    def deserialize(data: Dict[str, Any]) -> OptimizationState:
        parsed_data = {}
        for k in KEYS_TO_SERIALIZE:
            if k in data:
                if k == "slat":
                    parsed_data[k] = sp.SparseTensor.deserialize(data[k])
                else:
                    parsed_data[k] = data[k]

        return OptimizationState(**parsed_data)

    # =========================
    # VOLUME -----> MESH utils
    # =========================

    def get_deformed_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        config: OptimizerConfig,
        mc_based: bool = False,
        # When set to full_mode, all values will be computed
        # NB: this is slower, but used to export the .dmesh file
        complete_mode: bool = False,
        node_quantities: List[str] = LIGHT_NODE_QUANTITIES,
        element_quantities: List[str] = [],
    ) -> DeformedMesh:
        if vertices is None or faces is None:
            return DeformedMesh(original_vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
            
        if complete_mode:
            node_quantities = COMPLETE_NODE_QUANTITIES
            element_quantities = ELEMENT_QUANTITIES

        deformed_mesh = DeformedMesh(original_vertices=vertices, faces=faces)

        practictal_node_keys = []
        for k in node_quantities:
            if not "interpolatable" in k and (
                not hasattr(self, k) or getattr(self, k) is None
            ):
                continue
            practictal_node_keys.append(k)

        practictal_node_outputs = self.query_quantities(
            queries=vertices,
            keys=practictal_node_keys,
            config=config,
            is_element_quantity=False,
            mc_based=mc_based,
        )
        for k, v in zip(practictal_node_keys, practictal_node_outputs):
            setattr(deformed_mesh, k, v)
            if k == "u":
                deformed_vertices = vertices + v
                deformed_mesh.deformed_vertices = deformed_vertices

        practictal_element_keys = []
        for k in element_quantities:
            if not "interpolatable" in k and (
                not hasattr(self, k) or getattr(self, k) is None
            ):
                continue
            practictal_element_keys.append(k)

        practictal_element_outputs = self.query_quantities(
            queries=vertices,
            keys=practictal_element_keys,
            config=config,
            is_element_quantity=True,
            mc_based=mc_based,
        )
        for k, v in zip(practictal_element_keys, practictal_element_outputs):
            setattr(deformed_mesh, k, v)

        # Voxel quantities
        deformed_mesh.voxel_nodes = self.nodes
        deformed_mesh.voxel_elements = self.elements
        deformed_mesh.voxel_u = self.u
        deformed_mesh.voxel_forces = self.forces

        return deformed_mesh

    @torch.no_grad()
    def query_quantities(
        self,
        queries: np.ndarray,
        keys: List[str],
        config: OptimizerConfig,
        is_element_quantity: bool,
        mc_based: bool = False,
    ) -> np.ndarray | None:
        
        if queries is None or len(queries) == 0:
            return [None for _ in keys]
        if self.nodes is None or self.elements is None:
            return [None for _ in keys]

        # This is needed for everthing that relies on Marching Cubes!
        res = config.res if not mc_based else config.res - 1
        queried_vals = [
            (
                torch.from_numpy(getattr(self, key)).float().cuda()
                if "interpolatable" not in key
                else None
            )
            for key in keys
        ]

        # NB: This is just a trick because Flexicube can allow vertices to go beyond [-0.5, 0.5]
        queries = torch.clip(
            torch.from_numpy(queries).float().cuda(), -0.5 + 1e-4, 0.5 - 1e-4
        )

        actual_grid_points = (
            torch.from_numpy(self.nodes).float().cuda()
            if not is_element_quantity
            else torch.from_numpy(self.nodes[self.elements].mean(1)).float().cuda()
        )

        grid_indices, _ = get_indices(
            actual_grid_points, res, is_element_quantity=is_element_quantity
        )
        # NB: we had +1 because things can be on the edges
        flat_grid_indices = grid_to_flat_idx(grid_indices, res + 1)

        # =============================

        query_indices, local_coords = get_indices(queries, res, is_element_quantity)
        surrouding_indices = get_all_surroundings(query_indices)
        flat_surrounding_indices = grid_to_flat_idx(surrouding_indices, res + 1)

        valid_surrounding_queries = torch.isin(
            flat_surrounding_indices.flatten(),
            flat_grid_indices,
        ).reshape(-1, 8)
        interpolatable_mask = valid_surrounding_queries.sum(1) == 8

        # Create buffer to write everything!
        outputs = []
        for queried_val in queried_vals:
            output = torch.zeros(
                (
                    (
                        interpolatable_mask.shape[0],
                        queried_val.shape[1],
                    )
                    if queried_val is not None and len(queried_val.shape) == 2
                    else interpolatable_mask.shape[0]
                ),
            ).cuda()
            outputs.append(output)

        # =========================
        # INTERPOLATION
        # =========================

        assert local_coords.min() >= 0 and local_coords.max() <= 1.0

        # Recover the corner indices
        corner_indices = index_of_a_in_b(
            flat_surrounding_indices[interpolatable_mask].flatten(),
            flat_grid_indices,
        ).reshape(-1, 8)

        # Then, the values
        for output, queried_val in zip(outputs, queried_vals):
            if queried_val is not None:
                corner_values = queried_val[corner_indices]

                if len(queried_val.shape) != 2:
                    corner_values = corner_values[..., None]

                interpolated_values = trilinear_interpolation_np(
                    local_coords[interpolatable_mask],
                    corner_values,
                )
                if len(queried_val.shape) != 2:
                    interpolated_values = interpolated_values.flatten()

                output[interpolatable_mask] = interpolated_values

        # ================================================
        # OUTPUT (for queries that cannot be interpolated)
        # NB: this can be due to num precision or Flexicube which can output vertices outside
        # ================================================

        if torch.any(~interpolatable_mask):
            for output, queried_val in zip(outputs, queried_vals):
                if queried_val is None:
                    output[~interpolatable_mask] = 1.0
                else:
                    # For elements, just look at the NN
                    if is_element_quantity:
                        nearest_idx = find_nearest_point(
                            queries[~interpolatable_mask], actual_grid_points
                        )
                        output[~interpolatable_mask] = queried_val[nearest_idx]
                    # For nodes, barycentric interpolation of the closest point on the mesh
                    else:
                        # Create a mesh with the nodes and elements
                        tri_cubes_vertices, tri_cubes_faces = hex_to_tri_surface(
                            self.nodes, self.elements
                        )
                        mesh = trimesh.Trimesh(
                            vertices=tri_cubes_vertices,
                            faces=tri_cubes_faces,
                            process=False,
                        )
                        # Query nearest on surface
                        closest, _, face_id = mesh.nearest.on_surface(
                            queries[~interpolatable_mask].float().cpu().numpy()
                        )
                        # Extract the triangle corners for each hit face
                        tris = mesh.triangles[face_id]
                        fv = mesh.faces[face_id]
                        # Compute barycentric coordinates
                        bary = (
                            torch.from_numpy(
                                trimesh.triangles.points_to_barycentric(tris, closest)
                            )
                            .float()
                            .cuda()
                        )
                        # Then, interpolate
                        corner_attrs = queried_val[torch.from_numpy(fv).cuda()]
                        interp = torch.sum(
                            corner_attrs
                            * (
                                bary[:, :, None]
                                if len(queried_val.shape) != 1
                                else bary
                            ),
                            axis=1,
                        )
                        # Write:
                        output[~interpolatable_mask] = interp

        return [output.cpu().numpy() for output in outputs]

    # =========================
    # MEMORY
    # =========================

    def print_memory_summary(self):
        fmt = "{:<20} {:>10} {:>8}"
        print(fmt.format("name", "type", "size(MB)"))
        for k in KEYS_TO_SERIALIZE:
            if hasattr(self, k) and getattr(self, k) is not None:
                v = getattr(self, k)
                # If it is a tensor, convert it to numpy
                if isinstance(v, sp.SparseTensor):
                    print(
                        fmt.format(
                            k,
                            str(v.dtype),
                            b_to_mb(
                                tensor_to_bytes(v.coords) + tensor_to_bytes(v.feats)
                            ),
                        )
                    )
                elif isinstance(v, torch.Tensor):
                    print(fmt.format(k, str(v.dtype), b_to_mb(tensor_to_bytes(v))))
                elif isinstance(v, np.ndarray):
                    print(fmt.format(k, str(v.dtype), b_to_mb(v.nbytes)))

    # =========================
    # MOVE TO GPU
    # =========================

    def to(self, device: str):
        for k in KEYS_TO_MOVE_ACROSS_DEVICES:
            if hasattr(self, k) and getattr(self, k) is not None:
                v = getattr(self, k)
                # We don't move np.ndarray! Only the types below
                if isinstance(v, (torch.Tensor, sp.SparseTensor, Gaussian)):
                    setattr(v, k, v.to(device))
        return self


def tensor_to_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


def b_to_mb(size: int) -> int:
    return size / (1024**2)
