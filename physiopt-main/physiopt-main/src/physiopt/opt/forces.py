from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

import numpy as np

import polyscope.imgui as psim

from physiopt.vis.ui_utils import exp_slider, scientific_slider


@dataclass
class ForceConfig:
    """
    Class to hold external force configuration.

    magnitude: float
        Overall magnitude of the force.
    external_force: List[float]
        Direction of the force as a 3D vector.
    """

    # Magnitude is just used to increase the force by several orders of magnitude
    # whereas external_force is mostly for the direction
    magnitude: float = 1.0
    external_force: List[float] = field(default_factory=lambda: [0.0] * 3)

    def zero(self):
        self.external_force = deepcopy(ZERO_FORCES.external_force)
        self.magnitude = 1.0

    def get_total_force(self):
        return self.magnitude * np.array(self.external_force).astype(np.float32)

    def _normalize(self):
        self.external_force /= np.linalg.norm(self.external_force) + 1e-6

    def gui(self, force_idx: int = 0):
        update = False
        if psim.Button(f"Zero##force_{force_idx}"):
            self.zero()
            update |= True
        psim.SameLine()

        psim.BeginGroup()
        psim.PushItemWidth(250)
        clicked, self.magnitude = scientific_slider(
            f"##magnitude_force_{force_idx}",
            self.magnitude,
            v_min_exp=-2,
            v_max_exp=6,
        )
        psim.SameLine()
        psim.Text("magnitude")
        # Renormalize direction when changing magnitude!
        if clicked:
            self._normalize()

        update |= clicked
        clicked, self.external_force = psim.SliderFloat3(
            f"external##force_{force_idx}",
            self.external_force,
            v_min=-1.0,
            v_max=1.0,
            format=f"%.3f",
        )
        update |= clicked
        psim.PopItemWidth()
        psim.EndGroup()
        return update


ZERO_FORCES = ForceConfig(external_force=[0.0, 0.0, 0.0])
DEFAULT_FORCES_Y = ForceConfig(external_force=[0.0, 0.0, -500.0])
DEFAULT_FORCES_Z = ForceConfig(external_force=[0.0, 1000.0, 0.0])
