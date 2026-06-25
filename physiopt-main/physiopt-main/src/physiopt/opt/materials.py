from dataclasses import dataclass, field
from typing import List

from torchfem.materials import Material, IsotropicElasticity3D

from physiopt.vis.ui_utils import exp_slider

import polyscope.imgui as psim


GPA_UNIT: float = 1e9


# Placeholder to store material states without reinstantiating them
@dataclass
class MaterialConfig:
    """
    Class to hold material configuration.

    E: float
        Young's Modulus (GPa)
    nu: float
        Poisson Ratio (unitless)
    rho: float
        Density (kg/m^3)
    color: List[float]
        Color of the material (RGB)
    """

    # Young's Modulus
    # The E here is expressed in GPa, and converted when exporting
    E: float = 1
    # Poisson Ratio (unitless)
    nu: float = 0.3
    # Density (kg/m^3)
    rho: float = 650.0
    # Color (debug purposes)
    color: List[float] = field(default_factory=lambda: [0.7, 0.7, 0.7])

    current_material: int = 0

    def reset(self):
        self.E = DEFAULT_MATERIAL.E
        self.nu = DEFAULT_MATERIAL.nu
        self.current_material: int = 0
        self.color = DEFAULT_MATERIAL.color

    def get_fem_material(self) -> Material:
        import torch
        # Convert color to tensor if it's a list or tuple, to support vectorization
        color_tensor = self.color
        if isinstance(color_tensor, (list, tuple)):
            color_tensor = torch.tensor(color_tensor)
            
        return IsotropicElasticity3D(
            E=self.E * GPA_UNIT, nu=self.nu, rho=self.rho, color=color_tensor
        )

    def gui(self, material_idx: int = 0):
        material_changed = False
        material_now_custom = False
        psim.BeginGroup()
        if psim.Button(f"Reset##material_{material_idx}"):
            self.reset()
            material_changed |= True
            material_now_custom |= True
        psim.PushItemWidth(100)
        clicked, self.current_material = psim.Combo(
            f"mat##material_{material_idx}",
            self.current_material,
            MATERIAL_BANK_NAMES,
        )
        if clicked and self.current_material > 0:
            self.E = (
                MATERIAL_BANK[MATERIAL_BANK_NAMES[self.current_material]].E / GPA_UNIT
            )
            self.nu = MATERIAL_BANK[MATERIAL_BANK_NAMES[self.current_material]].nu
            self.rho = MATERIAL_BANK[
                MATERIAL_BANK_NAMES[self.current_material]
            ].rho.item()
            material_changed |= True
        clicked, self.color = psim.ColorEdit3(
            f"Color##material_{material_idx}",
            self.color,
            psim.ImGuiColorEditFlags_NoInputs | psim.ImGuiColorEditFlags_NoLabel,
        )
        if clicked:
            material_changed |= True

        psim.EndGroup()

        psim.SameLine()
        psim.BeginGroup()
        psim.PushItemWidth(220)
        # Young's Modulus
        # Removed power=10 as it's deprecated and not supported in newer imgui bindings
        # Use positional arguments: label, v, v_min, v_max
        clicked, self.E = psim.SliderFloat(
            f"E(GPa)##material_{material_idx}",
            self.E,
            0.01,
            100,
        )
        material_changed |= clicked
        material_now_custom |= clicked
        # Poisson Ratio
        clicked, self.nu = psim.SliderFloat(
            f"nu##material_{material_idx}", self.nu, v_min=0, v_max=1
        )
        material_changed |= clicked
        material_now_custom |= clicked
        clicked, self.rho = exp_slider(
            f"##material_{material_idx}", self.rho, v_min_exp=0, v_max_exp=5
        )
        psim.SameLine()
        psim.Text("rho(kg/m^3)")
        material_changed |= clicked
        material_now_custom |= clicked
        psim.PopItemWidth()
        psim.EndGroup()

        if material_now_custom:
            self.current_material = 0

        return material_changed


DEFAULT_MATERIAL = MaterialConfig()


def RANGE(a: float, b: float):
    return 0.5 * (a + b)


def GCM3_TO_KGM3(x: float):
    return 1000 * x


# ============================
# Common material definitions
# ============================


PLASTIC_BANK = {
    # Plastics
    # https://www.sonelastic.com/en/fundamentals/tables-of-materials-properties/polymers.html
    "nylon-6": IsotropicElasticity3D(
        E=RANGE(2.0, 4.0) * GPA_UNIT, nu=0.39, rho=GCM3_TO_KGM3(1.14)
    ),
    "pc": IsotropicElasticity3D(E=2.6 * GPA_UNIT, nu=0.36, rho=GCM3_TO_KGM3(1.20)),
    "pvc": IsotropicElasticity3D(
        E=RANGE(2.4, 4.1) * GPA_UNIT, nu=0.38, rho=GCM3_TO_KGM3(1.38)
    ),
    # "pe_hdpe": IsotropicElasticity3D(
    #     E=0.8 * GPA_UNIT,
    # ),
    # "pet": IsotropicElasticity3D(E=RANGE(2.0, 2.7) * GPA_UNIT),
    # "pp": IsotropicElasticity3D(E=RANGE(1.5, 2.0) * GPA_UNIT),
    # "pmma": IsotropicElasticity3D(E=RANGE(2.4, 3.4) * GPA_UNIT),
    "tpu": IsotropicElasticity3D(
        E=RANGE(0.01, 0.02) * GPA_UNIT, nu=0.38, rho=GCM3_TO_KGM3(1.5)
    ),
}

METAL_BANK = {
    # https://www.mit.edu/~6.777/matprops/aluminum.htm
    "aluminium": IsotropicElasticity3D(E=70 * GPA_UNIT, nu=0.33, rho=GCM3_TO_KGM3(2.7)),
    "bronze": IsotropicElasticity3D(
        E=RANGE(96.0, 120.0) * GPA_UNIT, nu=0.34, rho=GCM3_TO_KGM3(RANGE(7.4, 8.9))
    ),  # 96-120
    # https://www.matweb.com/search/DataSheet.aspx?MatGUID=654ca9c358264b5392d43315d8535b7d&ckck=1
    "iron": IsotropicElasticity3D(E=200 * GPA_UNIT, nu=0.291, rho=GCM3_TO_KGM3(7.86)),
}

WOOD_BANK = {
    # https://web.archive.org/web/20180720153345/https://www.fpl.fs.fed.us/documnts/fplgtr/fplgtr113/ch04.pdf
    # https://www.engineeringtoolbox.com/timber-mechanical-properties-d_1789.html
    "pine_wood": IsotropicElasticity3D(E=9 * GPA_UNIT, nu=0.350, rho=400),
    "oak_wood": IsotropicElasticity3D(E=11 * GPA_UNIT, nu=0.350, rho=620),
}

MISK_BANK = {
    # https://www.engineeringtoolbox.com/young-modulus-d_417.html
    # https://www.engineeringtoolbox.com/poissons-ratio-d_1224.html
    "glass": IsotropicElasticity3D(
        E=RANGE(50.0, 90.0) * GPA_UNIT, nu=RANGE(0.18, 0.3), rho=GCM3_TO_KGM3(2.2)
    ),  # 50-90
    "rubber": IsotropicElasticity3D(
        E=RANGE(0.01, 0.1) * GPA_UNIT, nu=RANGE(0.48, 0.5), rho=GCM3_TO_KGM3(1.34)
    ),
    "concrete": IsotropicElasticity3D(
        E=17 * GPA_UNIT, nu=RANGE(0.1, 0.2), rho=GCM3_TO_KGM3(2.4)
    ),
}


MATERIAL_BANK = {"custom": None} | PLASTIC_BANK | METAL_BANK | WOOD_BANK | MISK_BANK
MATERIAL_BANK_NAMES = list(MATERIAL_BANK.keys())
