import os
from typing import List, Tuple
from dataclasses import dataclass, field

from dataclass_wizard import YAMLWizard

DEFAULT_UI_CONFIG_PATH = "ui_config.yaml"
UI_CONFIG_PATH = os.environ.get("UI_CONFIG", DEFAULT_UI_CONFIG_PATH)


@dataclass
class UiConfig(YAMLWizard, key_transform="SNAKE"):
    """
    Configuration for UI settings.
    Unless specified, a ui_config.yaml file is automatically saved/loaded from the current working directory.
    """

    # ===================
    # WINDOW
    # ===================

    width: int = 1920
    height: int = 1080

    # =================
    # WHAT TO DISPLAY
    # ================

    # VOXELS
    display_ref_voxels: bool = False
    display_deformed_voxels: bool = True

    # Display forces (as vectors)
    display_forces: bool = True

    # MESHES
    display_ref_mesh: bool = True
    display_deformed_mesh: bool = True
    # display_fem_mesh: bool = True

    # GAUSSIANS
    display_gaussians: bool = False

    # FORCE SELECTION (VoxelSet)
    display_force_selection: bool = True

    # DEVELOPMENT MODE (with all features)
    dev_mode: bool = False

    # =================
    # WHERE TO DISPLAY
    # =================

    # VOXELS
    pos_ref_voxels: List[float] = field(default_factory=lambda: [1.2, 1.2, 0.0])
    pos_deformed_voxels: List[float] = field(default_factory=lambda: [1.2, 0.0, 1.2])

    pos_forces: List[float] = field(default_factory=lambda: [-1.2, 0.0, 1.2])

    # MESHES
    pos_ref_mesh: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pos_deformed_mesh: List[float] = field(default_factory=lambda: [1.2, 0.0, 0.0])
    # pos_fem_mesh: List[float] = field(default_factory=lambda: [-1.2, 0.0, 0.0])

    # GAUSSIANS
    pos_gaussians: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.2])

    # FORCE SELECTION (VoxelSet)
    pos_force_selection: List[float] = field(default_factory=lambda: [-1.2, 0.0, 0.0])


# Automatic loading/saving routines
if not os.path.exists(UI_CONFIG_PATH):
    GLOBAL_UI_CONFIG = UiConfig()
    # If the default ui config path does not exist.
    if UI_CONFIG_PATH == DEFAULT_UI_CONFIG_PATH:
        GLOBAL_UI_CONFIG.to_yaml_file(DEFAULT_UI_CONFIG_PATH)
else:
    GLOBAL_UI_CONFIG = UiConfig.from_yaml_file(UI_CONFIG_PATH)
