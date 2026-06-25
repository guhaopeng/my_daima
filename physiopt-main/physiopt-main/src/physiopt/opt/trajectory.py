from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Any
from copy import deepcopy
from dataclass_wizard.loader_selection import fromdict

import numpy as np
import torch

from physiopt.opt.optimizer_state import OptimizerConfig, OptimizationState

import polyscope as ps
import polyscope.imgui as psim
import deepdish as dd

# Allows to track version for backward compatibility
TRAJ_VERSION: str = "0.0.1"

# Keys over which we want to know [min, max]
# NB: it only makes sense for time-varying variables!
KEYS_TO_RANGE = [
    "u",
    "mises",
    "sigma",
]

import faulthandler

faulthandler.enable()

COND_KEYS = ["cond", "neg_cond", "z_s"]


@dataclass
class ConditionalPayload:

    cond: torch.Tensor = None
    neg_cond: torch.Tensor = None
    z_s: torch.Tensor = None

    @torch.no_grad()
    def serialize(self) -> Dict[str, Any]:
        data = {}
        for k in COND_KEYS:
            if hasattr(self, k) and getattr(self, k) is not None:
                v = getattr(self, k)
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                data[k] = v
        return data

    @torch.no_grad()
    def deserialize(data: Dict[str, Any]) -> ConditionalPayload:
        parsed_data = {}
        for k in COND_KEYS:
            if k in data:
                # WARNING: load as torch Tensor!
                parsed_data[k] = torch.from_numpy(data[k])

        return ConditionalPayload(**parsed_data)

    def to(self, device: str):
        for k in COND_KEYS:
            if hasattr(self, k):
                v = getattr(self, k)
                if v is not None:
                    setattr(self, k, v.to(device))


@dataclass
class Trajectory:
    """
    A trajectory stores all the OptimizerStates and the corresponding DenseOptimizerConfig
    This allows to store all parameters and potentially export trajectories for reproducibility.
    """

    # Stores each optimizer state
    states: List[OptimizationState]

    # Config file to track parameters used during sampling
    optimizer_config: OptimizerConfig

    # Stores the losses
    losses: Dict[str, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: [])
    )

    # Stores current slider `i_step`
    i_step: int = 0

    # Conditional payload
    cond_payload: ConditionalPayload = field(
        default_factory=lambda: ConditionalPayload()
    )

    # Update min/max values for each key
    def _update_ranges(self):
        self._ranges = {}

        for k in KEYS_TO_RANGE:
            # Make sure, there's at least one to create the entry!
            if not any([hasattr(state, k) for state in self.states]):
                continue
            min_k, max_k = float("inf"), -float("inf")
            for state in self.states:
                if not hasattr(state, k):
                    continue
                val: np.ndarray = getattr(state, k)
                if val is not None:
                    if len(val.shape) == 2:
                        min_k = min(min_k, np.linalg.norm(val, axis=1).min())
                        max_k = max(max_k, np.linalg.norm(val, axis=1).max())
                    elif len(val.shape) == 1:
                        min_k = min(min_k, val.min())
                        max_k = max(max_k, val.max())
                    else:
                        raise ValueError()

            self._ranges[k] = (min_k, max_k)

    @torch.no_grad()
    def serialize(self) -> Dict[str, Any]:
        data = {}
        data["states"] = [state.serialize() for state in self.states]
        data["optimizer_config"] = self.optimizer_config.to_yaml()
        losses = {
            k: np.array(v) if len(v) > 0 else np.empty(0)
            for k, v in self.losses.items()
        }
        data["losses"] = losses
        data["i_step"] = self.i_step
        data["cond_payload"] = self.cond_payload.serialize()
        return data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Trajectory:
        states = [OptimizationState.deserialize(state) for state in data["states"]]

        # This is a DIY fix to make previous configs backward compatible
        opt_config = yaml.load(data["optimizer_config"], yaml.Loader)
        if isinstance(opt_config["forces"], dict):
            opt_config["forces"] = [opt_config["forces"]]
        if isinstance(opt_config["init_forces"], dict):
            opt_config["init_forces"] = [opt_config["init_forces"]]

        config = fromdict(OptimizerConfig, opt_config)
        i_step = data["i_step"]
        losses = data["losses"]
        trajectory = Trajectory(None, config)
        trajectory.i_step = i_step
        trajectory.states = states
        trajectory.losses = defaultdict(lambda: [], losses)
        if "cond_payload" in data:
            trajectory.cond_payload = ConditionalPayload.deserialize(
                data["cond_payload"]
            )
        trajectory._update_ranges()
        return trajectory

    def __init__(
        self,
        state: OptimizationState,
        optimizer_config: OptimizerConfig,
    ) -> None:

        # Set a copy of the denoiser config
        self.optimizer_config = deepcopy(optimizer_config)

        # Set first state
        self.states = [state]

        # Set i_step to 0
        self.i_step = 0

        # Set losses history
        self.losses = defaultdict(lambda: [])

        # Update ranges
        self._update_ranges()

        # Init conditional payload
        self.cond_payload = ConditionalPayload()

    def add(self, state: OptimizationState):
        self.states.append(state)

        self.i_step = len(self.states) - 1
        self.post_update_i_step()

        # Update ranges
        self._update_ranges()

    @property
    def current_state(self):
        return self.states[self.i_step]

    @property
    def size(self):
        return len(self.states)

    def to(self, device: str):
        for state in self.states:
            state.to(device)

        self.cond_payload.to(device)

        return self

    def post_update_i_step(self):
        for idx, state in enumerate(self.states):
            if idx != self.i_step:
                self.states[idx] = state.to("cpu")


class TrajectoryHandler:
    """
    The Trajectory Handler stores all trajectories and allows to save/replay them + export them
    """

    trajectories: Dict[int, Trajectory]

    def __init__(self):
        self.counter: int = -1
        self.trajectories = {}

        self.current_idx: int = -1

    def add(self, trajectory: Trajectory, replace_current: bool = False):
        if replace_current:
            self.trajectories[self.current_idx] = trajectory
        else:
            self.counter += 1
            self.trajectories[self.counter] = trajectory
            self.current_idx = self.counter
            self.post_update_current_idx()

    def _get_prev_traj_idx(self, traj_idx) -> int:
        """
        If there isn't a previous one, it'll return the next one!
        """
        assert len(self.trajectories) > 1
        prev_map = {i: idx for i, idx in enumerate(self.trajectories.keys())}
        prev_invmap = {idx: i for i, idx in enumerate(self.trajectories.keys())}
        if prev_invmap[traj_idx] == 0:
            return prev_map[1]
        else:
            return prev_map[prev_invmap[traj_idx] - 1]

    def delete(self, traj_idx: int):
        if traj_idx not in self.trajectories:
            print(f"WARNING: tried to delete unknown trajectory {traj_idx:03d}!")
            return
        if len(self.trajectories) < 2:
            print(
                "WARNING: cannot remove this trajectory because there is only one trajectory!"
            )
            return

        self.current_idx = self._get_prev_traj_idx(traj_idx)
        self.post_update_current_idx()
        self.trajectories.pop(traj_idx)

    @property
    def current_trajectory(self):
        return self.trajectories[self.current_idx]

    def traj_name_selectable(
        self, label: str, idx: int, flags=psim.ImGuiSelectableFlags_None
    ) -> bool:

        clicked = psim.Selectable(
            label,
            self.current_idx == idx,
            flags=flags,
        )

        if clicked and idx != self.current_idx:
            self.current_idx = idx
            self.post_update_current_idx()

            return True
        return False

    def gui(self):

        TEXT_BASE_HEIGHT = psim.GetTextLineHeightWithSpacing()
        LIST_MAX_HEIGHT = 300

        update = False

        if psim.BeginTable(
            f"Trajectories##trajectory_handler",
            3,
            psim.ImGuiTableFlags_ScrollY,
            (
                0,
                min(TEXT_BASE_HEIGHT * (len(self.trajectories) + 1.5), LIST_MAX_HEIGHT),
            ),
        ):
            psim.TableSetupColumn(
                "Name",
                psim.ImGuiTableColumnFlags_WidthStretch,
                0.0,
            )
            psim.TableSetupColumn(
                "Steps",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableSetupColumn(
                "Delete",
                psim.ImGuiTableColumnFlags_WidthFixed
                | psim.ImGuiTableColumnFlags_NoHide,
                0.0,
            )
            psim.TableHeadersRow()

            to_delete = -1
            for traj_idx, traj in self.trajectories.items():

                psim.TableNextRow()

                # Display Name
                psim.TableNextColumn()
                update |= self.traj_name_selectable(f"{traj_idx:03d}_traj", traj_idx)

                # Number of Steps
                psim.TableNextColumn()
                psim.Text(f"{traj.size}")

                # Delete
                psim.TableNextColumn()
                if psim.Button(f"X##{traj_idx:03d}_traj_trajectory_handler"):
                    to_delete = traj_idx
                    update |= True

            if to_delete >= 0:
                self.delete(to_delete)

            psim.EndTable()

        return update

    def serialize(self):
        traj_dict = {}
        for traj_idx, traj in self.trajectories.items():
            traj_dict[traj_idx] = traj.serialize()

        data = {
            "trajectories": traj_dict,
            "version": TRAJ_VERSION,
        }
        return data

    @staticmethod
    def load_from_file(file_path: str) -> TrajectoryHandler:
        """Creates a TrajectoryHandler with the trajectories containes in the provided files"""
        handler = TrajectoryHandler()
        handler.add_from_file(file_path)
        return handler

    def add_from_file(self, file_path: str) -> None:
        """This will append trajectories to the current handler from a file."""
        data = dd.io.load(file_path)
        self.deserialize(data)

    def deserialize(self, data: Dict[str, Any]):

        for traj_data in data["trajectories"].values():
            trajectory = Trajectory.deserialize(traj_data)
            self.add(trajectory)

    def save(self, cpath: str | None = None):
        data = self.serialize()
        # cpath = self.cpath if cpath is None else cpath
        # os.makedirs(os.path.abspath(os.path.dirname(cpath)), exist_ok=True)
        dd.io.save(cpath, data)

    def post_update_current_idx(self):
        # Make sure the rest of the data points are all on the CPU:
        for traj_idx, traj in self.trajectories.items():
            if self.current_idx != traj_idx:
                self.trajectories[traj_idx] = traj.to("cpu")
