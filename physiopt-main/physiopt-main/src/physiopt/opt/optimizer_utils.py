import torch
import torch.nn as nn


# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py#L316


# Old tensors are overwritten (optionally reset to 0)
# New tensors get zero gradients
def update_tensors(
    optimizer: torch.optim.Adam,
    old_tensors: torch.Tensor,
    new_tensors: torch.Tensor,
    reset_avg: bool = False,
) -> torch.Tensor:
    assert isinstance(optimizer, torch.optim.Adam)

    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            old_exp_avg = (
                torch.zeros_like(stored_state["exp_avg"])
                if reset_avg
                else stored_state["exp_avg"]
            )
            old_exp_avg_sq = (
                torch.zeros_like(stored_state["exp_avg_sq"])
                if reset_avg
                else stored_state["exp_avg_sq"]
            )

            stored_state["exp_avg"] = torch.cat(
                (old_exp_avg, torch.zeros_like(new_tensors)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (old_exp_avg_sq, torch.zeros_like(new_tensors)), dim=0
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((old_tensors, new_tensors), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state

        else:
            group["params"][0] = nn.Parameter(
                torch.cat((old_tensors, new_tensors), dim=0).requires_grad_(True)
            )
    optimizable_tensors = group["params"][0]
    return optimizable_tensors
