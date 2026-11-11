"""Affine-ReLU modules, Lipschitz targets, and paired training."""

from copy import deepcopy
from math import pi

import torch
from torch import nn
from torch.nn import functional as F

EVAL_BATCH_SIZE = 2048
REFERENCE_SCALE = 26.0 * pi


class LoopedReLU(nn.Module):
    """Apply stored ReLU blocks in a prescribed order."""

    def __init__(self, dimension: int, modules: int, width: int, perturbation: float):
        super().__init__()
        self.input_map = nn.Linear(2, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(modules))
        self.readout = nn.Linear(width, 1)
        with torch.no_grad():
            for block in self.blocks:
                block.weight.mul_(perturbation)
                block.weight.diagonal().add_(1.0)
                block.bias.zero_()

        # The original 3D runs drew the input map after the blocks and readout.
        if dimension == 3:
            self.input_map = nn.Linear(3, width)

    def forward(self, points: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
        state = self.input_map(points)
        for index in order:
            state = self.blocks[index](state).relu()
        return self.readout(state).squeeze(-1)


def configure_worker(threads: int):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def target_values(points: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    """Evaluate one signed pyramid per cell, with zero on cell boundaries."""
    cells_per_axis = torch.tensor(heights.shape)
    scaled_points = points * cells_per_axis
    cell_indices = torch.minimum(scaled_points.long(), cells_per_axis - 1)
    distance = (scaled_points - cell_indices - 0.5).abs().amax(dim=1)
    amplitudes = heights[tuple(cell_indices.unbind(dim=1))]
    return amplitudes * (1.0 - 2.0 * distance).clamp_min(0.0)


def make_splits(dimension, grid_side, target, settings):
    data_rng = torch.Generator().manual_seed(settings["data_seed"])
    count = settings["train_points"]
    train_points = torch.rand((count, 2), generator=data_rng)
    if dimension == 3:
        # Keep the original 2D coordinates and draw the third separately.
        third_coordinate = torch.rand((count, 1), generator=data_rng)
        train_points = torch.cat((train_points, third_coordinate), dim=1)

    axis = torch.linspace(0.0, 1.0, grid_side)
    grid_points = torch.cartesian_prod(*([axis] * dimension))
    heights = torch.tensor(target["heights"])
    gain, shift = target["gain"], target["shift"]
    train_values = gain * target_values(train_points, heights) + shift
    grid_values = gain * target_values(grid_points, heights) + shift
    return (train_points, train_values), (grid_points, grid_values)


@torch.inference_mode()
def measure_errors(model, order, split, normalizer):
    points, targets = split
    # The archived metrics first scale the residuals, then scale the reductions.
    residuals = torch.cat([
        (model(point_batch, order) - target_batch) / REFERENCE_SCALE
        for point_batch, target_batch in zip(
            points.split(EVAL_BATCH_SIZE), targets.split(EVAL_BATCH_SIZE)
        )
    ])
    if not torch.isfinite(residuals).all():
        raise FloatingPointError("Nonfinite evaluation residuals.")

    scale = REFERENCE_SCALE / normalizer
    return {
        "mse": residuals.double().square().mean().item() * scale**2,
        "max_abs_error": residuals.abs().max().item() * scale,
    }


def fit(model, order, train, grid, normalizer, batches, settings):
    points, targets = train
    cosine_steps = settings["cosine_steps"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=settings["final_learning_rate"],
    )
    checkpoints = settings["checkpoints"]
    history = {}
    for step, indices in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(points[indices], order), targets[indices])
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), settings["max_gradient_norm"], error_if_nonfinite=True,
        )
        optimizer.step()
        if step <= cosine_steps:
            scheduler.step()
        if step in checkpoints:
            train_errors = measure_errors(model, order, train, normalizer)
            metrics = {
                "train_mse": train_errors["mse"],
                "grid": measure_errors(model, order, grid, normalizer),
            }
            if step == checkpoints[-1]:
                metrics["train_max_abs_error"] = train_errors["max_abs_error"]
            history[str(step)] = metrics
    return history


def train_target(dimension, modules, calls, grid_side, target, seeds, settings):
    """Fit both orders from identical initial weights and batches per seed."""
    first_pass = tuple(range(modules))
    repeat_first = (0,) * (calls - modules)
    orders = {"A": repeat_first + first_pass, "B": first_pass + repeat_first}
    train, grid = make_splits(dimension, grid_side, target, settings)

    count = settings["train_points"]
    cosine_steps = settings["cosine_steps"]
    hold_steps = settings["checkpoints"][-1] - cosine_steps
    batch_size = settings["batch_size"]
    pairs = []
    for seed in seeds:
        torch.manual_seed(seed)
        initial = LoopedReLU(dimension, modules, settings["width"], settings["init_perturbation"])
        batch_rng = torch.Generator().manual_seed(seed + settings["batch_seed_offset"])
        # Draw batches separately for the cosine and constant-rate phases.
        batches = torch.cat((
            torch.randint(count, (cosine_steps, batch_size), generator=batch_rng),
            torch.randint(count, (hold_steps, batch_size), generator=batch_rng),
        ))
        pair = {"seed": seed}
        for name, order in orders.items():
            pair[name] = fit(
                deepcopy(initial), order, train, grid,
                target["error_normalizer"], batches, settings,
            )
        pairs.append(pair)
    return {"id": target["id"], "pairs": pairs}
