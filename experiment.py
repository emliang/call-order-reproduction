"""Training and evaluation for the matched ReLU call-order experiment."""

from copy import deepcopy
from math import pi

import torch
from torch import nn
from torch.nn import functional as F


REFERENCE_SCALE = 26.0 * pi


class LoopedReLU(nn.Module):
    def __init__(self, dimension, modules, width, perturbation):
        super().__init__()
        self.input_map = nn.Linear(2, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(modules))
        self.readout = nn.Linear(width, 1)
        with torch.no_grad():
            for block in self.blocks:
                block.weight.mul_(perturbation)
                block.weight.diagonal().add_(1.0)
                block.bias.zero_()
        # The original 3D fits replace the input map after the other draws.
        if dimension == 3:
            self.input_map = nn.Linear(3, width)

    def forward(self, points, order):
        state = self.input_map(points)
        for index in order:
            state = self.blocks[index](state).relu()
        return self.readout(state).squeeze(-1)


def configure_worker(threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def target_values(points, heights, dimension):
    if dimension == 2:
        cells_per_side = heights.shape[0]
        scaled = cells_per_side * points
        cell = scaled.long().clamp(max=cells_per_side - 1)
        amplitude = heights[cell[:, 0], cell[:, 1]]
    else:
        shape = torch.tensor(heights.shape)
        scaled = points * shape
        cell = torch.minimum(scaled.long(), shape - 1)
        amplitude = heights[tuple(cell.unbind(dim=1))]
    distance = (scaled - cell - 0.5).abs().amax(dim=1)
    return amplitude * (1.0 - 2.0 * distance).clamp_min(0.0)


@torch.inference_mode()
def errors(model, order, split, normalizer):
    points, targets = split
    # Preserve the two-step residual scaling of the recorded metrics.
    residuals = torch.cat([
        (model(x, order) - y) / REFERENCE_SCALE
        for x, y in zip(points.split(2048), targets.split(2048))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["initial_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["cosine_steps"], eta_min=settings["final_learning_rate"],
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
        if step <= settings["cosine_steps"]:
            scheduler.step()
        if step in checkpoints:
            train_errors = errors(model, order, train, normalizer)
            entry = {
                "train_mse": train_errors["mse"],
                "grid": errors(model, order, grid, normalizer),
            }
            if step == checkpoints[-1]:
                entry["train_max_abs_error"] = train_errors["max_abs_error"]
            history[str(step)] = entry
    return history


def train_target(dimension, modules, calls, problem, target, seeds, settings):
    extra = calls - modules
    first_pass = tuple(range(modules))
    orders = {"A": (0,) * extra + first_pass, "B": first_pass + (0,) * extra}
    count = settings["train_points"]
    data_rng = torch.Generator().manual_seed(settings["data_seed"])
    train_x = torch.rand((count, 2), generator=data_rng)
    if dimension == 3:
        train_x = torch.cat((train_x, torch.rand((count, 1), generator=data_rng)), dim=1)
    axis = torch.linspace(0.0, 1.0, problem["grid_side"])
    grid_x = torch.cartesian_prod(*([axis] * dimension))
    heights = torch.tensor(target["heights"])
    gain, shift = target["gain"], target["shift"]
    train = train_x, gain * target_values(train_x, heights, dimension) + shift
    grid = grid_x, gain * target_values(grid_x, heights, dimension) + shift

    cosine_steps = settings["cosine_steps"]
    hold_steps = settings["checkpoints"][-1] - cosine_steps
    batch_size = settings["batch_size"]
    pairs = []
    for seed in seeds:
        torch.manual_seed(seed)
        initial = LoopedReLU(dimension, modules, settings["width"], settings["init_perturbation"])
        batch_rng = torch.Generator().manual_seed(seed + settings["batch_seed_offset"])
        batches = torch.cat((
            torch.randint(count, (cosine_steps, batch_size), generator=batch_rng),
            torch.randint(count, (hold_steps, batch_size), generator=batch_rng),
        ))
        pair = {"seed": seed}
        for name, order in orders.items():
            pair[name] = fit(deepcopy(initial), order, train, grid,
                             target["error_normalizer"], batches, settings)
        pairs.append(pair)
    return {"id": target["id"], "pairs": pairs}
