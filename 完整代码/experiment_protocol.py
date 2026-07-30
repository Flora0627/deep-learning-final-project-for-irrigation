"""Frozen protocol for the final irrigation experiments."""

from __future__ import annotations

from itertools import product
from typing import Any

PROTOCOL_VERSION = "formal_ppo_final_v1"

TRAIN_SIMYEARS = tuple(range(1, 13))
DEVELOPMENT_SIMYEARS = tuple(range(13, 31))
FINAL_HOLDOUT_SIMYEARS = tuple(range(31, 41))

PPO_SEEDS = (0, 1, 2)

WATER_BUDGETS_MM = {
    "moderate": 1800.0,
    "abundant": 2400.0,
}

TREATMENTS = {
    "no_forecast": {
        "use_forecast": False,
        "forecast_noise_std_mm": 0.0,
    },
    "perfect_forecast": {
        "use_forecast": True,
        "forecast_noise_std_mm": 0.0,
    },
    "noisy_forecast": {
        "use_forecast": True,
        "forecast_noise_std_mm": 2.0,
    },
}

IRRIGATION_COST_PER_FIELD = 0.16
INVALID_ACTION_PENALTY = 2.0
FORECAST_NOISE_SEED = 73_001
SELECTION_SEED = 2_026
EVALUATION_RESET_SEED = 92_001
SHUFFLE_SEED = 84_271

TARGET_TRAINING_STEPS = 150_000
TRAIN_BATCH_SIZE = 2_400
MINIBATCH_SIZE = 240
ROLLOUT_FRAGMENT_LENGTH = 120
CHECKPOINT_EVAL_INTERVAL_ITERATIONS = 2
EXPECTED_ACTUAL_STOP_STEPS = 151_200

PPO_PARAMS = {
    "gamma": 1.0,
    "lambda": 0.95,
    "lr": 3e-4,
    "clip_param": 0.2,
    "entropy_coeff": [
        [0, 0.03],
        [75_000, 0.01],
        [150_000, 0.003],
    ],
    "vf_loss_coeff": 0.5,
    "grad_clip": 0.5,
    "num_epochs": 10,
    "shuffle_batch_per_epoch": True,
}

MODEL_HIDDENS = [64, 64, 64]
VF_SHARE_LAYERS = False
USE_LSTM = False


def formal_model_matrix() -> tuple[dict[str, Any], ...]:
    """Return the frozen 3 × 2 × 3 model matrix in stable order."""

    return tuple(
        {
            "treatment": treatment,
            "budget_name": budget_name,
            "budget_mm": WATER_BUDGETS_MM[budget_name],
            "ppo_seed": seed,
        }
        for seed, budget_name, treatment in product(
            PPO_SEEDS, WATER_BUDGETS_MM, TREATMENTS
        )
    )


def validate_protocol() -> bool:
    """Fail fast if any frozen experimental invariant has changed."""

    year_sets = [
        set(TRAIN_SIMYEARS),
        set(DEVELOPMENT_SIMYEARS),
        set(FINAL_HOLDOUT_SIMYEARS),
    ]
    assert not (year_sets[0] & year_sets[1])
    assert not (year_sets[0] & year_sets[2])
    assert not (year_sets[1] & year_sets[2])
    assert FINAL_HOLDOUT_SIMYEARS == tuple(range(31, 41))
    assert len(formal_model_matrix()) == 18
    assert len(
        {
            (row["treatment"], row["budget_name"], row["ppo_seed"])
            for row in formal_model_matrix()
        }
    ) == 18
    assert TREATMENTS["noisy_forecast"]["forecast_noise_std_mm"] == 2.0
    assert IRRIGATION_COST_PER_FIELD == 0.16
    assert TRAIN_BATCH_SIZE == 2_400
    assert MINIBATCH_SIZE == 240
    assert PPO_PARAMS["num_epochs"] == 10
    assert PPO_PARAMS["lr"] == 3e-4
    assert PPO_PARAMS["entropy_coeff"] == [
        [0, 0.03],
        [75_000, 0.01],
        [150_000, 0.003],
    ]
    assert EXPECTED_ACTUAL_STOP_STEPS == (
        (TARGET_TRAINING_STEPS + TRAIN_BATCH_SIZE - 1)
        // TRAIN_BATCH_SIZE
        * TRAIN_BATCH_SIZE
    )
    return True


validate_protocol()
