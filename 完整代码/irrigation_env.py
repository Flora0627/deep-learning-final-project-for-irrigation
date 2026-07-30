"""Gymnasium environment for rainfall-forecast irrigation experiments.

The model is smaller than AquaCrop: it represents a transparent
daily root-zone water balance for three heterogeneous fields.  It contains no
crop-yield model or yield term in the reward.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


ACTION_TO_FIELDS = np.asarray(
    [
        [0, 0, 0],  # 0: no irrigation
        [1, 0, 0],  # 1: field 1
        [0, 1, 0],  # 2: field 2
        [0, 0, 1],  # 3: field 3
        [1, 1, 0],  # 4: fields 1 + 2
        [1, 0, 1],  # 5: fields 1 + 3
        [0, 1, 1],  # 6: fields 2 + 3
    ],
    dtype=np.int8,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "episode_days": 120,
    "selection_seed": 0,
    # “weather_seeds” is retained for compatibility, but each value is now a
    # LARS-WG “simyear” rather than a random-number-generator seed.
    "weather_seeds": tuple(range(1, 41)),
    "processed_weather_file": "CPWG_processed.csv",
    # CPWG uses one-based Julian days: 121 is May 1.
    "weather_start_jday": 121,
    # Forecast treatment. A zero standard deviation is a perfect forecast.
    "use_forecast": True,
    "forecast_horizon": 3,
    "forecast_observation_encoding": "daily",
    "forecast_noise_std_mm": 0.0,
    "forecast_noise_seed": 73_001,
    # Shared seasonal resource. Formal experiments pass 1800 or 2400 mm via
    # external configuration; 1800 mm is the default formal condition.
    "fixed_irrigation_mm": 10.0,
    "seasonal_water_budget_mm": 1800.0,
    "invalid_action_penalty": 2.0,
    # Field 1 retains water well, field 2 is intermediate, and field 3 loses
    # water faster. All moisture quantities below are capacity fractions.
    "field_capacity_mm": [180.0, 145.0, 115.0],
    "initial_moisture": [0.66, 0.60, 0.68],
    "irrigation_threshold": [0.52, 0.47, 0.56],
    "target_moisture": [0.70, 0.65, 0.74],
    "upper_moisture_limit": [0.88, 0.84, 0.90],
    "wilting_moisture": [0.24, 0.22, 0.28],
    "rain_infiltration_efficiency": [0.92, 0.85, 0.78],
    "irrigation_efficiency": [0.92, 0.88, 0.84],
    "et_multiplier": [0.85, 1.00, 1.20],
    "daily_water_loss_fraction": [0.0015, 0.0030, 0.0050],
    # Reward: moisture quality - dry stress - overwatering - water cost.
    "moisture_quality_weight": 2.0,
    "dry_stress_weight": 4.0,
    "overwatering_weight": 2.0,
    "irrigation_cost_per_field": 0.16,
    "rain_observation_scale_mm": 30.0,
    "eto_observation_scale_mm": 8.0,
}

PROCESSED_WEATHER_COLUMNS = ("simyear", "jday", "rain_mm", "eto_mm")
WEATHER_PROCESSOR = "precomputed_aquacropgym_faopm_csv"


def make_default_config(**overrides: Any) -> dict[str, Any]:
    """Return an independent configuration dictionary."""

    config = deepcopy(DEFAULT_CONFIG)
    config.update(overrides)
    return config


@dataclass(frozen=True)
class WeatherSequence:
    """One LARS-WG simulated-year weather sequence.

    ``seed`` is retained for backward compatibility and stores ``simyear``.
    """

    seed: int
    rain_mm: np.ndarray
    eto_mm: np.ndarray

    @property
    def fingerprint(self) -> str:
        payload = np.concatenate((self.rain_mm, self.eto_mm)).astype("<f8")
        return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


class IrrigationEnv(gym.Env[np.ndarray, int]):
    """Three-field daily irrigation environment.

    Every treatment uses the same ten-dimensional observation::

        [soil_1, soil_2, soil_3, previous_day_observed_rain, current_eto,
         remaining_budget, day_of_season, rain_t, rain_t_plus_1,
         rain_t_plus_2]

    The final three entries are zero-masked for the no-forecast treatment.
    Rainfall and ETo are divided by the configured observation scales; other
    entries are fractions. Day-0 previous rainfall is zero.

    Actual weather is loaded once during initialization from a precomputed
    LARS-WG CSV window containing 120 decision days plus two trailing days
    needed by the final three-day rainfall forecast. ``weather_seed`` remains
    the public compatibility name for the selected ``simyear``.

    The within-day order is: observe state and forecast, select and apply an
    irrigation action, realize day-t rainfall and ETo, update soil moisture,
    then advance to the next day. Actual day-t rainfall is never revealed in
    the non-forecast observation before the action.

    The action space is exactly ``Discrete(7)`` and follows ACTION_TO_FIELDS.
    A two-field action is rejected in full if the shared budget cannot fund
    both fixed applications; partial or order-dependent allocation is avoided.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__()
        merged = make_default_config()
        if config is not None:
            merged.update(dict(config))
        self.config = merged

        self.episode_days = int(merged["episode_days"])
        self.forecast_horizon = int(merged["forecast_horizon"])
        self.forecast_observation_encoding = str(
            merged["forecast_observation_encoding"]
        )
        self.use_forecast = bool(merged["use_forecast"])
        self.forecast_noise_std_mm = float(merged["forecast_noise_std_mm"])
        self.forecast_noise_seed = int(merged["forecast_noise_seed"])
        self.fixed_irrigation_mm = float(merged["fixed_irrigation_mm"])
        self.seasonal_water_budget_mm = float(merged["seasonal_water_budget_mm"])
        self.invalid_action_penalty = float(merged["invalid_action_penalty"])
        self.rain_scale = float(merged["rain_observation_scale_mm"])
        self.eto_scale = float(merged["eto_observation_scale_mm"])
        self.weather_start_jday = int(merged["weather_start_jday"])

        finite_scalars = {
            "forecast_noise_std_mm": self.forecast_noise_std_mm,
            "fixed_irrigation_mm": self.fixed_irrigation_mm,
            "seasonal_water_budget_mm": self.seasonal_water_budget_mm,
            "invalid_action_penalty": self.invalid_action_penalty,
            "rain_observation_scale_mm": self.rain_scale,
            "eto_observation_scale_mm": self.eto_scale,
        }

        for name, value in finite_scalars.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.episode_days != 120:
            raise ValueError("This experiment requires episode_days to be exactly 120")
        if self.forecast_horizon != 3:
            raise ValueError("This experiment requires a three-day forecast horizon")
        if self.forecast_observation_encoding != "daily":
            raise ValueError("forecast_observation_encoding must be daily")
        if self.forecast_noise_std_mm < 0:
            raise ValueError("forecast_noise_std_mm cannot be negative")
        if self.fixed_irrigation_mm <= 0:
            raise ValueError("fixed_irrigation_mm must be positive")
        if self.seasonal_water_budget_mm < 0:
            raise ValueError("seasonal_water_budget_mm cannot be negative")
        if self.invalid_action_penalty < 0:
            raise ValueError("invalid_action_penalty cannot be negative")
        if self.rain_scale <= 0 or self.eto_scale <= 0:
            raise ValueError("observation scales must be positive")
        if self.weather_start_jday < 1:
            raise ValueError("weather_start_jday must use one-based Julian days")

        self.weather_seeds = tuple(int(s) for s in merged["weather_seeds"])
        if not self.weather_seeds:
            raise ValueError("weather_seeds cannot be empty")
        if len(set(self.weather_seeds)) != len(self.weather_seeds):
            raise ValueError("weather_seeds must be unique")
        if any(simyear < 1 for simyear in self.weather_seeds):
            raise ValueError("weather_seeds/simyears must be positive")
        self.processed_weather_file = self._resolve_processed_weather_file(
            merged["processed_weather_file"]
        )
        # Retain the public attribute for callers that report the active source.
        self.weather_file = self.processed_weather_file

        self.field_capacity_mm = self._field_vector("field_capacity_mm", positive=True)
        self.initial_moisture = self._field_vector("initial_moisture")
        self.irrigation_threshold = self._field_vector("irrigation_threshold")
        self.target_moisture = self._field_vector("target_moisture")
        self.upper_moisture_limit = self._field_vector("upper_moisture_limit")
        self.wilting_moisture = self._field_vector("wilting_moisture")
        self.rain_infiltration_efficiency = self._field_vector(
            "rain_infiltration_efficiency"
        ) #rainfall truly enter the field
        self.irrigation_efficiency = self._field_vector("irrigation_efficiency")
        self.et_multiplier = self._field_vector("et_multiplier", positive=True)
        self.daily_water_loss_fraction = self._field_vector(
            "daily_water_loss_fraction"
        )

        if np.any(self.wilting_moisture >= self.irrigation_threshold):
            raise ValueError("wilting moisture must be below irrigation thresholds")
        if np.any(self.irrigation_threshold >= self.target_moisture):
            raise ValueError("irrigation thresholds must be below targets")
        if np.any(self.target_moisture >= self.upper_moisture_limit):
            raise ValueError("targets must be below upper moisture limits")
        bounded_vectors = (
            self.initial_moisture,
            self.wilting_moisture,
            self.irrigation_threshold,
            self.target_moisture,
            self.upper_moisture_limit,
            self.rain_infiltration_efficiency,
            self.irrigation_efficiency,
            self.daily_water_loss_fraction,
        )
        if any(np.any((v < 0.0) | (v > 1.0)) for v in bounded_vectors):
            raise ValueError("moisture, efficiency, and loss fractions must be in [0, 1]")

        self.moisture_quality_weight = float(merged["moisture_quality_weight"])
        self.dry_stress_weight = float(merged["dry_stress_weight"])
        self.overwatering_weight = float(merged["overwatering_weight"])
        self.irrigation_cost_per_field = float(merged["irrigation_cost_per_field"])

        reward_scalars = {
            "moisture_quality_weight": self.moisture_quality_weight,
            "dry_stress_weight": self.dry_stress_weight,
            "overwatering_weight": self.overwatering_weight,
            "irrigation_cost_per_field": self.irrigation_cost_per_field,
        }

        for name, value in reward_scalars.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

        self.action_space = spaces.Discrete(7)

        obs_dim = 7 + self.forecast_horizon
        high = np.ones(obs_dim, dtype=np.float32)
        high[3:5] = np.inf  # scaled previous rain and current ETo can exceed one
        high[7:] = np.inf
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32), high=high, dtype=np.float32
        )

        self._selection_rng = np.random.default_rng(int(merged["selection_seed"]))
        self._weather_bank, self.weather_processor = self._load_weather_bank()

        self.weather: Optional[WeatherSequence] = None
        self.weather_seed: Optional[int] = None
        self._forecast_errors = np.zeros(
            (self.episode_days, self.forecast_horizon), dtype=np.float64
        )
        self.day = 0
        self.soil_water_mm = np.zeros(3, dtype=np.float64)
        self.remaining_budget_mm = self.seasonal_water_budget_mm
        self.cumulative_reward = 0.0
        self.irrigation_event_days = 0
        self.number_of_invalid_actions = 0
        self.irrigation_by_field_mm = np.zeros(3, dtype=np.float64)
        self.irrigation_applications_by_field = np.zeros(3, dtype=np.int64)
        self.target_deviation_sum = np.zeros(3, dtype=np.float64)
        self.cumulative_drought_stress = np.zeros(3, dtype=np.float64)
        self.overwatering_days = np.zeros(3, dtype=np.int64)
        self.cumulative_overwatering_magnitude = np.zeros(3, dtype=np.float64)

    def _resolve_processed_weather_file(self, configured_path: Any) -> Path:
        """Resolve the processed CSV without embedding a local absolute path."""

        path_text = str(configured_path).strip()
        if not path_text:
            raise ValueError("processed_weather_file cannot be empty")
        configured = Path(path_text).expanduser()
        candidates = (
            (configured,)
            if configured.is_absolute()
            else (Path.cwd() / configured, Path(__file__).resolve().parent / configured)
        )
        checked: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in checked:
                continue
            checked.append(resolved)
            if resolved.is_file():
                return resolved
            if resolved.exists():
                raise FileNotFoundError(
                    "Configured processed_weather_file exists but is not a file: "
                    f"{resolved}"
                )
        raise FileNotFoundError(
            "Could not find configured processed_weather_file. Checked: "
            + ", ".join(str(candidate) for candidate in checked)
        )

    def _load_weather_bank(self) -> tuple[dict[int, WeatherSequence], str]:
        """Load precomputed rainfall/ETo and build one entry per simyear."""

        try:
            processed_weather = pd.read_csv(self.processed_weather_file)
        except FileNotFoundError:
            raise
        except Exception as error:
            raise ValueError(
                "Could not parse processed_weather_file "
                f"{self.processed_weather_file}: {error}"
            ) from error

        if processed_weather.empty:
            raise ValueError(
                f"processed_weather_file {self.processed_weather_file} contains no rows"
            )
        if tuple(processed_weather.columns) != PROCESSED_WEATHER_COLUMNS:
            raise ValueError(
                "processed_weather_file must contain exactly these columns in order: "
                f"{PROCESSED_WEATHER_COLUMNS!r}; found "
                f"{tuple(processed_weather.columns)!r}"
            )

        for column in PROCESSED_WEATHER_COLUMNS:
            processed_weather[column] = pd.to_numeric(
                processed_weather[column], errors="coerce"
            )
        numeric_values = processed_weather.loc[
            :, PROCESSED_WEATHER_COLUMNS
        ].to_numpy(dtype=np.float64)
        invalid_rows = np.flatnonzero(~np.isfinite(numeric_values).all(axis=1))
        if invalid_rows.size:
            first_bad_row = int(invalid_rows[0]) + 2
            raise ValueError(
                "processed_weather_file contains missing, non-numeric, or "
                f"non-finite data at CSV row "
                f"{first_bad_row}"
            )

        simyear_values = numeric_values[:, 0]
        jday_values = numeric_values[:, 1]
        if np.any(simyear_values != np.floor(simyear_values)) or np.any(
            simyear_values < 1
        ):
            raise ValueError("simyear values must be positive integers")
        if np.any(jday_values != np.floor(jday_values)) or np.any(jday_values < 1):
            raise ValueError("jday values must be positive integers")
        if np.any(numeric_values[:, 2] < 0.0):
            raise ValueError("rain_mm values must be non-negative")
        if np.any(numeric_values[:, 3] < 0.0):
            raise ValueError("eto_mm values must be non-negative")
        processed_weather["simyear"] = simyear_values.astype(np.int64)
        processed_weather["jday"] = jday_values.astype(np.int64)

        duplicated = processed_weather.duplicated(
            ["simyear", "jday"], keep=False
        )
        if duplicated.any():
            example = processed_weather.loc[
                duplicated, ["simyear", "jday"]
            ].iloc[0]
            raise ValueError(
                "processed_weather_file contains duplicate daily rows; for example "
                f"simyear={int(example['simyear'])}, jday={int(example['jday'])}"
            )

        available_years = set(processed_weather["simyear"].unique().tolist())
        missing_years = sorted(set(self.weather_seeds) - available_years)
        if missing_years:
            raise ValueError(
                f"Requested simulated weather years are absent from "
                f"{self.processed_weather_file}: {missing_years}. Available range is "
                f"{min(available_years)}..{max(available_years)}"
            )

        # The final decision is zero-based day 119. Its three-entry forecast
        # needs sequence positions 119, 120, and 121, so a 120-day episode
        # requires 120 + 3 - 1 source days.
        number_of_weather_days = self.episode_days + self.forecast_horizon - 1
        required_jdays = np.arange(
            self.weather_start_jday,
            self.weather_start_jday + number_of_weather_days,
            dtype=np.int64,
        )

        weather_bank: dict[int, WeatherSequence] = {}
        for simyear in self.weather_seeds:
            seasonal_rows = (
                processed_weather.loc[processed_weather["simyear"] == simyear]
                .sort_values("jday", kind="stable")
                .reset_index(drop=True)
            )
            if len(seasonal_rows) != number_of_weather_days:
                raise ValueError(
                    f"simyear {simyear} contains {len(seasonal_rows)} rows; "
                    f"expected exactly {number_of_weather_days}"
                )
            actual_jdays = seasonal_rows["jday"].to_numpy(dtype=np.int64)
            if not np.array_equal(actual_jdays, required_jdays):
                raise ValueError(
                    f"simyear {simyear} must contain exactly the consecutive "
                    f"jday range {required_jdays[0]}..{required_jdays[-1]}; "
                    f"found {actual_jdays.tolist()}"
                )
            rain_mm = seasonal_rows["rain_mm"].to_numpy(dtype=np.float64)
            eto_mm = seasonal_rows["eto_mm"].to_numpy(dtype=np.float64)
            if not np.isfinite(rain_mm).all() or not np.isfinite(eto_mm).all():
                raise ValueError(
                    f"simyear {simyear} contains missing or non-finite rainfall/ETo "
                    "values"
                )
            if np.any(rain_mm < 0.0) or np.any(eto_mm < 0.0):
                raise ValueError(
                    f"simyear {simyear} contains negative rainfall or reference ETo "
                    "values"
                )
            rain_mm = rain_mm.copy()
            eto_mm = eto_mm.copy()
            rain_mm.setflags(write=False)
            eto_mm.setflags(write=False)
            weather_bank[simyear] = WeatherSequence(
                seed=simyear,
                rain_mm=rain_mm,
                eto_mm=eto_mm,
            )
        return weather_bank, WEATHER_PROCESSOR

    def _field_vector(self, key: str, *, positive: bool = False) -> np.ndarray:
        value = np.asarray(self.config[key], dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"{key} must contain exactly three values")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} values must all be finite")
        if positive and np.any(value <= 0):
            raise ValueError(f"{key} values must be positive")

        return value.copy()

    @property
    def soil_moisture(self) -> np.ndarray:
        """Current field moisture as fractions of field capacity."""

        return self.soil_water_mm / self.field_capacity_mm

    def _required_water_mm(self, action: int) -> float:
        """Return the gross seasonal-budget requirement of an action."""

        if not self.action_space.contains(action):
            raise ValueError("action must be an integer in [0, 6]")
        action = int(action)
        return float(ACTION_TO_FIELDS[action].sum() * self.fixed_irrigation_mm)

    def get_valid_action_mask(
        self,
        remaining_budget_mm: Optional[float] = None,
    ) -> np.ndarray:
        """Return a Boolean mask of actions feasible under the given budget."""

        budget = float(
            self.remaining_budget_mm
            if remaining_budget_mm is None
            else remaining_budget_mm
        )
        if not np.isfinite(budget):
            raise ValueError("remaining_budget_mm must be finite")
        if budget < 0.0:
            raise ValueError("remaining_budget_mm cannot be negative")
        return np.asarray(
            [
                self._required_water_mm(action) <= budget + 1e-9
                for action in range(self.action_space.n)
            ],
            dtype=np.bool_,
        )

    def _choose_weather_seed(self, options: Optional[dict[str, Any]]) -> int:
        if options and "weather_seed" in options:
            requested = int(options["weather_seed"])
            if requested not in self._weather_bank:
                raise ValueError(
                    f"weather_seed/simyear {requested} is outside this "
                    "environment's fixed weather bank"
                )
            return requested
        return int(self._selection_rng.choice(self.weather_seeds))

    #noisy forecast
    def _make_forecast_errors(self, weather_seed: int) -> np.ndarray:
        # The same standard-normal errors are scaled for low/high treatments.
        # This pairs those treatments and keeps errors independent of PPO seeds.
        seed_sequence = np.random.SeedSequence(
            [self.forecast_noise_seed, int(weather_seed)]
        )
        noise_rng = np.random.default_rng(seed_sequence)
        standard_errors = noise_rng.normal(
            0.0, 1.0, size=(self.episode_days, self.forecast_horizon)
        )
        return standard_errors * self.forecast_noise_std_mm

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._selection_rng = np.random.default_rng(int(seed))

        self.weather_seed = self._choose_weather_seed(options)
        self.weather = self._weather_bank[self.weather_seed]
        self._forecast_errors = self._make_forecast_errors(self.weather_seed)

        self.day = 0
        self.soil_water_mm = self.initial_moisture * self.field_capacity_mm
        self.remaining_budget_mm = float(self.seasonal_water_budget_mm)
        self.cumulative_reward = 0.0
        self.irrigation_event_days = 0
        self.number_of_invalid_actions = 0
        self.irrigation_by_field_mm = np.zeros(3, dtype=np.float64)
        self.irrigation_applications_by_field = np.zeros(3, dtype=np.int64)
        self.target_deviation_sum = np.zeros(3, dtype=np.float64)
        self.cumulative_drought_stress = np.zeros(3, dtype=np.float64)
        self.overwatering_days = np.zeros(3, dtype=np.int64)
        self.cumulative_overwatering_magnitude = np.zeros(3, dtype=np.float64)

        initial_budget_mm = float(self.remaining_budget_mm)
        initial_valid_action_mask = self.get_valid_action_mask(initial_budget_mm)
        info = self._info(
            requested_action=0,
            executed_action=0,
            requested_field_mask=ACTION_TO_FIELDS[0],
            applied_field_mask=ACTION_TO_FIELDS[0],
            invalid_action=False,
            requested_water_mm=0.0,
            applied_water_mm=0.0,
            budget_before_mm=initial_budget_mm,
            budget_after_mm=initial_budget_mm,
            budget_feasible=bool(initial_valid_action_mask[0]),
            valid_action_mask_before=initial_valid_action_mask,
            valid_action_mask_after=initial_valid_action_mask,
            reward_components=None,
        )
        return self._observation(), info

    def _forecast(self) -> np.ndarray:
        if not self.use_forecast or self.day >= self.episode_days:
            return np.zeros(self.forecast_horizon, dtype=np.float64)
        assert self.weather is not None
        start = self.day
        actual_future_rain = self.weather.rain_mm[start : start + self.forecast_horizon]
        noisy = actual_future_rain + self._forecast_errors[self.day]
        return np.clip(noisy, 0.0, None)

    def _observation(self) -> np.ndarray:
        if self.weather is None:
            raise RuntimeError("Call reset() before requesting an observation")
        if self.day < self.episode_days:
            previous_rain = (0.0 if self.day == 0 else self.weather.rain_mm[self.day - 1] )
            current_eto = self.weather.eto_mm[self.day]
        else:
            previous_rain = self.weather.rain_mm[self.episode_days - 1]
            current_eto = 0.0
        budget_fraction = (
            self.remaining_budget_mm / self.seasonal_water_budget_mm
            if self.seasonal_water_budget_mm > 0
            else 0.0
        )
        values = [
            *self.soil_moisture,
            previous_rain / self.rain_scale,
            current_eto / self.eto_scale,
            budget_fraction,
            self.day / self.episode_days,
        ]
        values.extend(self._forecast() / self.rain_scale)
        return np.asarray(values, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.weather is None:
            raise RuntimeError("Call reset() before step()")
        if self.day >= self.episode_days:
            raise RuntimeError("Episode is complete; call reset()")
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer in [0, 6]")

        requested_action = int(action)
        decision_day = int(self.day)
        budget_before_mm = float(self.remaining_budget_mm)
        valid_action_mask_before = self.get_valid_action_mask(budget_before_mm)
        requested_field_mask = ACTION_TO_FIELDS[requested_action].copy()
        requested_water_mm = self._required_water_mm(requested_action)
        budget_feasible = bool(valid_action_mask_before[requested_action])
        invalid_action = not budget_feasible
        executed_action = 0 if invalid_action else requested_action
        applied_field_mask = (
            np.zeros(3, dtype=np.int8)
            if invalid_action
            else requested_field_mask.copy()
        )

        soil_moisture_before = self.soil_moisture.copy()
        previous_rainfall = (
            0.0
            if decision_day == 0
            else float(self.weather.rain_mm[decision_day - 1])
        )
        forecast_rainfall = self._forecast().copy()
        today_rain = float(self.weather.rain_mm[decision_day])
        today_eto = float(self.weather.eto_mm[decision_day])
        gross_by_field = applied_field_mask * self.fixed_irrigation_mm
        applied_water_mm = float(gross_by_field.sum())

        if invalid_action:
            self.number_of_invalid_actions += 1

        if applied_field_mask.any():
            effective_by_field = gross_by_field * self.irrigation_efficiency
            self.soil_water_mm += effective_by_field
            self.irrigation_event_days += 1
            self.irrigation_by_field_mm += gross_by_field
            self.irrigation_applications_by_field += applied_field_mask

        self.remaining_budget_mm = max(
            0.0, budget_before_mm - applied_water_mm
        )
        budget_after_mm = float(self.remaining_budget_mm)
        valid_action_mask_after = self.get_valid_action_mask(budget_after_mm)

        self.soil_water_mm += today_rain * self.rain_infiltration_efficiency
        background_loss = self.daily_water_loss_fraction * np.maximum(
            self.soil_water_mm, 0.0
        )
        self.soil_water_mm -= today_eto * self.et_multiplier + background_loss
        self.soil_water_mm = np.clip(
            self.soil_water_mm, 0.0, self.field_capacity_mm
        )

        moisture = self.soil_moisture
        absolute_deviation = np.abs(moisture - self.target_moisture)
        drought_stress = np.maximum(self.irrigation_threshold - moisture,0.0)
        overwatering = np.maximum(moisture - self.upper_moisture_limit, 0.0)

        self.target_deviation_sum += absolute_deviation
        self.cumulative_drought_stress += drought_stress
        self.overwatering_days += overwatering > 0.0
        self.cumulative_overwatering_magnitude += overwatering

        dry_normalizer = np.maximum(self.irrigation_threshold - self.wilting_moisture,1e-6)
        wet_normalizer = np.maximum(1.0 - self.upper_moisture_limit, 1e-6)
        moisture_quality = self.moisture_quality_weight * float(np.mean(1.0 - absolute_deviation))
        dry_penalty = self.dry_stress_weight * float(np.mean((drought_stress / dry_normalizer) ** 2) )
        overwatering_penalty = self.overwatering_weight * float(np.mean((overwatering / wet_normalizer) ** 2))
        irrigation_penalty = self.irrigation_cost_per_field * float(
            applied_field_mask.sum()
        )
        invalid_penalty = self.invalid_action_penalty if invalid_action else 0.0
        reward = (
            moisture_quality
            - dry_penalty
            - overwatering_penalty
            - irrigation_penalty
            - invalid_penalty
        )
        components = {
            "moisture_quality": moisture_quality,
            "dry_stress_penalty": dry_penalty,
            "overwatering_penalty": overwatering_penalty,
            "irrigation_penalty": irrigation_penalty,
            "invalid_action_penalty": invalid_penalty,
        }

        self.cumulative_reward += reward
        self.day += 1
        terminated = self.day >= self.episode_days
        info = self._info(
            requested_action=requested_action,
            executed_action=executed_action,
            requested_field_mask=requested_field_mask,
            applied_field_mask=applied_field_mask,
            invalid_action=invalid_action,
            requested_water_mm=requested_water_mm,
            applied_water_mm=applied_water_mm,
            budget_before_mm=budget_before_mm,
            budget_after_mm=budget_after_mm,
            budget_feasible=budget_feasible,
            valid_action_mask_before=valid_action_mask_before,
            valid_action_mask_after=valid_action_mask_after,
            reward_components=components,
            decision_day=decision_day,
            previous_rainfall_mm=previous_rainfall,
            actual_rainfall_mm=today_rain,
            forecast_rainfall_mm=forecast_rainfall,
            eto_mm=today_eto,
            soil_moisture_before=soil_moisture_before,
            irrigation_applied_by_field_mm=gross_by_field,
            daily_reward=float(reward),
        )
        if terminated:
            info["episode_summary"] = self.episode_summary()
        return self._observation(), float(reward), terminated, False, info

    def episode_summary(self) -> dict[str, Any]:
        """Return evaluation metrics accumulated so far.

        Drought stress and overwatering magnitude are moisture-fraction-days.
        Overwatering frequency is the fraction of completed days above each
        field's upper limit.
        """

        completed_days = max(self.day, 1)
        return {
            "weather_seed": self.weather_seed,
            "weather_fingerprint": None
            if self.weather is None
            else self.weather.fingerprint,
            "completed_days": int(self.day),
            "cumulative_episode_reward": float(self.cumulative_reward),
            "total_irrigation_water_used_mm": float(
                self.irrigation_by_field_mm.sum()
            ),
            "number_of_irrigation_events": int(self.irrigation_event_days),
            "number_of_invalid_actions": int(self.number_of_invalid_actions),
            "invalid_action_frequency": float(
                self.number_of_invalid_actions / completed_days
            ),
            "irrigation_applications_by_field": self.irrigation_applications_by_field.copy(),
            "mean_soil_moisture_deviation_by_field": (
                self.target_deviation_sum / completed_days
            ).copy(),
            "cumulative_drought_stress_by_field": self.cumulative_drought_stress.copy(),
            "overwatering_days_by_field": self.overwatering_days.copy(),
            "overwatering_frequency_by_field": (
                self.overwatering_days / completed_days
            ).copy(),
            "cumulative_overwatering_magnitude_by_field": (
                self.cumulative_overwatering_magnitude.copy()
            ),
            "irrigation_water_allocated_by_field_mm": self.irrigation_by_field_mm.copy(),
            "remaining_water_budget_mm": float(self.remaining_budget_mm),
        }

    def _info(
        self,
        *,
        requested_action: int,
        executed_action: int,
        requested_field_mask: np.ndarray,
        applied_field_mask: np.ndarray,
        invalid_action: bool,
        requested_water_mm: float,
        applied_water_mm: float,
        budget_before_mm: float,
        budget_after_mm: float,
        budget_feasible: bool,
        valid_action_mask_before: np.ndarray,
        valid_action_mask_after: np.ndarray,
        reward_components: Optional[dict[str, float]],
        decision_day: Optional[int] = None,
        previous_rainfall_mm: Optional[float] = None,
        actual_rainfall_mm: Optional[float] = None,
        forecast_rainfall_mm: Optional[np.ndarray] = None,
        eto_mm: Optional[float] = None,
        soil_moisture_before: Optional[np.ndarray] = None,
        irrigation_applied_by_field_mm: Optional[np.ndarray] = None,
        daily_reward: Optional[float] = None,
    ) -> dict[str, Any]:
        info = {
            "day": int(self.day if decision_day is None else decision_day),
            "weather_seed": self.weather_seed,
            "weather_fingerprint": None
            if self.weather is None
            else self.weather.fingerprint,
            # ``action`` remains the requested PPO action for compatibility.
            "action": int(requested_action),
            "requested_action": int(requested_action),
            "executed_action": int(executed_action),
            "requested_field_mask": requested_field_mask.copy(),
            "applied_field_mask": applied_field_mask.copy(),
            "invalid_action": bool(invalid_action),
            "requested_water_mm": float(requested_water_mm),
            "applied_water_mm": float(applied_water_mm),
            "budget_before_mm": float(budget_before_mm),
            "budget_after_mm": float(budget_after_mm),
            "budget_feasible": bool(budget_feasible),
            "valid_action_mask_before": valid_action_mask_before.copy(),
            "valid_action_mask_after": valid_action_mask_after.copy(),
            "soil_moisture": self.soil_moisture.copy(),
            "remaining_water_budget_mm": float(budget_after_mm),
            "cumulative_reward": float(self.cumulative_reward),
            "reward_components": reward_components,
            "number_of_invalid_actions": int(self.number_of_invalid_actions),
        }
        if decision_day is not None:
            assert previous_rainfall_mm is not None
            assert actual_rainfall_mm is not None
            assert forecast_rainfall_mm is not None
            assert eto_mm is not None
            assert soil_moisture_before is not None
            assert irrigation_applied_by_field_mm is not None
            assert daily_reward is not None
            info.update(
                {
                    "previous_observed_rainfall_mm": float(previous_rainfall_mm),
                    "actual_rainfall_mm": float(actual_rainfall_mm),
                    "rainfall_forecast_mm": forecast_rainfall_mm.copy(),
                    "eto_mm": float(eto_mm),
                    "soil_moisture_before": soil_moisture_before.copy(),
                    "irrigation_applied_by_field_mm": (
                        irrigation_applied_by_field_mm.copy()
                    ),
                    "daily_reward": float(daily_reward),
                }
            )
        return info

    def render(self) -> None:
        moisture = ", ".join(
            f"F{i + 1}={value:.3f}" for i, value in enumerate(self.soil_moisture)
        )
        print(
            f"day={self.day:03d} | {moisture} | "
            f"budget={self.remaining_budget_mm:.1f} mm"
        )


class ActionMaskedIrrigationEnv(IrrigationEnv):
    """RLlib action-masking view of :class:`IrrigationEnv`.

    The core environment keeps its original ten-dimensional physical
    observation and defensive invalid-action handling. This subclass only
    wraps that observation in the Dict schema expected by RLlib's official
    ``ActionMaskingTorchRLModule`` and fails fast if a policy ever selects an
    action that the supplied seasonal-budget mask marked invalid.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(config)
        self.physical_observation_space = self.observation_space
        if self.physical_observation_space.shape != (10,):
            raise AssertionError("Physical irrigation observation must have shape (10,)")
        self.observation_space = spaces.Dict(
            {
                "observations": self.physical_observation_space,
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.action_space.n,),
                    dtype=np.float32,
                ),
            }
        )

    def _masked_observation(
        self, base_observation: np.ndarray
    ) -> dict[str, np.ndarray]:
        physical = np.asarray(base_observation, dtype=np.float32)
        action_mask = self.get_valid_action_mask().astype(np.float32)
        if physical.shape != (10,):
            raise AssertionError("Physical irrigation observation must have shape (10,)")
        if action_mask.shape != (7,):
            raise AssertionError("Action mask must have shape (7,)")
        if action_mask[0] != 1.0 or not np.any(action_mask):
            raise AssertionError("Action 0 must always remain feasible")
        return {"observations": physical, "action_mask": action_mask}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        base_observation, info = super().reset(seed=seed, options=options)
        return self._masked_observation(base_observation), info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer in [0, 6]")
        action = int(action)
        mask_before = self.get_valid_action_mask()
        if not bool(mask_before[action]):
            raise AssertionError(
                "RLlib selected an action that was masked as invalid"
            )
        base_observation, reward, terminated, truncated, info = super().step(action)
        info["action_mask_supplied_to_policy"] = mask_before.astype(np.float32)
        info["masked_action_selected"] = False
        return (
            self._masked_observation(base_observation),
            reward,
            terminated,
            truncated,
            info,
        )
