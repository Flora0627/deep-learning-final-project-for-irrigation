# Irrigation PPO Experiment

This folder contains the complete code, data, trained outputs, and analysis for the irrigation PPO project:

**The Value of Rainfall Forecasts for PPO-Based Irrigation Timing and Field Prioritization**

The main workflow is provided in `Irrigation_PPO.ipynb`.

## Project files

- `Irrigation_PPO.ipynb`  
  Main notebook containing the experimental workflow, PPO training, evaluation, and result generation.

- `irrigation_env.py`  
  Defines the irrigation environment, including observations, actions, soil-moisture dynamics, reward calculation, and action masking.

- `experiment_protocol.py`  
  Stores the fixed experimental settings, including weather-year splits, forecast treatments, water budgets, PPO seeds, and training parameters.

- `prepare_weather.ipynb`  
  Documents the preparation of rainfall and reference evapotranspiration data.

- `data/`  
  Contains the weather data used by the experiment.

- `outputs/`  
  Contains the saved baseline parameters, PPO checkpoints, holdout evaluation results, and final analysis outputs.

- `CPWG_processed.csv`  
  Complete processed weather dataset supplied with the project.

- `run_preflight.py`  
  Performs basic checks before running the experiment.

- `requirements.txt`  
  Lists the required Python packages.

## Experimental design

The experiment compares three rainfall-forecast treatments:

- No Forecast
- Noisy three-day Forecast
- Perfect three-day Forecast

Two seasonal water budgets are considered:

- Moderate: 1,800 mm
- Abundant: 2,400 mm

The simulated weather years are divided into:

- Training: simyears 1–12
- Development: simyears 13–30
- Final holdout: simyears 31–40

Three PPO training seeds are used for each treatment and budget combination, producing 18 PPO models in total.

Each PPO model is trained for a target of 150,000 environment steps. Because complete training batches are used, the actual stopping point is 151,200 steps. Checkpoints are evaluated on the development years, and the checkpoint with the highest mean development reward is selected for final evaluation.

## Running the project

Install the required packages:

```bash
python -m pip install -r requirements.txt
