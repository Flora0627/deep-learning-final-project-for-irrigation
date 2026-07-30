# Final irrigation PPO experiment

This directory is the complete, portable entry point for the frozen irrigation experiment. The single workflow notebook is `Irrigation_PPO_Complete_Reproducible.ipynb`. It locates this project root from either this directory or a `notebooks/` launch directory by requiring both authoritative Python modules.

## Required contents

- `Irrigation_PPO_Complete_Reproducible.ipynb`
- `irrigation_env.py` — sole authority for dynamics, observations, reward, actions, and masking
- `experiment_protocol.py` — sole authority for splits, budgets, treatments, seeds, PPO parameters, and the 18-model matrix
- `data/` — sealed training/development and protected final-holdout weather plus manifests
- `requirements.txt`
- `README.md`

`run_preflight.py` is included and should be run before any long phase. `CPWG_processed.csv` is the unpartitioned source weather supplied with the submission; formal code uses only the two canonical sealed files under `data/`.

## Frozen protocol

- Training simyears 1–12; development simyears 13–30; protected final holdout 31–40.
- Moderate budget 1,800 mm; abundant budget 2,400 mm.
- No Forecast, Noisy Forecast (2 mm), Perfect Forecast.
- PPO seeds 0, 1, 2; 18 models total.
- Irrigation cost 0.16 per irrigated field.
- Target 150,000 environment steps; exact formal stop 151,200.
- Batch 2,400; minibatch 240; 10 epochs; learning rate 3e-4; frozen entropy schedule.
- Development checkpoint evaluation every two iterations; selection uses only mean cumulative reward on simyears 13–30.

## Safe execution order

1. Create a clean environment and run `python -m pip install -r requirements.txt`.
2. Run `python run_preflight.py`; require an overall PASS.
3. Open the notebook, keep all four gates `False`, restart the kernel, and Run All once.
4. Restart; enable only `RUN_THRESHOLD_OPTIMIZATION`; Run All and verify `outputs/baselines_final_v1/BASELINES_FROZEN.flag`.
5. Restart; enable only `RUN_FORMAL_TRAINING`; Run All and verify `outputs/formal_ppo_final_v1/FORMAL_18_MODELS_COMPLETE.flag`.
6. Only after both freezes, restart; enable only `RUN_FINAL_HOLDOUT`; Run All exactly once and verify `outputs/final_holdout_31_40_v1/FINAL_HOLDOUT_COMPLETE.flag`.
7. Restart; enable only `RUN_FINAL_ANALYSIS`; Run All and verify `outputs/final_analysis_v1/FINAL_ANALYSIS_COMPLETE.flag`.

Enable exactly one gate per pass. The protected holdout may not be used to train, tune, select checkpoints, revise thresholds, or modify the protocol.

## Formal outputs

- `outputs/baselines_final_v1/`
- `outputs/formal_ppo_final_v1/`
- `outputs/final_holdout_31_40_v1/`
- `outputs/final_analysis_v1/`

All formal writers use canonical filenames, audit their row counts and invariants, and refuse unsafe overwrite or incomplete resume states.
