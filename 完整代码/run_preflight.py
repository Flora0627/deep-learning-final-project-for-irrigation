from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "Irrigation_PPO_Complete_Reproducible.ipynb"
REPORT_PATH = ROOT / "PREFLIGHT_REPORT.md"
RESULTS: list[tuple[str, bool, str]] = []

def check(name: str, function: Callable[[], object]) -> None:
    try:
        detail = function()
        RESULTS.append((name, True, "PASS" if detail is None else str(detail)))
    except Exception as error:
        RESULTS.append((name, False, f"{type(error).__name__}: {error}"))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def compile_python(path: Path) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        py_compile.compile(str(path), cfile=str(Path(temporary) / (path.stem + ".pyc")), doraise=True)
    return "compiled"

def protocol_check() -> str:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import experiment_protocol as protocol
    assert protocol.validate_protocol()
    assert len(protocol.formal_model_matrix()) == 18
    assert protocol.TRAIN_SIMYEARS == tuple(range(1, 13))
    assert protocol.DEVELOPMENT_SIMYEARS == tuple(range(13, 31))
    assert protocol.FINAL_HOLDOUT_SIMYEARS == tuple(range(31, 41))
    return "18 models; frozen year splits valid"

def validate_weather(path: Path, expected_years: tuple[int, ...], expected_rows: int) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["simyear", "jday", "rain_mm", "eto_mm"]
        rows = list(reader)
    assert len(rows) == expected_rows
    assert tuple(sorted({int(row["simyear"]) for row in rows})) == expected_years
    keys = {(int(row["simyear"]), int(row["jday"])) for row in rows}
    assert len(keys) == expected_rows
    counts = {year: 0 for year in expected_years}
    for row in rows:
        year, day = int(row["simyear"]), int(row["jday"])
        assert 121 <= day <= 242
        assert float(row["rain_mm"]) >= 0 and float(row["eto_mm"]) >= 0
        counts[year] += 1
    assert set(counts.values()) == {122}
    return f"{expected_rows} rows; {len(expected_years)} simyears"

def manifest_check() -> str:
    manifest = json.loads((ROOT / "data" / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert sha256_file(ROOT / "CPWG_processed.csv") == manifest["submission_full_csv_sha256"]
    assert sha256_file(ROOT / "data" / "CPWG_processed_train_development_1_30.csv") == manifest["train_development_csv_sha256"]
    assert sha256_file(ROOT / "data" / "CPWG_processed_final_holdout_31_40.csv") == manifest["final_holdout_csv_sha256"]
    assert sha256_file(ROOT / "irrigation_env.py") == manifest["irrigation_env_sha256"]
    assert sha256_file(ROOT / "experiment_protocol.py") == manifest["experiment_protocol_sha256"]
    return "all sealed SHA-256 values match"

def notebook_check() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None, index
            assert cell.get("outputs") == [], index
            compile("".join(cell.get("source", [])), f"cell_{index}", "exec")
    forbidden = [
        "/" + "Users" + "/",
        "/" + "opt" + "/",
        "final_" + "test_episode_metrics",
        "final_" + "test_outputs",
        "noisy_" + "low",
        "noisy_" + "high",
    ]
    assert not [item for item in forbidden if item in source]
    gates = re.findall(
        r"^RUN_(?:THRESHOLD_OPTIMIZATION|FORMAL_TRAINING|FINAL_HOLDOUT|FINAL_ANALYSIS)\s*=\s*(True|False)$",
        source,
        flags=re.MULTILINE,
    )
    assert gates == ["False", "False", "False", "False"]
    headings = [f"## {index}." for index in range(1, 16)]
    positions = [source.index(heading) for heading in headings]
    assert positions == sorted(positions)
    formal_cells = [cell for cell in notebook["cells"] if "formal_source" in cell.get("metadata", {})]
    assert len(formal_cells) == 24
    required = [
        "CyclicTrainingWeatherEnv", "prepare_initial_states", "load_initial_state",
        "SELECTION_TOLERANCE = 1e-9", "EXPECTED_ACTUAL_STOP_STEPS",
        "BASELINES_FROZEN.flag", "FORMAL_18_MODELS_COMPLETE.flag",
        "EXPECTED_MAIN_EPISODES = 220", "EXPECTED_PPO_EPISODES = 180",
        "EXPECTED_BASELINE_EPISODES = 40", '"true", "zero", "shuffled"',
        "BOOTSTRAP_REPLICATES = 10_000", "stable_rng", "PRIMARY_METRICS",
        "PAIRED_COMPARISONS", "FINAL_ANALYSIS_COMPLETE.flag",
    ]
    missing = [item for item in required if item not in source]
    assert not missing, missing
    return f"{len(notebook['cells'])} cells compile; 15 sections; 24 formal cells traced; gates disabled"

def environment_smoke() -> str:
    completed = subprocess.run(
        [sys.executable, "irrigation_env.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stdout + "\n" + completed.stderr)[-3000:])
    assert "Smoke test passed" in completed.stdout
    return "120-day physical/action-mask smoke assertions passed"

check("Python compile: irrigation_env.py", lambda: compile_python(ROOT / "irrigation_env.py"))
check("Python compile: experiment_protocol.py", lambda: compile_python(ROOT / "experiment_protocol.py"))
check("Protocol invariants", protocol_check)
check("Notebook structure and source safety", notebook_check)
check("Weather schema: train/development", lambda: validate_weather(ROOT / "data" / "CPWG_processed_train_development_1_30.csv", tuple(range(1, 31)), 3660))
check("Weather schema: final holdout", lambda: validate_weather(ROOT / "data" / "CPWG_processed_final_holdout_31_40.csv", tuple(range(31, 41)), 1220))
check("Protocol manifest hashes", manifest_check)
check("Environment smoke test", environment_smoke)

overall = all(passed for _, passed, _ in RESULTS)
lines = [
    "# Preflight report", "", f"Overall status: **{'PASS' if overall else 'FAIL'}**", "",
    "No PPO training, threshold optimization, protected holdout episode, or final analysis was run.", "",
    "| Check | Status | Detail |", "|---|---:|---|",
]
for name, passed, detail in RESULTS:
    lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail.replace('|', chr(92) + '|').replace(chr(10), ' ')} |")
lines.append("")
REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
if not overall:
    raise SystemExit(1)
