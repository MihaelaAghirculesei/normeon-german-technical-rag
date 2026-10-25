"""The ablation matrix changes one knob per row -- otherwise a difference
against the baseline has more than one cause."""

import dataclasses
import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_matrix.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_experiment_changes_exactly_one_knob_of_the_baseline() -> None:
    matrix = _load()
    baseline, *experiments = matrix.EXPERIMENTS
    knobs = [f.name for f in dataclasses.fields(baseline) if f.name != "label"]

    assert baseline == matrix.Experiment(matrix.BASELINE)
    assert experiments
    for experiment in experiments:
        changed = [k for k in knobs if getattr(experiment, k) != getattr(baseline, k)]
        assert len(changed) == 1, (experiment.label, changed)


def test_experiment_labels_are_unique() -> None:
    labels = [e.label for e in _load().EXPERIMENTS]

    assert len(labels) == len(set(labels))
