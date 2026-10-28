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


def test_reference_rows_change_more_than_one_knob() -> None:
    matrix = _load()
    baseline = matrix.EXPERIMENTS[0]
    knobs = [f.name for f in dataclasses.fields(baseline) if f.name != "label"]

    for reference in matrix.REFERENCES:
        changed = [k for k in knobs if getattr(reference, k) != getattr(baseline, k)]
        assert len(changed) > 1, (reference.label, changed)


def test_experiment_labels_are_unique() -> None:
    matrix = _load()
    labels = [e.label for e in [*matrix.EXPERIMENTS, *matrix.REFERENCES]]

    assert len(labels) == len(set(labels))
