from pathlib import Path

import pytest
import yaml

from medical_ai_explainability.cli import _load_config, _validate_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


_VALID_CONFIG = {
    "random_state": 42,
    "data": {"test_size": 0.2},
    "model_selection": {"metric": "roc_auc"},
    "explainability": {"top_k": 8, "sample_index": 0},
    "reports": {"output_dir": "reports/generated"},
}


def test_load_config_accepts_valid_file(tmp_path):
    path = _write_config(tmp_path, _VALID_CONFIG)
    config = _load_config(path)
    assert config["random_state"] == 42


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        _load_config(tmp_path / "nonexistent.yaml")


def test_validate_config_raises_on_missing_root_key(tmp_path):
    bad = {k: v for k, v in _VALID_CONFIG.items() if k != "random_state"}
    path = _write_config(tmp_path, bad)
    with pytest.raises(ValueError, match="random_state"):
        _load_config(path)


def test_validate_config_raises_on_missing_section_key(tmp_path):
    bad = {**_VALID_CONFIG, "data": {}}
    path = _write_config(tmp_path, bad)
    with pytest.raises(ValueError, match="data.test_size"):
        _load_config(path)


def test_validate_config_raises_on_missing_explainability_key(tmp_path):
    bad = {**_VALID_CONFIG, "explainability": {"top_k": 8}}
    path = _write_config(tmp_path, bad)
    with pytest.raises(ValueError, match="explainability.sample_index"):
        _load_config(path)
