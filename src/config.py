"""Resolve configuration paths relative to the YAML file."""
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section, key in [("data", "path"), ("output", "directory")]:
        config[section][key] = str((path.parent / config[section][key]).resolve())
    if not 0 <= config["evaluation"]["threshold"] <= 1:
        raise ValueError("Decision threshold must be between zero and one")
    for key in ["psi_bins", "calibration_bins"]:
        value = config["evaluation"][key]
        if not isinstance(value, int) or value < 2:
            raise ValueError(f"{key} must be an integer of at least two")
    if config["evaluation"]["psi_epsilon"] <= 0:
        raise ValueError("PSI smoothing must be positive")
    return config
