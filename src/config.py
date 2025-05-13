"""
Project configuration loader.
"""

import yaml


def load_config(path="config.yaml"):
    """Load project configuration from a YAML file.

    Args:
        path (str): Path to the config YAML file.

    Returns:
        dict: A dictionary of configuration values.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
