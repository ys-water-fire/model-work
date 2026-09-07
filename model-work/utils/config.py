import yaml
import argparse
import os

_config_cache = None


def load_config():
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    parser = argparse.ArgumentParser(description='Load config file')
    default_config = os.path.join(os.path.dirname(__file__), '..', 'configs', 'source1.yaml')
    parser.add_argument('--config', type=str, default=default_config, help='Path to config file')
    args, _ = parser.parse_known_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache