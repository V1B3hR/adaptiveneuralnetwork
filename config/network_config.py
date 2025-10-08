import yaml


def load_network_config(config_path="config/network_config.yaml"):
    """
    Loads network configuration from YAML file.
    Returns a dictionary of config parameters.
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[CONFIG ERROR] Failed to load config: {e}")
        # Return defaults if config missing
        return {
            "nodes": 9,
            "topology": "grid",
            "spatial_dims": 2,  # Default to 2D for backward compatibility
            "external_signal_interval": 5,
            "api_endpoints": {
                "human": None,
                "ai": None,
                "world": None
            }
        }
