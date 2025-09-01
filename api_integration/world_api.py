import requests

def fetch_world_signal(api_url, params=None):
    """
    Fetches world/environmental signal (e.g. weather, news, sensor data) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map weather intensity or news sentiment to energy signal
        intensity = data.get("intensity", 0.3)
        energy_signal = intensity * 12
        return ("world", energy_signal)
    except Exception as e:
        print(f"[API ERROR] World signal fetch failed: {e}")
        return ("world", 2.0)
