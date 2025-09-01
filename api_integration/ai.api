import requests

def fetch_ai_signal(api_url, params=None):
    """
    Fetches AI-generated signal (e.g. model prediction, system status) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map AI prediction confidence to energy signal
        confidence = data.get("confidence", 0.5)
        energy_signal = confidence * 7
        return ("AI", energy_signal)
    except Exception as e:
        print(f"[API ERROR] AI signal fetch failed: {e}")
        return ("AI", 1.0)
