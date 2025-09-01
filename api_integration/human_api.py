import requests

def fetch_human_signal(api_url, params=None):
    """
    Fetches human-related data (e.g. sentiment, emotion, biometrics) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map sentiment score to energy signal
        sentiment = data.get("sentiment_score", 0.5)
        energy_signal = sentiment * 10  # Scale as needed
        return ("human", energy_signal)
    except Exception as e:
        print(f"[API ERROR] Human signal fetch failed: {e}")
        # Return baseline signal if API fails
        return ("human", 1.0)
