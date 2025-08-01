def generate_environment_suggestion(data):
    temp = data.get("temperature", 0)
    hum = data.get("humidity", 0)
    light = data.get("light", 0)

    advice = []

    if not 18 <= temp <= 24:
        advice.append(f"Recommended to keep temperature between 18–24°C; current value is {temp}°C.")
    if not 45 <= hum <= 60:
        advice.append(f"Recommended to keep humidity between 45%–60%; current value is {hum}%.")
    if light > 1000:
        advice.append("Light intensity is too strong; it's advised to use shading film or control lighting duration.")

    return advice or ["Environmental parameters are within the ideal range; no adjustment needed."]