"""Curated offline answers. This module never controls motors."""
ANSWERS = {
    "what is your name": "My name is RoboDog. I am an offline college robot project.",
    "what can you do": "I recognize voice commands and speak offline. Motor control is not connected yet.",
    "what is soil moisture": "Soil moisture is the water present in soil. A calibrated sensor can help measure it.",
    "read soil moisture": "A soil moisture sensor is not connected to this program yet.",
    "what is your battery level": "Battery monitoring is not connected to this program yet.",
    "are you online": "My voice recognition and prepared answers run offline after setup.",
}


def answer(text):
    return ANSWERS.get(" ".join(text.lower().split()))
