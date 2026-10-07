"""Local speech output; no cloud services or network requests."""
import os
import shutil
import subprocess


def say(text):
    if os.name == "nt":
        # Pass speech as data through stdin rather than interpolate it into code.
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$voice.Speak([Console]::In.ReadToEnd())"
        )
        args = ["powershell", "-NoProfile", "-Command", command]
    else:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            raise RuntimeError("Install espeak-ng for offline spoken responses on Linux.")
        args = [executable, "--stdin"]
    subprocess.run(args, input=text, text=True, check=True, timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
