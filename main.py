import os
import sys
import json
import time
import queue
import random
import threading
import subprocess
import shutil
import atexit

import numpy as np
import sounddevice as sd

from vosk import Model, KaldiRecognizer, SetLogLevel

from facerecognizer import (
    recognize_face,
    init_camera,
    cleanup_camera,
)

import legs

SetLogLevel(-1)


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "vosk-model-small-en-us-0.15"

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

IDLE_TIMEOUT = 120
COMMAND_COOLDOWN = 1.5
WAKE_VOLUME_THRESHOLD = 100

# Microphone software amplification
MIC_GAIN = 3.0

TTS_RATE = 175
TTS_VOLUME = 1.0

audio_queue = queue.Queue()

last_command_time = 0
last_activity_time = time.time()

sleeping = False
face_thread_running = False
robot_speaking = False

movement_lock = threading.Lock()

atexit.register(cleanup_camera)
atexit.register(legs.shutdown)


# =========================================================
# TEXT TO SPEECH
# =========================================================
#
# One dedicated worker thread owns the speech engine, so speak() is safe
# to call from any thread (main loop, face-recognition thread, movement
# thread) and utterances never overlap.
#
# The backend is picked per platform, because pyttsx3 in a background
# thread can run "successfully" on Windows and produce no sound at all:
#
#   Windows : one long-lived PowerShell process hosting the built-in
#             SAPI voice (System.Speech). Launched once, so there is no
#             per-sentence startup cost.
#   Linux/Pi: espeak-ng / espeak  (sudo apt install espeak-ng)
#   macOS   : the built-in `say` command
#   fallback: pyttsx3, only if none of the above is available

_speech_queue = queue.Queue()


class _WindowsSpeaker:

    name = "Windows SAPI (persistent PowerShell)"

    def __init__(self):

        rate = max(-10, min(10, round((TTS_RATE - 175) / 15)))
        volume = max(0, min(100, int(TTS_VOLUME * 100)))

        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.Rate = " + str(rate) + ";"
            "$s.Volume = " + str(volume) + ";"
            "[Console]::Out.WriteLine('READY');"
            "while (($line = [Console]::In.ReadLine()) -ne $null) {"
            "  if ($line.Length -gt 0) { $s.Speak($line) };"
            "  [Console]::Out.WriteLine('DONE')"
            "}"
        )

        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        if self.proc.stdout.readline().strip() != "READY":
            raise RuntimeError("PowerShell speech host did not start")

    def say(self, text):

        clean = text.replace("\r", " ").replace("\n", " ")
        self.proc.stdin.write(clean + "\n")
        self.proc.stdin.flush()

        # Blocks until PowerShell reports the sentence finished playing.
        if self.proc.stdout.readline() == "":
            raise RuntimeError("PowerShell speech host exited")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


class _CommandSpeaker:
    """Runs an external TTS program once per sentence (espeak / say)."""

    def __init__(self, name, build_cmd):
        self.name = name
        self._build_cmd = build_cmd

    def say(self, text):
        result = subprocess.run(
            self._build_cmd(text),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "TTS command failed")

    def close(self):
        pass


class _Pyttsx3Speaker:

    name = "pyttsx3 (fallback)"

    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", TTS_RATE)
        self.engine.setProperty("volume", TTS_VOLUME)

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def close(self):
        pass


def _make_speaker():

    problems = []

    if sys.platform.startswith("win"):
        try:
            return _WindowsSpeaker()
        except Exception as e:
            problems.append(f"Windows SAPI: {e}")

    elif sys.platform == "darwin":
        if shutil.which("say"):
            return _CommandSpeaker(
                "macOS say",
                lambda t: ["say", "-r", str(TTS_RATE), t],
            )

    else:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if exe:
            return _CommandSpeaker(
                os.path.basename(exe),
                lambda t: [exe, "-s", str(TTS_RATE),
                           "-a", str(int(TTS_VOLUME * 100)), t],
            )
        problems.append("espeak not found (sudo apt install espeak-ng)")

    try:
        return _Pyttsx3Speaker()
    except Exception as e:
        problems.append(f"pyttsx3: {e}")

    print("WARNING: no working text-to-speech backend - text output only.")
    for p in problems:
        print("   -", p)

    return None


def _tts_worker():

    global robot_speaking

    speaker = _make_speaker()

    if speaker is not None:
        print(f"TTS backend: {speaker.name}")

    while True:

        text, done_event = _speech_queue.get()

        if text is None:
            break

        # An empty string is a "flush marker": the queue is FIFO, so by the
        # time the worker reaches it every earlier sentence has finished.
        if text == "":
            if done_event is not None:
                done_event.set()
            continue

        robot_speaking = True
        print(f"ROBODOG: {text}")

        try:
            if speaker is not None:
                speaker.say(text)
        except Exception as e:
            print("TTS ERROR:", e)

            # One restart attempt - e.g. if the PowerShell host died.
            try:
                if speaker is not None:
                    speaker.close()
                speaker = _make_speaker()
                if speaker is not None:
                    speaker.say(text)
            except Exception as e2:
                print("TTS restart failed:", e2)
        finally:
            robot_speaking = False
            if done_event is not None:
                done_event.set()

    if speaker is not None:
        speaker.close()


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text, wait=True):

    done_event = threading.Event() if wait else None
    _speech_queue.put((text, done_event))

    if wait:
        done_event.wait()


# =========================================================
# VOSK COMMAND GRAMMAR
# =========================================================

VOSK_COMMANDS = [
    "move forward", "go forward", "forward",
    "move backward", "go backward", "backward",
    "turn left", "left",
    "turn right", "right",
    "stop", "halt",
    "sit", "sit down",
    "stand", "stand up",
    "jump",
    "spin",
    "bark",
    "hello", "hi", "hey",
    "camera",
    "shut down", "exit", "quit", "stop program",
    "help", "commands",
    "[unk]",
]


# =========================================================
# COMMAND CLASSIFIER
# =========================================================

# Built once at import time instead of being rebuilt on every recognized
# utterance.
COMMAND_MAP = {
    "move forward": "move_forward",
    "go forward": "move_forward",
    "forward": "move_forward",

    "move backward": "move_backward",
    "go backward": "move_backward",
    "backward": "move_backward",

    "turn left": "turn_left",
    "left": "turn_left",

    "turn right": "turn_right",
    "right": "turn_right",

    "stop": "stop",
    "halt": "stop",

    "sit": "sit",
    "sit down": "sit",

    "stand": "stand",
    "stand up": "stand",

    "jump": "jump",

    "spin": "spin",

    "bark": "bark",

    "hello": "greet",
    "hi": "greet",
    "hey": "greet",

    "camera": "identify_person",

    "shut down": "shutdown",
    "exit": "shutdown",
    "quit": "shutdown",
    "stop program": "shutdown",

    "help": "help",
    "commands": "help",
}


def classify_command(text):
    return COMMAND_MAP.get(text.lower().strip(), "unknown")


# =========================================================
# AUDIO CALLBACK
# =========================================================

def audio_callback(indata, frames, time_info, status):

    if status:
        print("AUDIO STATUS:", status)

    audio_queue.put(bytes(indata))


# =========================================================
# AMPLIFY + VOLUME IN ONE PASS
# =========================================================
#
# The original code parsed the raw bytes into an int16 array twice: once
# in amplify_audio() and again in get_volume() on the amplified output.
# Doing both in a single numpy pass avoids that duplicate conversion on
# every audio block (every ~0.25s while listening).

def process_audio_block(data, gain):

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)

    if audio.size == 0:
        return data, 0

    audio *= gain
    np.clip(audio, -32768, 32767, out=audio)

    audio_i16 = audio.astype(np.int16)
    volume = np.mean(np.abs(audio_i16.astype(np.int32)))

    return audio_i16.tobytes(), volume


# =========================================================
# FACE RECOGNITION
# =========================================================

def start_face_recognition():

    global face_thread_running

    if face_thread_running:
        speak("Face recognition is already running")
        return

    face_thread_running = True

    def worker():

        global face_thread_running

        try:
            speak("Let me see who you are")

            name = recognize_face()

            if name != "unknown":

                responses = [
                    f"Hello {name}",
                    f"Nice to see you {name}",
                    f"I recognize you {name}",
                ]

                speak(random.choice(responses))

            else:
                speak("I do not recognize you yet")

        except Exception as e:
            print("Face recognition error:", e)
            speak("Face recognition failed")

        finally:
            face_thread_running = False

    threading.Thread(target=worker, daemon=True).start()


# =========================================================
# ACTION EXECUTION
# =========================================================
#
# Movement intents are routed through the physical legs (see legs.py,
# PCA9685-driven). They run on a background thread, guarded by
# movement_lock, so a new command can't yank the servos mid-gait while a
# previous one is still executing, and so a multi-second walk cycle
# doesn't stall the audio loop reading the microphone.

MOVEMENT_ACTIONS = {
    "move_forward": (lambda: legs.walk_forward(3), "Moving forward"),
    "move_backward": (lambda: legs.walk_backward(3), "Moving backward"),
    "turn_left": (lambda: legs.turn_left(3), "Turning left"),
    "turn_right": (lambda: legs.turn_right(3), "Turning right"),
    "stop": (lambda: legs.stand(), "Stopping"),
    "sit": (lambda: legs.sit(), "Sitting"),
    "stand": (lambda: legs.stand(), "Standing"),
    "jump": (lambda: legs.jump(), "Jumping"),
    "spin": (lambda: legs.spin(), "Spinning"),
    "bark": (lambda: legs.bark_wiggle(), "Woof woof"),
}

ACTION_RESPONSES = {
    "greet": "Hello. I am ready.",
    "help": (
        "You can ask me to move forward, move backward, turn left, "
        "turn right, stop, sit, stand, jump, spin, bark, say camera to see who "
        "I am looking at, or say shut down to turn me off"
    ),
}


def run_movement(func, announcement=None):

    if not movement_lock.acquire(blocking=False):
        speak("I'm already moving, hang on")
        return

    def worker():
        try:
            if announcement:
                speak(announcement, wait=False)
            func()
        except Exception as e:
            print("Movement error:", e)
        finally:
            movement_lock.release()

    threading.Thread(target=worker, daemon=True).start()


def execute_action(intent):

    if intent == "identify_person":
        start_face_recognition()
        return

    if intent in MOVEMENT_ACTIONS:
        func, announcement = MOVEMENT_ACTIONS[intent]
        run_movement(func, announcement=announcement)
        return

    response = ACTION_RESPONSES.get(intent)

    if response is not None:
        speak(response)


# =========================================================
# GRACEFUL SHUTDOWN (voice command)
# =========================================================

def shutdown_robot():

    # Let any gait that is mid-step finish, so the legs are not cut off
    # halfway through a movement.
    if movement_lock.acquire(timeout=10):
        try:
            speak("Shutting down. Goodbye", wait=False)
            legs.rest()
        except Exception as e:
            print("Shutdown movement error:", e)
        finally:
            movement_lock.release()
    else:
        speak("Shutting down. Goodbye", wait=False)

    # Block until the farewell has finished playing, before the speech
    # thread is stopped in main()'s finally block.
    speak("", wait=True)


# =========================================================
# LOAD VOSK MODEL
# =========================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Vosk model not found: {MODEL_PATH}")
        sys.exit(1)

    print("Loading offline voice model...")
    model = Model(MODEL_PATH)
    print("Voice model loaded")

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    global last_command_time
    global last_activity_time
    global sleeping
    global robot_speaking

    print("=" * 60)
    print("ROBODOG OFFLINE SYSTEM")
    print("=" * 60)

    # -----------------------------------------------------
    # CAMERA + VOICE MODEL INITIALIZATION IN PARALLEL
    # -----------------------------------------------------
    # These two setup steps are independent and both take real wall-clock
    # time (camera warmup frames, loading the Vosk model into memory), so
    # running them concurrently instead of back-to-back shaves real time
    # off startup.

    print("Initializing camera...")

    camera_status = {}

    def _camera_worker():
        camera_status["ok"] = init_camera()

    camera_thread = threading.Thread(target=_camera_worker)
    camera_thread.start()

    print("Initializing voice system...")
    model = load_model()

    camera_thread.join()
    print("Camera ready" if camera_status.get("ok") else "Camera failed to initialize")

    command_grammar = json.dumps(VOSK_COMMANDS)
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, command_grammar)

    # -----------------------------------------------------
    # STARTUP MESSAGE
    # -----------------------------------------------------

    run_movement(legs.stand)
    speak("Robodog is ready")

    print("Listening...")
    print("Say a command such as: move forward")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):

            # =============================================
            # MAIN LOOP
            # =============================================

            while True:

                current_time = time.time()

                # -----------------------------------------
                # ENTER IDLE MODE AFTER 2 MINUTES
                # -----------------------------------------

                if not sleeping and current_time - last_activity_time > IDLE_TIMEOUT:
                    sleeping = True
                    print("\nROBODOG ENTERING IDLE MODE")
                    run_movement(legs.rest)

                # -----------------------------------------
                # GET MICROPHONE DATA
                # -----------------------------------------

                data = audio_queue.get()

                # -----------------------------------------
                # AMPLIFY + GET VOLUME IN ONE PASS
                # -----------------------------------------

                data, volume = process_audio_block(data, MIC_GAIN)

                # -----------------------------------------
                # WAKE UP FROM IDLE MODE
                # -----------------------------------------

                if sleeping and volume > WAKE_VOLUME_THRESHOLD:
                    sleeping = False
                    last_activity_time = time.time()
                    print("\nROBODOG WOKE UP")
                    run_movement(legs.stand)

                # -----------------------------------------
                # IGNORE MICROPHONE WHILE SPEAKING
                # -----------------------------------------

                if robot_speaking:
                    recognizer.Reset()
                    continue

                # -----------------------------------------
                # SEND AUDIO TO VOSK
                # -----------------------------------------

                if recognizer.AcceptWaveform(data):

                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").lower().strip()

                    if not text:
                        continue

                    text = text.replace("[unk]", "").strip()

                    if not text:
                        continue

                    print()
                    print("RECOGNIZED TEXT:", text)

                    intent = classify_command(text)
                    print("CLASSIFIED INTENT:", intent)

                    if intent == "unknown":
                        continue

                    # -------------------------------------
                    # SHUTDOWN - ends main.py cleanly
                    # -------------------------------------
                    # Checked before the cooldown so it can't be swallowed
                    # by a command spoken just before it. Plain "stop" still
                    # only halts movement; it does not exit the program.

                    if intent == "shutdown":
                        print()
                        print("COMMAND: shutdown")
                        shutdown_robot()
                        break

                    current_time = time.time()

                    # -------------------------------------
                    # PREVENT DUPLICATE COMMANDS
                    # -------------------------------------

                    if current_time - last_command_time < COMMAND_COOLDOWN:
                        continue

                    last_command_time = current_time
                    last_activity_time = current_time

                    print()
                    print(f"YOU SAID: {text}")
                    print(f"COMMAND: {intent}")

                    execute_action(intent)

    except KeyboardInterrupt:
        print("\nStopping ROBODOG...")
        speak("Goodbye")

    finally:
        cleanup_camera()
        _speech_queue.put((None, None))
        _tts_thread.join(timeout=3)


# =========================================================
# START PROGRAM
# =========================================================

def tts_test():
    """python main.py --tts-test  ->  checks you can actually HEAR the dog."""

    print("Speaking test sentences - you should hear all three.")
    speak("Audio test one. Robodog is ready.")
    speak("Audio test two. Moving forward.")
    speak("Audio test three. Woof woof.")
    print("Done. If you heard nothing, check the speaker/volume and the")
    print("'TTS backend' / 'TTS ERROR' lines printed above.")
    _speech_queue.put((None, None))
    _tts_thread.join(timeout=3)


if __name__ == "__main__":
    if "--tts-test" in sys.argv:
        tts_test()
    else:
        main()