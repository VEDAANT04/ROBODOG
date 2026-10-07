import os
import sys
import json
import time
import queue
import random
import threading
import atexit

import numpy as np
import sounddevice as sd
from tts import say as _say

from vosk import Model, KaldiRecognizer, SetLogLevel

# Face recognition is optional: if OpenCV isn't installed, the voice and
# movement features still run and 'recognize me' just says it can't.
try:
    from facerecognizer import (
        recognize_face,
        init_camera,
        cleanup_camera,
    )
except Exception as _face_import_error:
    print(f"[face] Face recognition disabled ({_face_import_error})")

    def recognize_face(*args, **kwargs):
        return "unknown"

    def init_camera():
        return False

    def cleanup_camera():
        pass

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
# TEXT TO SPEECH - PERSISTENT PYTTSX3 ENGINE
# =========================================================
#
# The original implementation spawned a new PowerShell process and loaded
# the .NET speech assembly for every single utterance, which typically
# costs 0.5-1.5s of pure startup overhead before a single word is spoken.
# A single long-lived pyttsx3 engine, driven from one dedicated worker
# thread (engines are not thread-safe to call concurrently), removes that
# cost almost entirely while keeping the same blocking speak() interface
# the rest of the program relies on.

_speech_queue = queue.Queue()


def _tts_worker():

    global robot_speaking

    while True:

        text, done_event = _speech_queue.get()

        if text is None:
            break

        robot_speaking = True
        print(f"ROBODOG: {text}")

        try:
            _say(text)
        except Exception as e:
            print("TTS ERROR:", e)
        finally:
            robot_speaking = False
            if done_event is not None:
                done_event.set()


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
    "hello", "hi", "hey", "camera",
    "recognize me", "identify me", "who am i", "recognize face",
    "help", "commands",
    "off", "power off",
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

    # Greetings and "camera" all trigger face recognition, which then
    # greets the person by name if they are in the dataset.
    "hello": "identify_person",
    "hi": "identify_person",
    "hey": "identify_person",
    "camera": "identify_person",

    "recognize me": "identify_person",
    "identify me": "identify_person",
    "who am i": "identify_person",
    "recognize face": "identify_person",

    "help": "help",
    "commands": "help",

    "off": "power_off",
    "power off": "power_off",
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
                speak("Hello. I do not recognize you yet")

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
        "turn right, stop, sit, stand, jump, spin, bark, say camera to "
        "recognize you, or say off to shut me down"
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


def power_off():
    """Say goodbye, let any movement in progress finish, crouch, and
    return so main() can exit (atexit then releases the servos)."""

    speak("Powering off. Goodbye")

    if movement_lock.acquire(timeout=15):
        try:
            legs.rest()
        except Exception as e:
            print("Rest error:", e)
        finally:
            movement_lock.release()


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

                    if intent == "power_off":
                        power_off()
                        break

                    execute_action(intent)

    except KeyboardInterrupt:
        print("\nStopping ROBODOG...")
        speak("Goodbye")

    finally:
        cleanup_camera()
        _speech_queue.put((None, None))


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":
    main()