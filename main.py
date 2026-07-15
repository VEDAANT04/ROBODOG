import queue
import sounddevice as sd
import json
import os
import time
import re
import random
import subprocess
import threading
import atexit
from difflib import SequenceMatcher

# ✅ FIX 1: Corrected import with proper space
from facerecognizer import recognize_face, init_camera, cleanup_camera

# ✅ FIX 2 & 3: Proper vosk import with protection
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)  # Safe after import
except ImportError:
    print("❌ Vosk not installed. Install with: pip install vosk")
    exit(1)

# ✅ Register cleanup function
atexit.register(cleanup_camera)

# ✅ OPTIMIZATION 1: TTS caching and selection
TTS_ENGINE = None

def init_tts():
    """Initialize TTS once at startup"""
    global TTS_ENGINE
    
    # Try win32com first (fastest on Windows)
    try:
        from win32com.client import Dispatch
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Rate = 0
        speaker.Volume = 100
        TTS_ENGINE = ("win32com", speaker)
        print("✅ Using win32com for TTS")
        return
    except:
        pass
    
    # Try pyttsx3 (reliable fallback)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        TTS_ENGINE = ("pyttsx3", engine)
        print("✅ Using pyttsx3 for TTS")
        return
    except:
        pass
    
    # PowerShell as last resort
    TTS_ENGINE = ("powershell", None)
    print("✅ Using PowerShell for TTS")

def speak(text):
    """Fast, reliable text-to-speech"""
    print(f"🐕 Dog says: {text}")
    
    if TTS_ENGINE is None:
        init_tts()
    
    engine_type, engine = TTS_ENGINE
    
    try:
        if engine_type == "win32com":
            engine.Speak(text)
        elif engine_type == "pyttsx3":
            engine.say(text)
            engine.runAndWait()
        else:  # powershell
            # ✅ FIX 4: Escape quotes to prevent injection
            escaped_text = text.replace('"', '\\"')
            cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{escaped_text}")'
            subprocess.Popen(
                ['powershell', '-Command', cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            time.sleep(0.5)
    except Exception as e:
        print(f"   (TTS error: {str(e)[:30]})")

print("\n🔊 Testing audio output...")
init_tts()
speak("Hello robodog ready")
time.sleep(1)

# ✅ Initialize camera early (for faster face recognition later)
print("\n📹 Initializing camera...")
init_camera()
print()

CONFIG = {
    "voice_confidence_threshold": 0.70,
    "min_word_length": 2,
    "max_word_count": 12,
    "debug_mode": False,
}

GREETING_RESPONSES = {
    "vedant": [
        "Hey Vedant! Great to see you!",
        "Wassup Vedant! Welcome back!",
        "Hello Vedant! Good to see you!",
        "Vedant! Nice to see you again!",
        "Hey buddy! Good going!",
        "Welcome back Vedant! How's it going?",
    ],
    "vedaant": [
        "Hey Vedant! Great to see you!",
        "Wassup Vedant! Welcome back!",
        "Hello Vedant! Good to see you!",
        "Vedant! Nice to see you again!",
        "Hey buddy! Good going!",
        "Welcome back Vedant! How's it going?",
    ],
    "unknown": [
        "Nice to meet you! What's your name?",
        "Hello there! Who are you?",
        "Wassup! I don't think we've met!",
        "Hey! New friend here?",
        "Good to see you! Do I know you?",
        "Hello! Welcome! I'm robodog!",
    ]
}

# ═══════════════════════════════════════════════════════════════════
# 🧹 TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════
def clean_text(text):
    """Clean garbled voice recognition text"""
    if not text:
        return ""
    
    text = text.strip()
    
    # ✅ OPTIMIZATION 2: More efficient regex patterns
    text = re.sub(r'\w*\.{2,}\)?', '', text)  # Better pattern
    text = re.sub(r'\w+ing\)', '', text)
    text = re.sub(r'\).*', '', text)
    text = re.sub(r'[)(\[\]{}!@#$%^&*+=<>?|`~]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    text = ' '.join(text.split())
    text = text.lower().strip()
    
    # Remove short words
    words = [w for w in text.split() if len(w) >= 1]
    text = ' '.join(words)
    
    return text

def is_noise(text):
    """Detect noise/garbage"""
    if not text:
        return True
    
    words = text.split()
    word_count = len(words)
    
    if word_count < 1 or word_count > CONFIG["max_word_count"]:
        return True
    
    if len(text) < CONFIG["min_word_length"]:
        return True
    
    return False

# ═══════════════════════════════════════════════════════════════════
# 🧠 INTENT CLASSIFIER (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════
class TunedIntentClassifier:
    """Smart intent classification with caching"""
    
    def __init__(self):
        # ✅ OPTIMIZATION 3: Pre-compile patterns
        self.intents = {
            "move_forward": ["forward", "ahead", "go", "move", "proceed"],
            "move_backward": ["back", "backward", "reverse"],
            "turn_left": ["left", "turn left"],
            "turn_right": ["right", "turn right"],
            "stop": ["stop", "halt", "pause"],
            "sit": ["sit"],
            "stand": ["stand"],
            "jump": ["jump"],
            "spin": ["spin"],
            "bark": ["bark", "woof"],
            "greet": ["hello", "hi", "hey", "greet"],
            "help": ["help", "commands"],
            "identify_person": ["recognize", "identify", "who", "face"],
        }
        
        # ✅ OPTIMIZATION 4: Cache for recent classifications
        self.cache = {}
        self.cache_size = 50
    
    def similarity(self, a, b):
        """Calculate string similarity (0-1)"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def classify(self, text):
        """Classify intent from text with cache"""
        text = text.lower().strip()
        
        if not text:
            return "unknown"
        
        # ✅ OPTIMIZATION 5: Check cache first
        if text in self.cache:
            return self.cache[text]
        
        # EXACT MATCH (fastest)
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text:
                    self.cache[text] = intent
                    if len(self.cache) > self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    return intent
        
        # FUZZY MATCH (only if exact fails)
        best_intent = "unknown"
        best_score = 0
        
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                sim = self.similarity(text, keyword)
                if sim > best_score:
                    best_score = sim
                    best_intent = intent
        
        if best_score > CONFIG["voice_confidence_threshold"]:
            self.cache[text] = best_intent
            return best_intent
        
        return "unknown"

classifier = TunedIntentClassifier()

# ═══════════════════════════════════════════════════════════════════
# 🐕 ACTIONS (WITH THREADING FOR FACE RECOGNITION)
# ═══════════════════════════════════════════════════════════════════
def execute_action(intent):
    """Execute the recognized action"""
    
    if intent == "move_forward":
        speak("Moving forward")
        
    elif intent == "move_backward":
        speak("Moving backward")
        
    elif intent == "turn_left":
        speak("Turning left")
        
    elif intent == "turn_right":
        speak("Turning right")
        
    elif intent == "stop":
        speak("Stopping")
        
    elif intent == "sit":
        speak("Sitting down")
        
    elif intent == "stand":
        speak("Standing up")
        
    elif intent == "jump":
        speak("Jumping")
        
    elif intent == "spin":
        speak("Spinning around")
        
    elif intent == "bark":
        speak("Woof woof!")
        
    # ✨ GREETING WITH FACE RECOGNITION (Non-blocking)
    elif intent == "greet":
        speak("Let me see who you are")
        
        # ✅ OPTIMIZATION 6: Run face recognition in background thread
        def greet_thread():
            try:
                print("\n📹 Opening camera...\n")
                name = recognize_face()
                
                if name in GREETING_RESPONSES:
                    greeting = random.choice(GREETING_RESPONSES[name])
                else:
                    greeting = random.choice(GREETING_RESPONSES["unknown"])
                
                speak(greeting)
            except Exception as e:
                print(f"   ⚠️  Face recognition error: {str(e)[:50]}")
                speak("Sorry, camera not available")
        
        thread = threading.Thread(target=greet_thread, daemon=True)
        thread.start()
        # ✅ PERFORMANCE FIX: REMOVED thread.join(timeout=10)
        # This was blocking the main voice loop!
        
    elif intent == "help":
        speak("I can move forward and backward, turn, sit, stand, jump, spin, and bark. Just say hello!")
        
    elif intent == "identify_person":
        speak("Let me see who you are")
        
        # ✅ OPTIMIZATION 7: Same threading approach
        def identify_thread():
            try:
                print("\n📹 Opening camera...\n")
                name = recognize_face()
                
                if name in ["vedant", "vedaant"]:
                    speak("You are Vedaant!")
                elif name != "unknown":
                    speak(f"You are {name}!")
                else:
                    speak("I don't recognize you yet")
            except Exception as e:
                print(f"   ⚠️  Face recognition error: {str(e)[:50]}")
                speak("Camera not available")
        
        thread = threading.Thread(target=identify_thread, daemon=True)
        thread.start()
        # ✅ PERFORMANCE FIX: REMOVED thread.join(timeout=10)
        # This was blocking the main voice loop!

# ═══════════════════════════════════════════════════════════════════
# 🎤 VOSK VOICE SETUP
# ═══════════════════════════════════════════════════════════════════
def load_voice_model():
    """Load Vosk model"""
    
    model_name = "vosk-model-small-en-us-0.15"
    
    # ✅ FIX 7: Check if model exists before loading
    if not os.path.exists(model_name):
        print(f"\n❌ Model '{model_name}' not found!")
        print("📥 Download from: https://alphacephei.com/vosk/models")
        print(f"📦 Extract to: {os.getcwd()}\n")
        exit(1)
    
    try:
        print(f"\n🔄 Loading {model_name}...")
        model = Model(model_name)
        print(f"✅ Using: {model_name}\n")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        exit(1)

# Load model
try:
    model = load_voice_model()
    recognizer = KaldiRecognizer(model, 16000)
    
    # ✅ OPTIMIZATION 8: Set word hints for better recognition
    recognizer.SetWords([
        "forward", "backward", "left", "right", "stop",
        "sit", "stand", "jump", "spin", "bark",
        "hello", "hi", "hey", "help", "recognize",
        "identify", "who", "face"
    ])
    
except Exception as e:
    print(f"❌ Vosk error: {e}")
    exit(1)

# Audio input setup
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # ✅ FIX 8: Don't print in callback - blocks audio capture
    if status:
        pass  # Silent (removed print statement)
    q.put(bytes(indata))

# ═══════════════════════════════════════════════════════════════════
# 🎯 MAIN VOICE LOOP
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🎤 ROBODOG VOICE RECOGNITION ACTIVE")
print("=" * 60)
print("Commands: hello, move forward, sit, bark, recognize me")
print("Press Ctrl+C to stop\n")

try:
    # ✅ PERFORMANCE FIX: Changed blocksize from 8000 to 4000
    # This reduces audio buffer latency from 500ms to 250ms
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=4000,  # ← Changed from 8000 to 4000
        dtype='int16',
        channels=1,
        callback=audio_callback
    ):
        
        print("✅ Listening...\n")
        
        last_text = ""
        last_intent = "unknown"
        
        while True:
            data = q.get()
            
            if recognizer.AcceptWaveform(data):
                result_json = recognizer.Result()
                
                try:
                    result = json.loads(result_json)
                    raw_text = result.get("text", "").strip()
                    
                    # Clean the text
                    text = clean_text(raw_text)
                    
                    if not text:
                        continue
                    
                    # ✅ OPTIMIZATION 9: Skip duplicate commands
                    if text.lower() == last_text.lower():
                        continue
                    
                    last_text = text
                    
                    # Skip if noise
                    if is_noise(text):
                        continue
                    
                    # Classify and execute
                    intent = classifier.classify(text)
                    
                    if intent != "unknown":
                        print(f"👂 You said: {text}")
                        execute_action(intent)
                        print()
                    
                except json.JSONDecodeError:
                    pass
            
            # Show partial while listening (less frequently)
            else:
                try:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text and len(partial_text) > 3:
                        cleaned = clean_text(partial_text)
                        if cleaned:
                            print(f"   (listening: {cleaned[:30]}...)", end="\r")
                except:
                    pass

except KeyboardInterrupt:
    print("\n\n👋 Goodbye!")
    speak("Goodbye see you")

except Exception as e:
    print(f"\n❌ Error: {e}")

finally:
    # ✅ Cleanup camera on exit
    cleanup_camera()