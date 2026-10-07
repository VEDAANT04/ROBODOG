import pyttsx3
import time

engine = pyttsx3.init()

engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

print("Speaking message 1")
engine.say("Robodog is ready")
engine.runAndWait()

time.sleep(2)

print("Speaking message 2")
engine.say("Moving forward")
engine.runAndWait()

time.sleep(2)

print("Speaking message 3")
engine.say("Turning left")
engine.runAndWait()

print("Test finished")