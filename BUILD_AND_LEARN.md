# Build the robot and learn the AI pipeline

## Where the project stands

The local code contains Vosk speech recognition, exact phrase-to-command mapping,
spoken responses, and OpenCV LBPH face enrollment/recognition. Movement handlers
had only spoken acknowledgements; they contained no servo or controller commands.
The Raspberry Pi folder contains a face setup document, not motor firmware.

The first improvements add dependency-free typed practice, Linux speech output,
optional camera loading, bounded audio buffering, rejection of unknown words,
and six prepared question/answer pairs. Stop bypasses the command cooldown, but
listening still pauses during speech. This is a software prototype, not a verified
walking robot.

The supplied [Orion project](https://github.com/AshishA26/Orion-Quadruped) describes
a 12-joint robot with Jetson high-level computing and STM32 real-time control.
Use its separation of responsibilities as a reference. Its mechanical dimensions,
servo calibration, firmware, and Jetson dependencies must be checked against our
actual hardware before reuse. Check licensing before copying source or designs.

## The shortest useful college demonstration

Aim for a dog that stands, walks slowly, stops reliably, accepts a few offline
commands, and reports one real agricultural measurement. Pick the measurement
after confirming available sensors. Soil moisture requires a probe in contact
with soil; a reading from a handheld probe is a simpler first demonstration than
automating probe placement. Do not claim crop disease diagnosis or autonomous
field navigation without implementing and evaluating those systems.

1. Confirm the deadline, Pi model/RAM/OS, servo model/count, controller, battery,
   frame assembly, microphone, speaker, and available sensors.
2. Validate one joint with conservative calibrated limits, then one leg, then
   supported standing. Record joint directions and neutral positions.
3. Establish slow walking through a dedicated gait controller. Add a physical
   stop and controller-side timeout before enabling voice movement commands.
4. Map approved voice intents to that controller's documented protocol. Require
   acknowledgements; a spoken response must distinguish a request from success.
5. Integrate one calibrated sensor and speak its actual reading, unit, and age.
6. Rehearse offline. Measure command accuracy and response time with several
   speakers, speaker echo, and background noise. Test invalid commands and loss
   of communication. Record failures as well as successes.

Face recognition, jumping, spinning, navigation, and a generative chatbot can
follow the basic demo. They do not need to delay standing, walking, and stopping.
No reliable completion date can be estimated until hardware and deadline are known.

## Lesson one: what a voice chatbot actually does

The pipeline is **sound -> text -> intent or answer -> action/result -> speech**.

- **Speech recognition (ASR):** Vosk turns microphone samples into words using an
  already trained model. Running that model is called inference.
- **Intent classification:** `classify_command` maps `go forward` to
  `move_forward`. Today this is a Python dictionary, not a trained classifier.
- **Robot control:** A future controller will translate that intent into gait
  targets, then joint angles. Inverse kinematics calculates joint angles for a
  desired foot position. Language generation should not set servo angles.
- **Dialogue:** `robot_faq.py` retrieves an exact prepared answer. This is a small
  rule-based chatbot; it does not learn new facts automatically or remember turns.
- **Speech synthesis (TTS):** `offline_speech.py` converts response text into sound.
  ASR and TTS are separate systems and neither requires an LLM.

Start with `python main.py --text`. Ask `what is soil moisture`, then rephrase it
as `explain soil moisture`. The second phrasing is unknown because the FAQ uses
exact matching. Add that phrasing as an alias and verify both work. This shows
the difference between string matching and understanding paraphrases.

## Growing toward an AI engineering portfolio by December 2026

Treat this as a learning sequence, not a guarantee of job readiness.

**October:** Practice Python modules, functions, dictionaries, exceptions, Git,
virtual environments, tests, and logging using this project. Learn sampling rate,
ASR, TTS, training versus inference, and how to measure command accuracy. Keep a
small evaluation set of command recordings with different speakers and noise.

**November:** Learn labeled datasets, train/validation/test separation, precision,
recall, and confusion matrices. Compare the current rules with a small intent
classifier on held-out utterances. Split recordings by speaker/session so near
duplicates do not leak between training and evaluation. Log sensor values and
evaluate face recognition on new sessions rather than training images.

**December:** Learn tokens, embeddings, retrieval, and language-model inference.
Build retrieval over a small set of reviewed agriculture notes. RAG retrieves
relevant passages and gives them to a language model to help produce an answer;
it does not guarantee correctness. Benchmark a local model on the actual Pi
before choosing it: RAM use, answer latency, temperature, and power matter.
Keep chat output separate from the validated motion interface.

Publish a reproducible setup, measured results, an architecture explanation, and
a short demo. Being able to explain failures and design choices is as important
as getting one successful demonstration.

## Sources and setup corrections

- [Vosk documentation](https://alphacephei.com/vosk/) documents offline operation,
  Raspberry Pi support, and configurable vocabulary.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) provides local speech synthesis.
- The older local setup guide describes sharing a virtual environment. Recreate
  environments from requirements on each machine instead.
- The guide's `opencv-python` package does not supply the `cv2.face` API used by
  this code; use the contrib package in a clean environment.
- Running `facerecognizer.py` alone currently defines functions but does not start
  recognition. Use the existing workflow or the voice command.
