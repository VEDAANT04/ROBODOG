# RoboDog offline college prototype

The current program recognizes a fixed vocabulary, speaks locally, answers a few
prepared questions, and optionally recognizes enrolled faces. **It does not drive
motors.** Movement requests are explicitly reported as a demo.

## Try it now without hardware

From this folder:

```powershell
python main.py --text
```

Try `hello`, `move forward`, `stop`, `what is your name`, `what is soil moisture`,
`read soil moisture`, and `quit`. This mode needs only Python and prints replies.
It is a curated FAQ chatbot, not a generative language model.

## Microphone and spoken responses

Create a fresh environment on each machine. The existing Windows `venv` points
to an unavailable Python installation; do not copy it to the Raspberry Pi.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

Linux / Raspberry Pi (setup requires package downloads):

```bash
sudo apt-get install python3-venv libportaudio2 espeak-ng
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Keep `vosk-model-small-en-us-0.15` beside `main.py`. It is already present in this
workspace. Speech recognition and synthesis run locally once dependencies and
the model are installed. Windows uses System.Speech; Linux uses eSpeak NG.
The default microphone must support mono 16 kHz recording. Actual audio operation
and Linux deployment still need hardware verification.

Face recognition is optional and loads only when requested. In a fresh environment
install `requirements-face.txt` instead of `requirements.txt`, then run
`python run_workflow.py` from the project folder to collect and train faces.
The current face code needs `opencv-contrib-python` for `cv2.face`. Avoid installing
multiple OpenCV wheel variants in one environment. A Raspberry Pi CSI camera may
need a different capture backend; the current code uses OpenCV camera index 0.

## Current limitations

- No motor transport, gait controller, calibration, battery input, or agricultural
  sensor integration is implemented here.
- Listening pauses during speech. A spoken stop cannot interrupt a reply. Before
  motor integration, implement controller-side command expiry and a physical stop.
- The recognition grammar accepts fixed phrases. Rejecting `[unk]` reduces one
  failure mode but does not guarantee correct recognition in field noise.
- Idle mode is an activity indicator, not an explicit wake-word requirement.
- The FAQ provides general explanations and reports missing sensors honestly.
- Face recognition and live microphone/speaker performance have not been validated
  in this review.

Run software regression checks with `python -m unittest discover -s tests -v`.
See [BUILD_AND_LEARN.md](BUILD_AND_LEARN.md) for the build sequence and lessons.
