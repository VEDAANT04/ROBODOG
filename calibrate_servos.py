"""
Interactive calibration for the 12-servo PCA9685 setup used by legs.py.

Run this FIRST, before anything else. Every physical build has slightly
different horn alignment, so "neutral = 90 degrees" is only a starting
guess until confirmed here.

For each channel:
  <number>  move to that angle (0-180) and look at the leg
  s         save the last angle you tried as this channel's neutral
  i         toggle direction inversion for this channel (left/right legs
            are usually mirrored - this is how you tell legs.py which way
            "forward" means for this particular servo)
  n         move on to the next channel
  q         save everything and quit
"""

import json
import os

import legs  # shares the same kit, config path and simulation fallback

CONFIG_PATH = legs.CONFIG_PATH
NUM_CHANNELS = 16

CHANNEL_LABELS = {
    0: "Front-Left hip", 1: "Front-Left thigh", 2: "Front-Left knee",
    3: "Front-Right hip", 4: "Front-Right thigh", 5: "Front-Right knee",
    6: "Back-Left hip", 7: "Back-Left thigh", 8: "Back-Left knee",
    9: "Back-Right hip", 10: "Back-Right thigh", 11: "Back-Right knee",
}

kit = legs.kit


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {str(ch): {"neutral": 90, "invert": False} for ch in CHANNEL_LABELS}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved {CONFIG_PATH}")


def main():
    config = load_config()

    print("=" * 60)
    print("ROBODOG SERVO CALIBRATION")
    print("=" * 60)
    if legs.SIMULATION:
        print("NOTE: simulation mode - no real servos will move.")
    print("Commands: <angle 0-180> / s (save neutral) / i (invert) / n (next) / q (quit)")
    print()

    for channel in sorted(CHANNEL_LABELS):

        label = CHANNEL_LABELS[channel]
        cfg = config.setdefault(str(channel), {"neutral": 90, "invert": False})
        last_tried = cfg["neutral"]

        kit.servo[channel].angle = last_tried

        print(f"--- Channel {channel}: {label} "
              f"(current neutral={cfg['neutral']}, invert={cfg['invert']}) ---")

        while True:
            cmd = input("angle / s / i / n / q > ").strip().lower()

            if cmd == "q":
                save_config(config)
                return

            if cmd == "n":
                break

            if cmd == "i":
                cfg["invert"] = not cfg["invert"]
                print(f"  invert is now {cfg['invert']}")
                continue

            if cmd == "s":
                cfg["neutral"] = last_tried
                print(f"  saved neutral={last_tried} for channel {channel}")
                continue

            try:
                angle = max(0, min(180, int(cmd)))
            except ValueError:
                print("  enter a number 0-180, or s / i / n / q")
                continue

            last_tried = angle
            kit.servo[channel].angle = angle

    save_config(config)
    print("All channels calibrated.")


if __name__ == "__main__":
    main()