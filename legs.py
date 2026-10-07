"""
12-servo quadruped leg controller, driven by a PCA9685 16-channel PWM
board over I2C.

Runs anywhere: if the servo hardware (or the adafruit libraries) can't be
found, e.g. on a Windows laptop, it falls back to SIMULATION mode and just
prints what the legs would do. On the Raspberry Pi with the PCA9685
connected it uses the real servos automatically.

Hardware assumptions (adjust LEG_CHANNELS below if your wiring differs):

    channel  leg            joint
    0        front-left     hip
    1        front-left     thigh
    2        front-left     knee
    3        front-right    hip
    4        front-right    thigh
    5        front-right    knee
    6        back-left      hip
    7        back-left      thigh
    8        back-left      knee
    9        back-right     hip
    10       back-right     thigh
    11       back-right     knee

Run calibrate_servos.py first - it writes servo_config.json, which this
module loads to translate logical +/- degree offsets into each servo's
real, calibrated angle (every physical build's "neutral" differs).
"""

import json
import os
import time

# Resolve the config next to this file so it's found no matter which
# directory the program is launched from (laptop vs. Pi, cron, etc.).
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_config.json")
NUM_CHANNELS = 16

LEG_CHANNELS = {
    "FL": {"hip": 0, "thigh": 1, "knee": 2},
    "FR": {"hip": 3, "thigh": 4, "knee": 5},
    "BL": {"hip": 6, "thigh": 7, "knee": 8},
    "BR": {"hip": 9, "thigh": 10, "knee": 11},
}

ALL_CHANNELS = [ch for joints in LEG_CHANNELS.values() for ch in joints.values()]


# =========================================================
# SERVO DRIVER (REAL HARDWARE OR SIMULATION)
# =========================================================

class _FakeServo:
    """Stand-in for a single servo when no hardware is present."""

    def __init__(self, channel):
        self.channel = channel
        self.angle = 90
        self.actuation_range = 180

    def set_pulse_width_range(self, *args, **kwargs):
        pass


class _FakeKit:
    """Stand-in for adafruit_servokit.ServoKit. Exposes .servo[channel]
    with the same .angle attribute, so the rest of the module (and
    calibrate_servos.py, if it does `from legs import kit`) works as is."""

    def __init__(self, channels=16):
        self.servo = [_FakeServo(i) for i in range(channels)]


try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=NUM_CHANNELS)
    SIMULATION = False
    print("[legs] PCA9685 found - using real servos.")
except Exception as e:
    kit = _FakeKit(NUM_CHANNELS)
    SIMULATION = True
    print(f"[legs] No servo hardware available ({type(e).__name__}: {e}).")
    print("[legs] Running in SIMULATION mode - movements are only printed.")


def _default_config():
    return {str(ch): {"neutral": 90, "invert": False} for ch in ALL_CHANNELS}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print("No servo_config.json found - using uncalibrated defaults (neutral=90).")
        print("Run calibrate_servos.py before trusting any movement.")
        return _default_config()
    with open(CONFIG_PATH) as f:
        return json.load(f)


CONFIG = load_config()

# Tracks the last commanded logical angle per joint so smooth_move() can
# interpolate from where the leg actually is, not from zero every time.
_LAST_POSE = {(leg, joint): 0 for leg in LEG_CHANNELS for joint in LEG_CHANNELS[leg]}


def _actual_angle(channel, delta):
    """Translate a logical +/- delta around 'neutral' into a real servo
    angle, respecting this channel's calibrated neutral point and
    direction (left/right legs are usually mirrored)."""

    cfg = CONFIG.get(str(channel), {"neutral": 90, "invert": False})
    sign = -1 if cfg.get("invert") else 1
    angle = cfg["neutral"] + sign * delta
    return max(0, min(180, angle))


def set_joint(leg, joint, delta):
    channel = LEG_CHANNELS[leg][joint]
    kit.servo[channel].angle = _actual_angle(channel, delta)


# =========================================================
# SMOOTH, SIMULTANEOUS MOVEMENT
# =========================================================
#
# Jumping every servo straight to its target angle looks jerky and puts a
# sudden current spike on all 12 motors at once. Interpolating every
# commanded channel together over a short duration gives visibly smoother
# motion and spreads out the current draw.

def smooth_move(targets, duration=0.3, steps=15):
    """targets: {(leg, joint): delta_angle, ...}"""

    start = {key: _LAST_POSE.get(key, 0) for key in targets}
    step_delay = duration / steps

    for i in range(1, steps + 1):
        t = i / steps
        for key, target_delta in targets.items():
            leg, joint = key
            interp = start[key] + (target_delta - start[key]) * t
            set_joint(leg, joint, interp)
        time.sleep(step_delay)

    _LAST_POSE.update(targets)

    if SIMULATION:
        # One compact line per move instead of 15 x 12 servo prints.
        summary = ", ".join(f"{leg}.{joint}={delta:+.0f}" for (leg, joint), delta in targets.items())
        print(f"[sim] move ({duration:.2f}s): {summary}")


# =========================================================
# POSES
# =========================================================
#
# All numbers below are offsets in degrees from each joint's calibrated
# neutral point, not absolute servo angles. Tune these five constants
# first - they control most of how correct the standing/sitting posture
# looks - before touching the gait functions further down.

HIP_NEUTRAL = 0
THIGH_STAND = 30
KNEE_STAND = -30
THIGH_SIT = 60
KNEE_SIT = -70
THIGH_LIFT = 50
KNEE_LIFT = -60
THIGH_STRIDE = 15


def _pose_all_legs(hip, thigh, knee):
    targets = {}
    for leg in LEG_CHANNELS:
        targets[(leg, "hip")] = hip
        targets[(leg, "thigh")] = thigh
        targets[(leg, "knee")] = knee
    return targets


def stand():
    smooth_move(_pose_all_legs(HIP_NEUTRAL, THIGH_STAND, KNEE_STAND), duration=0.4)


def sit():
    # Front legs stay standing, back legs tuck under.
    targets = {
        ("FL", "hip"): HIP_NEUTRAL, ("FL", "thigh"): THIGH_STAND, ("FL", "knee"): KNEE_STAND,
        ("FR", "hip"): HIP_NEUTRAL, ("FR", "thigh"): THIGH_STAND, ("FR", "knee"): KNEE_STAND,
        ("BL", "hip"): HIP_NEUTRAL, ("BL", "thigh"): THIGH_SIT, ("BL", "knee"): KNEE_SIT,
        ("BR", "hip"): HIP_NEUTRAL, ("BR", "thigh"): THIGH_SIT, ("BR", "knee"): KNEE_SIT,
    }
    smooth_move(targets, duration=0.5)


def rest():
    """Lower the body into a relaxed crouch. Call this when idle to cut
    servo holding current and heat instead of standing indefinitely."""
    smooth_move(_pose_all_legs(HIP_NEUTRAL, 70, -80), duration=0.5)


# =========================================================
# TROT GAIT
# =========================================================
#
# Diagonal pairs move together: (FL + BR) swing while (FR + BL) plant,
# then they swap. This is the simplest stable quadruped gait and only
# needs two alternating phases.

DIAGONAL_A = ["FL", "BR"]
DIAGONAL_B = ["FR", "BL"]


def _step_phase(swing_legs, stance_legs, direction=1):

    # Lift the swing legs and swing them forward (or back).
    targets = {}
    for leg in swing_legs:
        targets[(leg, "hip")] = HIP_NEUTRAL
        targets[(leg, "thigh")] = THIGH_LIFT
        targets[(leg, "knee")] = KNEE_LIFT
    for leg in stance_legs:
        targets[(leg, "hip")] = HIP_NEUTRAL
        targets[(leg, "thigh")] = THIGH_STAND - direction * THIGH_STRIDE
        targets[(leg, "knee")] = KNEE_STAND
    smooth_move(targets, duration=0.18)

    # Plant the swing legs down shifted forward - this is what propels
    # the body - while stance legs push back through neutral.
    targets = {}
    for leg in swing_legs:
        targets[(leg, "hip")] = HIP_NEUTRAL
        targets[(leg, "thigh")] = THIGH_STAND + direction * THIGH_STRIDE
        targets[(leg, "knee")] = KNEE_STAND
    for leg in stance_legs:
        targets[(leg, "hip")] = HIP_NEUTRAL
        targets[(leg, "thigh")] = THIGH_STAND
        targets[(leg, "knee")] = KNEE_STAND
    smooth_move(targets, duration=0.18)


def walk_forward(cycles=3):
    stand()
    for _ in range(cycles):
        _step_phase(DIAGONAL_A, DIAGONAL_B, direction=1)
        _step_phase(DIAGONAL_B, DIAGONAL_A, direction=1)
    stand()


def walk_backward(cycles=3):
    stand()
    for _ in range(cycles):
        _step_phase(DIAGONAL_A, DIAGONAL_B, direction=-1)
        _step_phase(DIAGONAL_B, DIAGONAL_A, direction=-1)
    stand()


def _turn(cycles, direction):
    """In-place turn: left-side and right-side legs stride in opposite
    directions, instead of the same direction used for forward/backward."""

    stand()
    left_legs = ["FL", "BL"]
    right_legs = ["FR", "BR"]

    for _ in range(cycles):
        targets = {}
        for leg in left_legs:
            targets[(leg, "hip")] = HIP_NEUTRAL
            targets[(leg, "thigh")] = THIGH_STAND + direction * THIGH_STRIDE
            targets[(leg, "knee")] = KNEE_STAND
        for leg in right_legs:
            targets[(leg, "hip")] = HIP_NEUTRAL
            targets[(leg, "thigh")] = THIGH_STAND - direction * THIGH_STRIDE
            targets[(leg, "knee")] = KNEE_STAND
        smooth_move(targets, duration=0.25)
        stand()


def turn_left(cycles=3):
    _turn(cycles, direction=-1)


def turn_right(cycles=3):
    _turn(cycles, direction=1)


def spin():
    turn_left(cycles=5)


def jump():
    crouch = _pose_all_legs(HIP_NEUTRAL, THIGH_LIFT, KNEE_LIFT)
    extend = _pose_all_legs(HIP_NEUTRAL, 10, -10)
    smooth_move(crouch, duration=0.2)
    smooth_move(extend, duration=0.12)
    stand()


def bark_wiggle():
    """No dedicated head/tail servo in a plain 12-channel build, so a
    quick front-leg bounce stands in for a 'bark' reaction."""

    up = {
        ("FL", "thigh"): THIGH_LIFT, ("FL", "knee"): KNEE_LIFT,
        ("FR", "thigh"): THIGH_LIFT, ("FR", "knee"): KNEE_LIFT,
    }
    smooth_move(up, duration=0.12)
    stand()


def shutdown():
    """Stop sending pulses on every channel. Setting a servo's angle to
    None cuts its PWM output, which both saves power and stops it from
    straining against nothing while the program isn't running."""

    for channel in ALL_CHANNELS:
        try:
            kit.servo[channel].angle = None
        except Exception:
            pass