"""Small Linux input-code subset used by the controller worker.

The numeric values are part of Linux's stable userspace input ABI.  Keeping
them here lets the curve and event mapper remain importable on systems where
``python-evdev`` is not installed (including Windows packaging hosts).
"""

# Event types and synchronization codes.
EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03
SYN_REPORT = 0
SYN_DROPPED = 3

# Standard absolute axes.
ABS_X = 0x00
ABS_Y = 0x01
ABS_Z = 0x02
ABS_RX = 0x03
ABS_RY = 0x04
ABS_RZ = 0x05
ABS_GAS = 0x09
ABS_BRAKE = 0x0A
ABS_HAT0X = 0x10
ABS_HAT0Y = 0x11

# Standard gamepad buttons.
BTN_TL2 = 0x138

ABSOLUTE_CODE_NAMES = {
    ABS_X: "ABS_X",
    ABS_Y: "ABS_Y",
    ABS_Z: "ABS_Z",
    ABS_RX: "ABS_RX",
    ABS_RY: "ABS_RY",
    ABS_RZ: "ABS_RZ",
    ABS_GAS: "ABS_GAS",
    ABS_BRAKE: "ABS_BRAKE",
    ABS_HAT0X: "ABS_HAT0X",
    ABS_HAT0Y: "ABS_HAT0Y",
}

ABSOLUTE_NAME_CODES = {name: code for code, name in ABSOLUTE_CODE_NAMES.items()}
