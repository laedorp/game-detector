"""Detection-driven aiming outputs, separate from manual controller precision."""

from .controller import (
    AimActivationError,
    AimActivationSensor,
    AimConfig,
    AimingController,
    AimingControllerError,
    TargetTracker,
    UdpAimingController,
    choose_target,
    head_target_point,
)
from .makcu import MakcuAimConfig, MakcuAimingController, MakcuError

__all__ = [
    "AimActivationError",
    "AimActivationSensor",
    "AimConfig",
    "AimingController",
    "AimingControllerError",
    "TargetTracker",
    "MakcuAimConfig",
    "MakcuAimingController",
    "MakcuError",
    "UdpAimingController",
    "choose_target",
    "head_target_point",
]