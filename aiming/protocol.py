"""Authenticated latest-state protocol for remote detection aiming."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import struct


MAGIC = b"GDA1"
MIN_PAIRING_KEY_LENGTH = 16
_PAYLOAD = struct.Struct("!4sIBff")
_TAG_SIZE = 16


class AimProtocolError(ValueError):
    """Raised when a remote aim packet is malformed or unauthenticated."""


@dataclass(frozen=True, slots=True)
class AimCommand:
    sequence: int
    active: bool
    x: float
    y: float

    def __post_init__(self) -> None:
        if not 0 <= self.sequence <= 0xFFFFFFFF:
            raise ValueError("sequence must fit in an unsigned 32-bit integer")
        if not math.isfinite(self.x) or not -1.0 <= self.x <= 1.0:
            raise ValueError("x must be finite and between -1 and 1")
        if not math.isfinite(self.y) or not -1.0 <= self.y <= 1.0:
            raise ValueError("y must be finite and between -1 and 1")


def validate_pairing_key(value: str) -> str:
    key = value.strip()
    if len(key) < MIN_PAIRING_KEY_LENGTH:
        raise ValueError(
            f"pairing key must contain at least {MIN_PAIRING_KEY_LENGTH} characters"
        )
    return key


def encode_aim_command(command: AimCommand, pairing_key: str) -> bytes:
    key = validate_pairing_key(pairing_key).encode("utf-8")
    payload = _PAYLOAD.pack(
        MAGIC,
        command.sequence,
        int(command.active),
        command.x,
        command.y,
    )
    tag = hmac.new(key, payload, hashlib.sha256).digest()[:_TAG_SIZE]
    return payload + tag


def decode_aim_command(packet: bytes, pairing_key: str) -> AimCommand:
    if len(packet) != _PAYLOAD.size + _TAG_SIZE:
        raise AimProtocolError("aim packet has an invalid length")
    key = validate_pairing_key(pairing_key).encode("utf-8")
    payload = packet[: _PAYLOAD.size]
    supplied_tag = packet[_PAYLOAD.size :]
    expected_tag = hmac.new(key, payload, hashlib.sha256).digest()[:_TAG_SIZE]
    if not hmac.compare_digest(supplied_tag, expected_tag):
        raise AimProtocolError("aim packet authentication failed")
    magic, sequence, active, x, y = _PAYLOAD.unpack(payload)
    if magic != MAGIC:
        raise AimProtocolError("aim packet has an unsupported protocol version")
    if active not in (0, 1):
        raise AimProtocolError("aim packet has an invalid active flag")
    try:
        return AimCommand(sequence, bool(active), x, y)
    except ValueError as exc:
        raise AimProtocolError(str(exc)) from exc


def is_newer_sequence(candidate: int, previous: int) -> bool:
    """Return whether candidate follows previous in wrapping uint32 order."""

    delta = (candidate - previous) & 0xFFFFFFFF
    return 0 < delta < 0x80000000