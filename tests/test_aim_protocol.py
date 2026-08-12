from __future__ import annotations

import unittest

from aiming.protocol import (
    AimCommand,
    AimProtocolError,
    decode_aim_command,
    encode_aim_command,
    is_newer_sequence,
)


PAIRING_KEY = "0123456789abcdef0123456789abcdef"


class AimProtocolTests(unittest.TestCase):
    def test_authenticated_command_round_trips(self) -> None:
        original = AimCommand(42, True, -0.25, 0.75)

        decoded = decode_aim_command(
            encode_aim_command(original, PAIRING_KEY),
            PAIRING_KEY,
        )

        self.assertEqual(decoded.sequence, original.sequence)
        self.assertEqual(decoded.active, original.active)
        self.assertAlmostEqual(decoded.x, original.x)
        self.assertAlmostEqual(decoded.y, original.y)

    def test_wrong_key_and_tampering_are_rejected(self) -> None:
        packet = encode_aim_command(AimCommand(1, True, 0.1, -0.2), PAIRING_KEY)
        with self.assertRaises(AimProtocolError):
            decode_aim_command(packet, "fedcba9876543210fedcba9876543210")
        damaged = bytearray(packet)
        damaged[8] ^= 1
        with self.assertRaises(AimProtocolError):
            decode_aim_command(bytes(damaged), PAIRING_KEY)

    def test_sequence_order_handles_wrap_and_rejects_stale_packets(self) -> None:
        self.assertTrue(is_newer_sequence(11, 10))
        self.assertFalse(is_newer_sequence(10, 10))
        self.assertFalse(is_newer_sequence(9, 10))
        self.assertTrue(is_newer_sequence(0, 0xFFFFFFFF))

    def test_values_outside_normalized_range_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AimCommand(0, True, 1.01, 0.0)
        with self.assertRaises(ValueError):
            AimCommand(0, True, 0.0, float("nan"))


if __name__ == "__main__":
    unittest.main()