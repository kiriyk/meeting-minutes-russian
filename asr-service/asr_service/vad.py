"""Lightweight energy-based VAD for Stage 2.

This is a practical fallback until Silero VAD is integrated.
"""

from __future__ import annotations

import audioop


class VAD:
    def __init__(self, mode: str = "silero", aggressiveness: int = 2) -> None:
        self.mode = mode
        self.aggressiveness = aggressiveness

        # Lower threshold => more sensitive. Tuned for 16-bit PCM.
        thresholds = {
            0: 200,
            1: 300,
            2: 450,
            3: 650,
        }
        self.threshold = thresholds.get(aggressiveness, 450)

    def is_speech(self, pcm_frame: bytes) -> bool:
        if not pcm_frame:
            return False
        rms = audioop.rms(pcm_frame, 2)
        return rms >= self.threshold
