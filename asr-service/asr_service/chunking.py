"""Chunking placeholder for stage 1.

Stage 3 will add engine-specific segmentation logic and overlap handling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    live_frame_ms: int = 20
    t_one_emit_ms: int = 200
    gigaam_segment_target_s: int = 10
    gigaam_segment_max_s: int = 20
