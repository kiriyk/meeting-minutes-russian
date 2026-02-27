"use client";

import { useEffect, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { useRecordingState } from "@/contexts/RecordingStateContext";
import { useTranscripts } from "@/contexts/TranscriptContext";

interface AsrTranscriptPayload {
  type: "partial_transcript" | "final_segment";
  engine?: string;
  text?: string;
  time_range_ms?: [number, number];
  confidence?: number | null;
}

interface AsrRuntimeStatusPayload {
  type: "status";
  session_id?: string;
  runtime?: {
    session_id: string;
    asr: Record<string, { enabled: boolean; backend: string; acceleration: string }>;
    vad: { mode: string; backend: string; acceleration: string };
    diarization: { mode: string; backend: string; acceleration: string };
  };
}

export function AsrLiveBridge() {
  const recordingState = useRecordingState();
  const { addTranscript } = useTranscripts();
  const seqRef = useRef(1_000_000);
  const activeSegmentSeqRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (!recordingState.isRecording) return;
    activeSegmentSeqRef.current.clear();

    const enabled =
      typeof window !== "undefined" &&
      localStorage.getItem("asrLiveCaptionsEnabled") === "true";
    const tOneEnabled =
      typeof window !== "undefined" &&
      localStorage.getItem("asrEngineTOne") !== "false";
    const gigaamEnabled =
      typeof window !== "undefined" &&
      localStorage.getItem("asrEngineGigaam") !== "false";
    const port = parseInt(localStorage.getItem("asrServicePort") || "8765", 10);

    const syncConfig = async () => {
      await invoke("set_asr_gateway_config", {
        enabled,
        port: Number.isFinite(port) ? port : 8765,
        tOneEnabled,
        gigaamEnabled,
      });
    };

    void syncConfig();
    if (!enabled) return;

    let unlistenTranscript: UnlistenFn | null = null;
    let unlistenRuntime: UnlistenFn | null = null;

    const toClockTime = () => {
      const now = new Date();
      return `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}`;
    };

    const setup = async () => {
      unlistenTranscript = await listen<AsrTranscriptPayload>(
        "asr-transcript-update",
        (event) => {
          const payload = event.payload;
          if (
            payload.type !== "partial_transcript" &&
            payload.type !== "final_segment"
          ) {
            return;
          }

          const startMs = payload.time_range_ms?.[0] ?? 0;
          const endMs = payload.time_range_ms?.[1] ?? 0;
          const text = payload.text || "";
          const engine = payload.engine || "t_one";
          const segmentKey = `${engine}:${startMs}`;

          // Keep stable sequence IDs per segment so partial updates replace the same row.
          let sequenceId = activeSegmentSeqRef.current.get(segmentKey);
          if (sequenceId === undefined) {
            sequenceId = seqRef.current++;
            activeSegmentSeqRef.current.set(segmentKey, sequenceId);
          }

          if (payload.type === "final_segment") {
            activeSegmentSeqRef.current.delete(segmentKey);
          }

          addTranscript({
            text: `[${engine}] ${text}`,
            timestamp: toClockTime(),
            source: "asr-gateway",
            sequence_id: sequenceId,
            chunk_start_time: startMs / 1000,
            is_partial: payload.type === "partial_transcript",
            confidence: payload.confidence ?? 0,
            audio_start_time: startMs / 1000,
            audio_end_time: endMs / 1000,
            duration: Math.max(0, (endMs - startMs) / 1000),
          });
        },
      );

      unlistenRuntime = await listen<AsrRuntimeStatusPayload>(
        "asr-runtime-status",
        (event) => {
          const runtime = event.payload?.runtime;
          if (!runtime) return;
          try {
            localStorage.setItem("asrRuntimeStatus", JSON.stringify(runtime));
            window.dispatchEvent(new Event("asr-runtime-status-updated"));
          } catch {
            // ignore storage issues
          }
        },
      );
    };

    void setup();

    return () => {
      if (unlistenTranscript) {
        unlistenTranscript();
      }
      if (unlistenRuntime) {
        unlistenRuntime();
      }
    };
  }, [recordingState.isRecording, addTranscript]);

  return null;
}
