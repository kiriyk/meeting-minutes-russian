import { useCallback, useState } from "react";
import { invoke as invokeTauri } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { Transcript } from "@/types";

export type OfflineAsrEngine = "gigaam" | "t_one";

interface UseMeetingOperationsProps {
  meeting: any;
  onMeetingUpdated?: () => Promise<void>;
}

export function useMeetingOperations({
  meeting,
  onMeetingUpdated,
}: UseMeetingOperationsProps) {
  const [isRetranscribing, setIsRetranscribing] = useState(false);
  const [selectedOfflineEngine, setSelectedOfflineEngine] =
    useState<OfflineAsrEngine>("gigaam");

  // Open meeting folder in file explorer
  const handleOpenMeetingFolder = useCallback(async () => {
    try {
      await invokeTauri("open_meeting_folder", { meetingId: meeting.id });
    } catch (error) {
      console.error("Failed to open meeting folder:", error);
      toast.error((error as string) || "Failed to open recording folder");
    }
  }, [meeting.id]);

  const handleRetranscribeWithGigaam = useCallback(async () => {
    if (isRetranscribing) {
      return;
    }

    try {
      setIsRetranscribing(true);
      toast.info("Starting GigaAM re-transcription...");
      const filePath = await invokeTauri<string>(
        "api_find_meeting_audio_file",
        {
          meetingId: meeting.id,
        },
      );

      const port = parseInt(
        localStorage.getItem("asrServicePort") || "8765",
        10,
      );
      const baseUrl = `http://127.0.0.1:${Number.isFinite(port) ? port : 8765}`;

      const startResp = await fetch(`${baseUrl}/offline_transcribe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_path: filePath,
          engine: selectedOfflineEngine,
          segment_seconds: 20,
        }),
      });

      if (!startResp.ok) {
        const errText = await startResp.text();
        throw new Error(
          `offline_transcribe failed: ${startResp.status} ${errText}`,
        );
      }

      const { job_id } = (await startResp.json()) as { job_id: string };
      if (!job_id) {
        throw new Error("Offline job id is missing");
      }

      const startedAt = Date.now();
      const timeoutMs = 20 * 60 * 1000;

      while (true) {
        if (Date.now() - startedAt > timeoutMs) {
          throw new Error("Offline transcription timed out");
        }

        const statusResp = await fetch(`${baseUrl}/jobs/${job_id}`);
        if (!statusResp.ok) {
          const errText = await statusResp.text();
          throw new Error(
            `Job polling failed: ${statusResp.status} ${errText}`,
          );
        }

        const job = (await statusResp.json()) as any;
        if (job.status === "failed") {
          throw new Error(job.error || "Offline transcription failed");
        }
        if (job.status === "completed") {
          const segments = (job.result?.segments ?? []) as any[];
          if (segments.length === 0) {
            throw new Error(
              "Offline transcription completed but returned no segments",
            );
          }
          const transcripts: Transcript[] = segments.map(
            (seg: any, index: number) => {
              const startMs = Array.isArray(seg.time_range_ms)
                ? Number(seg.time_range_ms[0] ?? 0)
                : 0;
              const endMs = Array.isArray(seg.time_range_ms)
                ? Number(seg.time_range_ms[1] ?? startMs)
                : startMs;
              const startSec = startMs / 1000;
              const endSec = endMs / 1000;
              const mins = Math.floor(startSec / 60);
              const secs = Math.floor(startSec % 60);
              return {
                id: `${meeting.id}-gigaam-${index + 1}`,
                text: String(seg.text ?? "").trim(),
                timestamp: `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`,
                audio_start_time: startSec,
                audio_end_time: endSec,
                duration: Math.max(0, endSec - startSec),
              };
            },
          );

          await invokeTauri("api_replace_meeting_transcripts", {
            meetingId: meeting.id,
            transcripts,
          });

          if (onMeetingUpdated) {
            await onMeetingUpdated();
          }

          toast.success(
            `Meeting re-transcribed via ${selectedOfflineEngine} (${transcripts.length} segments)`,
          );
          return;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    } catch (error) {
      console.error("Failed to re-transcribe meeting with GigaAM:", error);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to re-transcribe meeting",
      );
    } finally {
      setIsRetranscribing(false);
    }
  }, [isRetranscribing, meeting.id, onMeetingUpdated, selectedOfflineEngine]);

  return {
    handleOpenMeetingFolder,
    handleRetranscribeWithGigaam,
    isRetranscribing,
    selectedOfflineEngine,
    setSelectedOfflineEngine,
  };
}
