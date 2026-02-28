import { useCallback, useState } from "react";
import { invoke as invokeTauri } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { Transcript } from "@/types";

export type OfflineAsrEngine = "gigaam" | "t_one";

interface UseMeetingOperationsProps {
  meeting: any;
  onMeetingUpdated?: () => Promise<void>;
}

interface OfflineJobStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress?: number;
  queue_position?: number | null;
  error?: string | null;
  result?: {
    segments?: any[];
  } | null;
}

export function useMeetingOperations({
  meeting,
  onMeetingUpdated,
}: UseMeetingOperationsProps) {
  const [isRetranscribing, setIsRetranscribing] = useState(false);
  const [activeOfflineJobId, setActiveOfflineJobId] = useState<string | null>(
    null,
  );
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
      const port = parseInt(
        localStorage.getItem("asrServicePort") || "8765",
        10,
      );
      const baseUrl = `http://127.0.0.1:${Number.isFinite(port) ? port : 8765}`;
      let jobId = activeOfflineJobId;
      if (jobId) {
        toast.info(`Resuming offline job: ${jobId.slice(0, 8)}...`);
      } else {
        toast.info(
          `Starting ${selectedOfflineEngine === "gigaam" ? "GigaAM" : "T-One"} re-transcription...`,
        );
        const filePath = await invokeTauri<string>(
          "api_find_meeting_audio_file",
          {
            meetingId: meeting.id,
          },
        );

        const startResp = await fetch(`${baseUrl}/offline_transcribe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            file_path: filePath,
            engine: selectedOfflineEngine,
            // Phrase-based segmentation is used server-side; this is only a hard cap.
            segment_seconds: 90,
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
        jobId = job_id;
        setActiveOfflineJobId(jobId);
      }

      const startedAt = Date.now();
      let shownQueuePosition: number | null = null;
      let progressMark = -1;
      let warnedLongRunning = false;

      while (true) {
        if (!warnedLongRunning && Date.now() - startedAt > 30 * 60 * 1000) {
          warnedLongRunning = true;
          toast.info("Offline transcription is still running in background...");
        }

        const statusResp = await fetch(`${baseUrl}/jobs/${jobId}`);
        if (!statusResp.ok) {
          const errText = await statusResp.text();
          throw new Error(
            `Job polling failed: ${statusResp.status} ${errText}`,
          );
        }

        const job = (await statusResp.json()) as OfflineJobStatus;
        if (
          job.status === "queued" &&
          typeof job.queue_position === "number" &&
          shownQueuePosition !== job.queue_position
        ) {
          shownQueuePosition = job.queue_position;
          toast.info(`Re-ASR queued. Position: ${job.queue_position}`);
        }
        if (typeof job.progress === "number") {
          const rounded = Math.floor(job.progress);
          const mark = Math.floor(rounded / 10);
          if (mark > progressMark && mark >= 1 && rounded < 100) {
            progressMark = mark;
            toast.info(`Re-ASR progress: ${rounded}%`);
          }
        }

        if (job.status === "failed") {
          setActiveOfflineJobId(null);
          throw new Error(job.error || "Offline transcription failed");
        }
        if (job.status === "completed") {
          setActiveOfflineJobId(null);
          const segments = (job.result?.segments ?? []) as any[];
          if (segments.length === 0) {
            toast.info(
              "Offline transcription completed: no speech segments detected. Existing transcript left unchanged.",
            );
            return;
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
                id: `${meeting.id}-${selectedOfflineEngine}-${index + 1}`,
                text: String(seg.text ?? "").trim(),
                timestamp: `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`,
                speaker:
                  typeof seg.speaker === "string" && seg.speaker.trim().length > 0
                    ? seg.speaker.trim()
                    : undefined,
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
      console.error("Failed to re-transcribe meeting with selected engine:", error);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to re-transcribe meeting",
      );
    } finally {
      setIsRetranscribing(false);
    }
  }, [
    activeOfflineJobId,
    isRetranscribing,
    meeting.id,
    onMeetingUpdated,
    selectedOfflineEngine,
  ]);

  return {
    handleOpenMeetingFolder,
    handleRetranscribeWithGigaam,
    isRetranscribing,
    selectedOfflineEngine,
    setSelectedOfflineEngine,
  };
}
