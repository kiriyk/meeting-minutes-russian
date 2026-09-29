import { useCallback } from "react";
import { invoke as invokeTauri } from "@tauri-apps/api/core";
import { toast } from "sonner";

export function useMeetingOperations({ meeting }: { meeting: { id: string } }) {
  // Open meeting folder in file explorer
  const handleOpenMeetingFolder = useCallback(async () => {
    try {
      await invokeTauri("open_meeting_folder", { meetingId: meeting.id });
    } catch (error) {
      console.error("Failed to open meeting folder:", error);
      toast.error((error as string) || "Failed to open recording folder");
    }
  }, [meeting.id]);

  return { handleOpenMeetingFolder };
}
