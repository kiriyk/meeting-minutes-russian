"use client";

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Copy, FolderOpen, RefreshCw, Check } from 'lucide-react';
import Analytics from '@/lib/analytics';
import { OfflineAsrEngine } from '@/hooks/meeting-details/useMeetingOperations';
import { RetranscribeDialog } from './RetranscribeDialog';
import { useConfig } from '@/contexts/ConfigContext';


interface TranscriptButtonGroupProps {
  transcriptCount: number;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
  // Fork: offline Re-ASR (GigaAM / T-One via asr-service)
  selectedOfflineEngine?: OfflineAsrEngine;
  onOfflineEngineChange?: (engine: OfflineAsrEngine) => void;
  onOfflineRetranscribe?: () => Promise<void>;
  isOfflineRetranscribing?: boolean;
}


export function TranscriptButtonGroup({
  transcriptCount,
  onCopyTranscript,
  onOpenMeetingFolder,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
  selectedOfflineEngine = 'gigaam',
  onOfflineEngineChange,
  onOfflineRetranscribe,
  isOfflineRetranscribing = false,
}: TranscriptButtonGroupProps) {
  const { betaFeatures } = useConfig();
  const offlineEngineLabel = selectedOfflineEngine === 'gigaam' ? 'GigaAM' : 'T-One';
  const [showRetranscribeDialog, setShowRetranscribeDialog] = useState(false);

  const handleRetranscribeComplete = useCallback(async () => {
    // Refetch transcripts to show the updated data
    if (onRefetchTranscripts) {
      await onRefetchTranscripts();
    }
  }, [onRefetchTranscripts]);

  return (
    <div className="flex items-center justify-center w-full gap-2">
      <ButtonGroup>
        <Button
          variant="outline"
          size="sm"
          className="px-2 @[22rem]:px-3"
          onClick={() => {
            Analytics.trackButtonClick('copy_transcript', 'meeting_details');
            onCopyTranscript();
          }}
          disabled={transcriptCount === 0}
          title={transcriptCount === 0 ? 'No transcript available' : 'Copy Transcript'}
        >
          <Copy />
          <span className="hidden @[22rem]:inline">Copy</span>
        </Button>

        <Button
          size="sm"
          variant="outline"
          className="px-2 @[22rem]:px-4"
          onClick={() => {
            Analytics.trackButtonClick('open_recording_folder', 'meeting_details');
            onOpenMeetingFolder();
          }}
          title="Open Recording Folder"
        >
          <FolderOpen className="@[22rem]:mr-2" size={18} />
          <span className="hidden @[22rem]:inline">Recording</span>
        </Button>

        {onOfflineEngineChange && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                className="px-2 @[22rem]:px-3"
                title="Select re-transcription engine"
              >
                {offlineEngineLabel}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {(['gigaam', 't_one'] as const).map((engine) => (
                <DropdownMenuItem
                  key={engine}
                  onClick={() => onOfflineEngineChange(engine)}
                  className="flex items-center justify-between gap-2"
                >
                  <span>{engine === 'gigaam' ? 'GigaAM' : 'T-One'}</span>
                  {selectedOfflineEngine === engine && (
                    <Check className="h-4 w-4 text-green-600" />
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        {onOfflineRetranscribe && (
          <Button
            size="sm"
            variant="outline"
            className="px-2 @[22rem]:px-4"
            onClick={() => {
              Analytics.trackButtonClick('retranscribe_selected_engine', 'meeting_details');
              void onOfflineRetranscribe();
            }}
            disabled={isOfflineRetranscribing}
            title={`Re-transcribe saved meeting with ${offlineEngineLabel}`}
          >
            <RefreshCw className={`@[22rem]:mr-2 ${isOfflineRetranscribing ? 'animate-spin' : ''}`} size={18} />
            <span className="hidden @[22rem]:inline">
              {isOfflineRetranscribing ? 'Processing...' : 'Re-ASR'}
            </span>
          </Button>
        )}

        {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
          <Button
            size="sm"
            variant="outline"
            className="bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border-blue-200 px-2 @[22rem]:px-4"
            onClick={() => {
              Analytics.trackButtonClick('enhance_transcript', 'meeting_details');
              setShowRetranscribeDialog(true);
            }}
            title="Retranscribe to enhance your recorded audio"
          >
            <RefreshCw className="@[22rem]:mr-2" size={18} />
            <span className="hidden @[22rem]:inline">Enhance</span>
          </Button>
        )}
      </ButtonGroup>

      {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
        <RetranscribeDialog
          open={showRetranscribeDialog}
          onOpenChange={setShowRetranscribeDialog}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onComplete={handleRetranscribeComplete}
        />
      )}
    </div>
  );
}
