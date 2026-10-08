import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  RefreshCw,
  Globe,
  Loader2,
  AlertCircle,
  X,
  Cpu,
  Download,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import { Button } from "../ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { toast } from "sonner";
import { useConfig } from "@/contexts/ConfigContext";
import { LANGUAGES } from "@/constants/languages";
import {
  useTranscriptionModels,
  RETRANSCRIPTION_PROVIDERS,
} from "@/hooks/useTranscriptionModels";
import Analytics from "@/lib/analytics";

interface RetranscribeDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  meetingId: string;
  meetingFolderPath: string | null;
  onComplete?: () => void;
}
interface RetranscriptionProgress {
  meeting_id: string;
  stage: string;
  progress_percentage: number;
  message: string;
}
interface RetranscriptionResult {
  meeting_id: string;
  segments_count: number;
  duration_seconds: number;
  warnings?: string[];
}
interface ModelStatus {
  available: boolean;
  downloading: boolean;
  progress: number;
  active_download?: DiarizationModel | null;
}
type DiarizationModel = "community-1" | "legacy";
interface SpeakerDownloadEvent {
  model: DiarizationModel;
  progress?: number;
  error?: string;
}
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export function RetranscribeDialog({
  open,
  onOpenChange,
  meetingId,
  meetingFolderPath,
  onComplete,
}: RetranscribeDialogProps) {
  const { selectedLanguage, transcriptModelConfig } = useConfig();
  const [isProcessing, setIsProcessing] = useState(false);
  const [cancellationRequested, setCancellationRequested] = useState(false);
  const [listenersReady, setListenersReady] = useState(false);
  const [progress, setProgress] = useState<RetranscriptionProgress | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [selectedLang, setSelectedLang] = useState(selectedLanguage || "auto");
  const [diarizationEnabled, setDiarizationEnabled] = useState(true);
  const [diarizationModel, setDiarizationModel] =
    useState<DiarizationModel>("community-1");
  const diarizationModelRef = useRef(diarizationModel);
  diarizationModelRef.current = diarizationModel;
  const [speakerModels, setSpeakerModels] = useState<ModelStatus | null>(null);
  const [activeDownload, setActiveDownload] = useState<DiarizationModel | null>(
    null,
  );
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const {
    availableModels,
    selectedModelKey,
    setSelectedModelKey,
    loadingModels,
    fetchModels,
    resetSelection,
  } = useTranscriptionModels(transcriptModelConfig, RETRANSCRIPTION_PROVIDERS);
  const onCompleteRef = useRef(onComplete);
  const onOpenChangeRef = useRef(onOpenChange);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);
  useEffect(() => {
    onOpenChangeRef.current = onOpenChange;
  }, [onOpenChange]);
  const previousOpen = useRef(false);
  const selectedModel = useMemo(
    () =>
      availableModels.find(
        (m) => `${m.provider}:${m.name}` === selectedModelKey,
      ),
    [availableModels, selectedModelKey],
  );
  const isRussianModel =
    selectedModel?.provider === "gigaam" || selectedModel?.provider === "tone";
  const isParakeetModel = selectedModel?.provider === "parakeet";
  const canStart =
    !!meetingFolderPath &&
    !!selectedModel &&
    !loadingModels &&
    listenersReady &&
    !isProcessing;

  useEffect(() => {
    const wasOpen = previousOpen.current;
    previousOpen.current = open;
    if (open && !wasOpen) {
      resetSelection();
      setError(null);
      setProgress(null);
      setCancellationRequested(false);
      setSelectedLang(selectedLanguage || "auto");
      void fetchModels();
    }
  }, [open, fetchModels, resetSelection, selectedLanguage]);

  useEffect(() => {
    if (!open) return;
    let disposed = false;
    let downloadEventVersion = 0;
    let activeDownloadEventVersion = 0;
    const unlisteners: UnlistenFn[] = [];
    setListenersReady(false);
    const register = async <T,>(
      name: string,
      handler: (payload: T) => void,
    ) => {
      const unlisten = await listen<T>(name, (event) => {
        if (!disposed) handler(event.payload);
      });
      if (disposed) unlisten();
      else unlisteners.push(unlisten);
    };
    const setup = async () => {
      await register<RetranscriptionProgress>(
        "retranscription-progress",
        (payload) => {
          if (payload.meeting_id === meetingId) setProgress(payload);
        },
      );
      await register<RetranscriptionResult>(
        "retranscription-complete",
        (payload) => {
          if (payload.meeting_id !== meetingId) return;
          setIsProcessing(false);
          setCancellationRequested(false);
          toast.success(
            `Retranscription complete! ${payload.segments_count} segments created.`,
          );
          payload.warnings?.forEach((warning) => toast.warning(warning));
          onCompleteRef.current?.();
          onOpenChangeRef.current(false);
          void Analytics.track("enhance_transcript_completed", {
            success: "true",
            duration_seconds: String(payload.duration_seconds),
            segments_count: String(payload.segments_count),
          }).catch(console.error);
        },
      );
      await register<{ meeting_id: string; error: string }>(
        "retranscription-error",
        (payload) => {
          if (payload.meeting_id !== meetingId) return;
          setIsProcessing(false);
          setCancellationRequested(false);
          if (payload.error === "Retranscription cancelled") {
            toast.info(
              "Retranscription cancelled. Existing transcript preserved.",
            );
            onOpenChangeRef.current(false);
          } else {
            setError(payload.error);
            void Analytics.trackError(
              "enhance_transcript_failed",
              payload.error,
            ).catch(console.error);
          }
        },
      );
      await register<SpeakerDownloadEvent>(
        "diarization-model-download-progress",
        (payload) => {
          activeDownloadEventVersion++;
          setActiveDownload(payload.model);
          if (payload.model !== diarizationModel) {
            setSpeakerModels((status) =>
              status ? { ...status, active_download: payload.model } : status,
            );
            return;
          }
          downloadEventVersion++;
          setSpeakerModels({
            available: false,
            downloading: true,
            progress: payload.progress ?? 0,
            active_download: payload.model,
          });
        },
      );
      await register<SpeakerDownloadEvent>(
        "diarization-model-download-complete",
        (payload) => {
          activeDownloadEventVersion++;
          setActiveDownload((active) =>
            active === payload.model ? null : active,
          );
          if (payload.model !== diarizationModel) {
            setSpeakerModels((status) =>
              status ? { ...status, active_download: null } : status,
            );
            return;
          }
          downloadEventVersion++;
          setSpeakerModels({
            available: true,
            downloading: false,
            progress: 100,
          });
          setDownloadError(null);
        },
      );
      await register<SpeakerDownloadEvent>(
        "diarization-model-download-error",
        (payload) => {
          activeDownloadEventVersion++;
          setActiveDownload((active) =>
            active === payload.model ? null : active,
          );
          if (payload.model !== diarizationModel) {
            setSpeakerModels((status) =>
              status ? { ...status, active_download: null } : status,
            );
            return;
          }
          downloadEventVersion++;
          setSpeakerModels({
            available: false,
            downloading: false,
            progress: 0,
          });
          setDownloadError(payload.error ?? "Speaker model download failed");
        },
      );
      if (disposed) return;
      setListenersReady(true);
      // Query after listeners are registered so completion during opening cannot be lost.
      try {
        const queryVersion = downloadEventVersion;
        const activeQueryVersion = activeDownloadEventVersion;
        const status = await invoke<ModelStatus>(
          "diarization_get_model_status",
          { model: diarizationModel },
        );
        if (!disposed && queryVersion === downloadEventVersion)
          setSpeakerModels(status);
        if (!disposed && activeQueryVersion === activeDownloadEventVersion)
          setActiveDownload(status.active_download ?? null);
      } catch (err) {
        if (!disposed) setDownloadError(errorMessage(err));
      }
    };
    void setup().catch((err) => {
      if (!disposed)
        setError(`Cannot subscribe to progress: ${errorMessage(err)}`);
    });
    return () => {
      disposed = true;
      unlisteners.forEach((unlisten) => unlisten());
    };
  }, [open, meetingId, diarizationModel]);

  const handleStart = async () => {
    if (!canStart || !selectedModel) return;
    setIsProcessing(true);
    setCancellationRequested(false);
    setError(null);
    setProgress(null);
    const language = isRussianModel
      ? "ru"
      : isParakeetModel || selectedLang === "auto"
        ? null
        : selectedLang;
    void Analytics.track("enhance_transcript_started", {
      language: language || "auto",
      model_provider: selectedModel.provider,
      model_name: selectedModel.name,
    }).catch(console.error);
    try {
      await invoke("start_retranscription_command", {
        meetingId,
        meetingFolderPath,
        language,
        model: selectedModel.name,
        provider: selectedModel.provider,
        diarizationEnabled,
        diarizationModel,
      });
    } catch (err) {
      setIsProcessing(false);
      setError(errorMessage(err));
    }
  };
  const handleCancel = async () => {
    if (!isProcessing) {
      onOpenChange(false);
      return;
    }
    setCancellationRequested(true);
    try {
      await invoke("cancel_retranscription_command");
    } catch (err) {
      setCancellationRequested(false);
      toast.error(errorMessage(err));
    }
  };
  const handleDownload = async () => {
    setDownloadError(null);
    setSpeakerModels({ available: false, downloading: true, progress: 0 });
    setActiveDownload(diarizationModel);
    try {
      await invoke("diarization_download_models", { model: diarizationModel });
      if (diarizationModelRef.current !== diarizationModel) return;
      setActiveDownload((active) =>
        active === diarizationModel ? null : active,
      );
      setSpeakerModels({ available: true, downloading: false, progress: 100 });
    } catch (err) {
      if (diarizationModelRef.current !== diarizationModel) return;
      setActiveDownload((active) =>
        active === diarizationModel ? null : active,
      );
      setSpeakerModels({ available: false, downloading: false, progress: 0 });
      setDownloadError(errorMessage(err));
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!isProcessing) onOpenChange(next);
      }}
    >
      <DialogContent
        className="sm:max-w-[450px]"
        onEscapeKeyDown={(event) => {
          if (isProcessing) event.preventDefault();
        }}
        onInteractOutside={(event) => {
          if (isProcessing) event.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {isProcessing ? (
              <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
            ) : error ? (
              <AlertCircle className="h-5 w-5 text-red-600" />
            ) : (
              <RefreshCw className="h-5 w-5 text-blue-600" />
            )}
            {isProcessing
              ? "Retranscribing..."
              : error
                ? "Retranscription Failed"
                : "Retranscribe Meeting"}
          </DialogTitle>
          <DialogDescription>
            {isProcessing
              ? cancellationRequested
                ? "Cancellation requested. Waiting for the current processing step to finish..."
                : progress?.message || "Processing audio..."
              : "Re-process saved audio with a different model and identify speakers."}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          {!isProcessing && !error && (
            <>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Model</span>
                </div>
                <Select
                  value={selectedModelKey}
                  onValueChange={setSelectedModelKey}
                  disabled={loadingModels}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={
                        loadingModels ? "Loading models..." : "Select model"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem
                        key={`${model.provider}:${model.name}`}
                        value={`${model.provider}:${model.name}`}
                      >
                        {model.displayName} ({Math.round(model.size_mb)} MB)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {!loadingModels && availableModels.length === 0 && (
                  <p className="text-sm text-muted-foreground">
                    Download a Whisper, Parakeet, GigaAM or T-one model in
                    Settings before starting.
                  </p>
                )}
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Globe className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Language</span>
                </div>
                {isRussianModel ? (
                  <p className="text-xs text-muted-foreground">
                    GigaAM and T-one recognize Russian.
                  </p>
                ) : isParakeetModel ? (
                  <p className="text-xs text-muted-foreground">
                    Parakeet detects the language automatically.
                  </p>
                ) : (
                  <Select value={selectedLang} onValueChange={setSelectedLang}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent className="max-h-60">
                      {LANGUAGES.map((lang) => (
                        <SelectItem key={lang.code} value={lang.code}>
                          {lang.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <div className="space-y-2 rounded-lg border p-3">
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={diarizationEnabled}
                    onChange={(event) =>
                      setDiarizationEnabled(event.target.checked)
                    }
                  />
                  Identify speakers
                </label>
                {diarizationEnabled && (
                  <>
                    <label
                      htmlFor="diarization-model"
                      className="text-xs font-medium"
                    >
                      Speaker model
                    </label>
                    <Select
                      value={diarizationModel}
                      onValueChange={(value) => {
                        setSpeakerModels(null);
                        setDownloadError(null);
                        setDiarizationModel(value as DiarizationModel);
                      }}
                    >
                      <SelectTrigger id="diarization-model">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="community-1">
                          pyannote Community-1 (recommended)
                        </SelectItem>
                        <SelectItem value="legacy">
                          Pyannote + TitaNet (legacy)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      {diarizationModel === "community-1"
                        ? "pyannote speaker-diarization-community-1 via speakrs. About 60 MB. Runs locally on CPU and detects speech and speakers together."
                        : "About 44 MB. Runs locally on CPU with separate speech detection."}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {speakerModels?.available
                        ? "Speaker models are ready. Speakers will be labeled across the entire recording."
                        : "Without speaker models, text and timestamps will still be transcribed, with a warning."}
                    </p>
                    {!speakerModels?.available &&
                      (speakerModels?.downloading ? (
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span>
                            Downloading speaker models: {speakerModels.progress}
                            %
                          </span>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              void invoke("diarization_cancel_download", {
                                model: diarizationModel,
                              }).catch((err) =>
                                setDownloadError(errorMessage(err)),
                              );
                            }}
                          >
                            Cancel download
                          </Button>
                        </div>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={
                            !listenersReady ||
                            speakerModels === null ||
                            !!activeDownload
                          }
                          onClick={handleDownload}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          Download speaker models
                        </Button>
                      ))}
                    {downloadError && (
                      <p className="text-xs text-red-700">{downloadError}</p>
                    )}
                  </>
                )}
              </div>
            </>
          )}
          {isProcessing && progress && (
            <div className="space-y-2">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all"
                  style={{
                    width: `${Math.min(progress.progress_percentage, 100)}%`,
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-600">
                <span>{progress.stage}</span>
                <span>{progress.progress_percentage}%</span>
              </div>
            </div>
          )}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}
        </div>
        <DialogFooter>
          {isProcessing ? (
            <Button
              variant="outline"
              onClick={handleCancel}
              disabled={
                cancellationRequested ||
                progress?.stage === "saving" ||
                progress?.stage === "complete"
              }
            >
              <X className="h-4 w-4 mr-2" />
              {cancellationRequested ? "Cancellation requested" : "Cancel"}
            </Button>
          ) : error ? (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Close
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setError(null);
                  setProgress(null);
                  void fetchModels();
                }}
              >
                Try Again
              </Button>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
              <Button
                onClick={handleStart}
                className="bg-blue-600 hover:bg-blue-700"
                disabled={!canStart}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Start Retranscription
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
