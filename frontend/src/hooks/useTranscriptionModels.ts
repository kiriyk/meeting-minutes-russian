import { useState, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";

export type TranscriptionProvider = "whisper" | "parakeet" | "gigaam" | "tone";
export const RETRANSCRIPTION_PROVIDERS: readonly TranscriptionProvider[] = [
  "whisper",
  "parakeet",
  "gigaam",
  "tone",
];
const IMPORT_PROVIDERS: readonly TranscriptionProvider[] = [
  "whisper",
  "parakeet",
];
const PROVIDERS = {
  whisper: { command: "whisper_get_available_models", label: "🏠 Whisper" },
  parakeet: { command: "parakeet_get_available_models", label: "⚡ Parakeet" },
  gigaam: { command: "gigaam_get_available_models", label: "GigaAM" },
  tone: { command: "tone_get_available_models", label: "T-one" },
};
export interface RawModelInfo {
  name: string;
  size_mb: number;
  status:
    | "Available"
    | "Missing"
    | { Downloading: number | { progress: number } }
    | { Error: string };
}
export interface ModelOption {
  provider: TranscriptionProvider;
  name: string;
  displayName: string;
  size_mb: number;
}
interface TranscriptModelConfig {
  provider?: string;
  model?: string;
}

/** Import keeps its original providers; retranscription opts into all engines. */
export function useTranscriptionModels(
  config: TranscriptModelConfig | undefined,
  providers: readonly TranscriptionProvider[] = IMPORT_PROVIDERS,
) {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [loadingModels, setLoadingModels] = useState(false);
  const userSelectedRef = useRef(false);
  const fetchGeneration = useRef(0);
  const providerKey = providers.join(",");
  const configuredProvider = config?.provider;
  const configuredModel = config?.model;
  const setSelectedModelKeyWithTracking = useCallback((key: string) => {
    userSelectedRef.current = true;
    setSelectedModelKey(key);
  }, []);
  const fetchModels = useCallback(async () => {
    const generation = ++fetchGeneration.current;
    setLoadingModels(true);
    const groups = await Promise.all(
      providerKey.split(",").map(async (key) => {
        const provider = key as TranscriptionProvider;
        try {
          if (provider === "gigaam" || provider === "tone")
            await invoke(`${provider}_init`);
          const models = await invoke<RawModelInfo[]>(
            PROVIDERS[provider].command,
          );
          return models
            .filter((m) => m.status === "Available")
            .map((m) => ({
              provider,
              name: m.name,
              size_mb: m.size_mb,
              displayName: `${PROVIDERS[provider].label}: ${m.name}`,
            }));
        } catch (error) {
          console.error(`Failed to fetch ${provider} models:`, error);
          return [];
        }
      }),
    );
    if (generation !== fetchGeneration.current) return;
    const allModels = groups.flat();
    setAvailableModels(allModels);
    let provider =
      configuredProvider === "localWhisper" ? "whisper" : configuredProvider;
    let name = configuredModel || "";
    if (provider === "russianAsr" || provider === "gigaam") {
      const colon = name.indexOf(":");
      if (colon >= 0) {
        provider = name.slice(0, colon);
        name = name.slice(colon + 1);
      } else if (provider === "russianAsr") {
        provider = name.startsWith("gigaam")
          ? "gigaam"
          : name === "t-one"
            ? "tone"
            : provider;
      }
    }
    if (provider === "gigaam_engine") provider = "gigaam";
    const match = allModels.find(
      (m) => m.provider === provider && m.name === name,
    );
    if (!userSelectedRef.current) {
      const selected = match ?? allModels[0];
      setSelectedModelKey(
        selected ? `${selected.provider}:${selected.name}` : "",
      );
    }
    setLoadingModels(false);
  }, [providerKey, configuredProvider, configuredModel]);
  const resetSelection = useCallback(() => {
    userSelectedRef.current = false;
  }, []);
  return {
    availableModels,
    selectedModelKey,
    setSelectedModelKey: setSelectedModelKeyWithTracking,
    loadingModels,
    fetchModels,
    resetSelection,
  };
}
