export interface ModelWithStatus {
  status?: unknown;
}

interface ProviderCommands {
  initialize: string;
  hasAvailableModels: string;
  getAvailableModels: string;
}

const PROVIDER_COMMANDS: Record<string, ProviderCommands> = {
  localWhisper: {
    initialize: 'whisper_init',
    hasAvailableModels: 'whisper_has_available_models',
    getAvailableModels: 'whisper_get_available_models',
  },
  parakeet: {
    initialize: 'parakeet_init',
    hasAvailableModels: 'parakeet_has_available_models',
    getAvailableModels: 'parakeet_get_available_models',
  },
};

const RUSSIAN_ASR_COMMANDS: Record<'gigaam' | 'tone', ProviderCommands> = {
  gigaam: {
    initialize: 'gigaam_init',
    hasAvailableModels: 'gigaam_has_available_models',
    getAvailableModels: 'gigaam_get_available_models',
  },
  tone: {
    initialize: 'tone_init',
    hasAvailableModels: 'tone_has_available_models',
    getAvailableModels: 'tone_get_available_models',
  },
};

/** Mirrors `parse_russian_asr_model_id` in Rust: "gigaam:<name>", "tone:<name>" or a plain id. */
function russianAsrEngine(model: string): 'gigaam' | 'tone' | null {
  const raw = model.trim().toLowerCase();
  const separator = raw.indexOf(':');
  if (separator !== -1) {
    const engine = raw.slice(0, separator).trim();
    if (engine === 'gigaam' || engine === 'gigaam_engine') return 'gigaam';
    if (engine === 'tone' || engine === 'tone_engine') return 'tone';
    return null;
  }
  if (raw.startsWith('gigaam')) return 'gigaam';
  if (raw.includes('t-one') || raw.startsWith('tone')) return 'tone';
  return null;
}

export function getProviderCommands(provider: string, model?: string): ProviderCommands | null {
  if (provider === 'russianAsr') {
    const engine = model ? russianAsrEngine(model) : null;
    return engine ? RUSSIAN_ASR_COMMANDS[engine] : null;
  }
  return PROVIDER_COMMANDS[provider] ?? null;
}

export function hasDownloadingModel(models: ModelWithStatus[]): boolean {
  return models.some(({ status }) => (
    status === 'Downloading'
    || (status !== null && typeof status === 'object' && 'Downloading' in status)
  ));
}
