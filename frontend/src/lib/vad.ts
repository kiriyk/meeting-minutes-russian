import { invoke } from '@tauri-apps/api/core';

export type VadEngine = 'silero_v6' | 'earshot';

export const DEFAULT_VAD_ENGINE: VadEngine = 'silero_v6';
const STORAGE_KEY = 'vadEngine';

export const VAD_ENGINE_OPTIONS: { value: VadEngine; label: string; description: string }[] = [
  { value: 'silero_v6', label: 'Silero VAD v6', description: 'Нейросетевой детектор речи (ONNX). Рекомендуется.' },
  { value: 'earshot', label: 'Earshot', description: 'Очень быстрый детектор на чистом Rust, без ONNX Runtime.' },
];

function isVadEngine(value: unknown): value is VadEngine {
  return value === 'silero_v6' || value === 'earshot';
}

export function readStoredVadEngine(): VadEngine {
  try {
    const stored = typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null;
    return isVadEngine(stored) ? stored : DEFAULT_VAD_ENGINE;
  } catch {
    return DEFAULT_VAD_ENGINE;
  }
}

/** Persists the choice and applies it in Rust; takes effect on the next recording or batch job. */
export async function applyVadEngine(engine: VadEngine): Promise<void> {
  try {
    window.localStorage.setItem(STORAGE_KEY, engine);
  } catch (err) {
    console.error('Failed to persist VAD engine:', err);
  }
  await invoke('set_vad_engine', { engine });
}
