// GigaAM Russian ASR model integration
import { invoke } from '@tauri-apps/api/core';

export type GigaAmModelStatus = 'Available' | 'Missing';

export interface GigaAmModelInfo {
    name: string;
    path: string;
    size_mb: number;
    status: GigaAmModelStatus;
    description: string;
}

export class GigaAmAPI {
    static async init(): Promise<void> {
        await invoke('gigaam_init');
    }

    static async getAvailableModels(): Promise<GigaAmModelInfo[]> {
        return await invoke('gigaam_get_available_models');
    }

    static async loadModel(modelName: string): Promise<void> {
        await invoke('gigaam_load_model', { modelName });
    }

    static async getCurrentModel(): Promise<string | null> {
        return await invoke('gigaam_get_current_model');
    }

    static async isModelLoaded(): Promise<boolean> {
        return await invoke('gigaam_is_model_loaded');
    }

    static async transcribeAudio(audioData: number[]): Promise<string> {
        return await invoke('gigaam_transcribe_audio', { audioData });
    }

    static async getModelsDirectory(): Promise<string> {
        return await invoke('gigaam_get_models_directory');
    }

    static async hasAvailableModels(): Promise<boolean> {
        return await invoke('gigaam_has_available_models');
    }
}
