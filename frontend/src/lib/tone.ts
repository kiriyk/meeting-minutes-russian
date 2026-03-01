import { invoke } from '@tauri-apps/api/core';

export type ToneModelStatus =
    | 'Available'
    | 'Missing'
    | { Downloading: number }
    | { Error: string };

export interface ToneModelInfo {
    name: string;
    path: string;
    size_mb: number;
    status: ToneModelStatus;
    description: string;
}

export class ToneAPI {
    static async init(): Promise<void> {
        await invoke('tone_init');
    }

    static async getAvailableModels(): Promise<ToneModelInfo[]> {
        return await invoke('tone_get_available_models');
    }

    static async loadModel(modelName: string): Promise<void> {
        await invoke('tone_load_model', { modelName });
    }

    static async getCurrentModel(): Promise<string | null> {
        return await invoke('tone_get_current_model');
    }

    static async isModelLoaded(): Promise<boolean> {
        return await invoke('tone_is_model_loaded');
    }

    static async transcribeAudio(audioData: number[]): Promise<string> {
        return await invoke('tone_transcribe_audio', { audioData });
    }

    static async getModelsDirectory(): Promise<string> {
        return await invoke('tone_get_models_directory');
    }

    static async hasAvailableModels(): Promise<boolean> {
        return await invoke('tone_has_available_models');
    }

    static async downloadModel(modelName: string): Promise<void> {
        await invoke('tone_download_model', { modelName });
    }

    static async cancelDownload(modelName: string): Promise<void> {
        await invoke('tone_cancel_download', { modelName });
    }

    static async deleteModel(modelName: string): Promise<string> {
        return await invoke('tone_delete_model', { modelName });
    }
}
