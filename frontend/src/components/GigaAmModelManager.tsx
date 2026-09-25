import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { GigaAmAPI, GigaAmModelInfo } from '../lib/gigaam';

interface GigaAmModelManagerProps {
    selectedModel?: string;
    onModelSelect?: (modelName: string) => void;
    autoSave?: boolean;
}

export function GigaAmModelManager({
    selectedModel,
    onModelSelect,
    autoSave = false,
}: GigaAmModelManagerProps) {
    const [models, setModels] = useState<GigaAmModelInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingModel, setLoadingModel] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const init = async () => {
            try {
                setLoading(true);
                await GigaAmAPI.init();
                const list = await GigaAmAPI.getAvailableModels();
                setModels(list);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to load GigaAM models');
            } finally {
                setLoading(false);
            }
        };
        init();
    }, []);

    // Listen for loading events
    useEffect(() => {
        let unlistenStarted: (() => void) | null = null;
        let unlistenCompleted: (() => void) | null = null;
        let unlistenFailed: (() => void) | null = null;

        const setup = async () => {
            unlistenStarted = await listen<{ modelName: string }>('gigaam-model-loading-started', () => {
                setLoadingModel(true);
            });
            unlistenCompleted = await listen<{ modelName: string }>('gigaam-model-loading-completed', (e) => {
                setLoadingModel(false);
                toast.success('GigaAM готов к работе', { duration: 3000 });
                if (onModelSelect) onModelSelect(e.payload.modelName);
            });
            unlistenFailed = await listen<{ modelName: string; error: string }>('gigaam-model-loading-failed', (e) => {
                setLoadingModel(false);
                toast.error('Не удалось загрузить GigaAM', { description: e.payload.error, duration: 5000 });
            });
        };
        setup();
        return () => {
            unlistenStarted?.();
            unlistenCompleted?.();
            unlistenFailed?.();
        };
    }, [onModelSelect]);

    const saveAndSelect = async (modelName: string) => {
        if (autoSave) {
            try {
                await invoke('api_save_transcript_config', { provider: 'gigaam', model: modelName, apiKey: null });
            } catch (e) {
                console.error('Failed to save GigaAM config:', e);
            }
        }
        if (onModelSelect) onModelSelect(modelName);
    };

    const handleSelect = async (model: GigaAmModelInfo) => {
        if (model.status !== 'Available') return;
        try {
            setLoadingModel(true);
            await GigaAmAPI.loadModel(model.name);
            await saveAndSelect(model.name);
        } catch (e) {
            setLoadingModel(false);
            toast.error('Не удалось загрузить модель', {
                description: e instanceof Error ? e.message : String(e),
                duration: 5000,
            });
        }
    };

    if (loading) {
        return (
            <div className="animate-pulse space-y-3">
                <div className="h-20 bg-gray-100 rounded-lg" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800">Ошибка загрузки GigaAM: {error}</p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {models.map((model) => {
                const isAvailable = model.status === 'Available';
                const isSelected = selectedModel === model.name;

                return (
                    <motion.div
                        key={model.name}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        onClick={() => isAvailable && !loadingModel && handleSelect(model)}
                        className={`
              relative rounded-lg border-2 p-4 transition-all
              ${isSelected && isAvailable ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white'}
              ${isAvailable && !loadingModel ? 'cursor-pointer hover:border-gray-300' : 'cursor-default opacity-60'}
            `}
                    >
                        <div className="flex items-start justify-between">
                            <div className="flex items-center gap-3">
                                <span className="text-2xl">🇷🇺</span>
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h3 className="font-semibold text-gray-900">GigaAM v3</h3>
                                        {isSelected && isAvailable && (
                                            <span className="bg-blue-600 text-white px-2 py-0.5 rounded-full text-xs font-medium">✓</span>
                                        )}
                                    </div>
                                    <p className="text-sm text-gray-600">{model.description}</p>
                                    <p className="text-xs text-gray-400 mt-0.5">{model.size_mb} MB • Русский язык • RNN-T E2E</p>
                                </div>
                            </div>

                            <div className="ml-4 flex items-center">
                                {loadingModel && isSelected ? (
                                    <span className="text-xs text-blue-600 font-medium animate-pulse">Загрузка…</span>
                                ) : isAvailable ? (
                                    <div className="flex items-center gap-1.5 text-green-600">
                                        <div className="w-2 h-2 bg-green-500 rounded-full" />
                                        <span className="text-xs font-medium">Ready</span>
                                    </div>
                                ) : (
                                    <span className="text-xs text-gray-400">Файлы не найдены</span>
                                )}
                            </div>
                        </div>

                        {model.status === 'Missing' && (
                            <p className="mt-2 text-xs text-amber-600 bg-amber-50 rounded p-2">
                                Файлы модели не найдены. Скопируйте{' '}
                                <code className="font-mono">encoder.onnx</code>,{' '}
                                <code className="font-mono">decoder.onnx</code>,{' '}
                                <code className="font-mono">joint.onnx</code>,{' '}
                                <code className="font-mono">vocab.json</code> в папку моделей.
                            </p>
                        )}
                    </motion.div>
                );
            })}

            {selectedModel && models.find(m => m.name === selectedModel)?.status === 'Available' && (
                <p className="text-xs text-gray-500 text-center pt-1">
                    Используется GigaAM v3 для транскрипции
                </p>
            )}
        </div>
    );
}
