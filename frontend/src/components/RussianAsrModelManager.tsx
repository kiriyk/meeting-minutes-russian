import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { GigaAmAPI, GigaAmModelInfo } from '../lib/gigaam';
import { ToneAPI, ToneModelInfo } from '../lib/tone';

interface RussianAsrModelManagerProps {
    /** currently selected model in format "gigaam:modelName" or "tone:modelName" */
    selectedModel?: string;
    onModelSelect?: (fullModelId: string) => void;
    autoSave?: boolean;
}

type ActiveEngine = 'gigaam' | 'tone';

function parseModelId(fullId?: string): { engine: ActiveEngine; name: string } | null {
    if (!fullId) return null;
    const colon = fullId.indexOf(':');
    if (colon < 0) return null;
    const engine = fullId.slice(0, colon) as ActiveEngine;
    const name = fullId.slice(colon + 1);
    return { engine, name };
}

export function RussianAsrModelManager({
    selectedModel,
    onModelSelect,
    autoSave = false,
}: RussianAsrModelManagerProps) {
    const [gigaAmModels, setGigaAmModels] = useState<GigaAmModelInfo[]>([]);
    const [toneModels, setToneModels] = useState<ToneModelInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingId, setLoadingId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const selected = parseModelId(selectedModel);

    // Init and discover both engines
    useEffect(() => {
        const init = async () => {
            try {
                setLoading(true);
                await Promise.all([GigaAmAPI.init(), ToneAPI.init()]);
                const [ga, to] = await Promise.all([
                    GigaAmAPI.getAvailableModels(),
                    ToneAPI.getAvailableModels(),
                ]);
                setGigaAmModels(ga);
                setToneModels(to);
            } catch (e) {
                setError(e instanceof Error ? e.message : 'Failed to load Russian ASR models');
            } finally {
                setLoading(false);
            }
        };
        init();
    }, []);

    // Listen for GigaAM loading events
    useEffect(() => {
        let u1: (() => void) | null = null, u2: (() => void) | null = null, u3: (() => void) | null = null;
        const setup = async () => {
            u1 = await listen('gigaam-model-loading-started', () => setLoadingId('gigaam'));
            u2 = await listen<{ modelName: string }>('gigaam-model-loading-completed', (e) => {
                setLoadingId(null);
                toast.success('GigaAM готов', { duration: 3000 });
                saveAndSelect('gigaam', e.payload.modelName);
            });
            u3 = await listen<{ error: string }>('gigaam-model-loading-failed', (e) => {
                setLoadingId(null);
                toast.error('GigaAM: ошибка загрузки', { description: e.payload.error, duration: 5000 });
            });
        };
        setup();
        return () => { u1?.(); u2?.(); u3?.(); };
    }, []);

    // Listen for T-One loading events
    useEffect(() => {
        let u1: (() => void) | null = null, u2: (() => void) | null = null, u3: (() => void) | null = null;
        const setup = async () => {
            u1 = await listen('tone-model-loading-started', () => setLoadingId('tone'));
            u2 = await listen<{ modelName: string }>('tone-model-loading-completed', (e) => {
                setLoadingId(null);
                toast.success('T-One готов', { duration: 3000 });
                saveAndSelect('tone', e.payload.modelName);
            });
            u3 = await listen<{ error: string }>('tone-model-loading-failed', (e) => {
                setLoadingId(null);
                toast.error('T-One: ошибка загрузки', { description: e.payload.error, duration: 5000 });
            });
        };
        setup();
        return () => { u1?.(); u2?.(); u3?.(); };
    }, []);

    const saveAndSelect = async (engine: ActiveEngine, modelName: string) => {
        const fullId = `${engine}:${modelName}`;
        if (autoSave) {
            try {
                await invoke('api_save_transcript_config', {
                    provider: 'russianAsr',
                    model: fullId,
                    apiKey: null,
                });
            } catch (e) {
                console.error('Failed to save russian ASR config:', e);
            }
        }
        onModelSelect?.(fullId);
    };

    const handleSelectGigaAm = async (model: GigaAmModelInfo) => {
        if (model.status !== 'Available' || loadingId) return;
        try {
            setLoadingId('gigaam');
            await GigaAmAPI.loadModel(model.name);
            await saveAndSelect('gigaam', model.name);
        } catch (e) {
            setLoadingId(null);
            toast.error('GigaAM: не удалось загрузить модель', {
                description: e instanceof Error ? e.message : String(e),
                duration: 5000,
            });
        }
    };

    const handleSelectTone = async (model: ToneModelInfo) => {
        if (model.status !== 'Available' || loadingId) return;
        try {
            setLoadingId('tone');
            await ToneAPI.loadModel(model.name);
            await saveAndSelect('tone', model.name);
        } catch (e) {
            setLoadingId(null);
            toast.error('T-One: не удалось загрузить модель', {
                description: e instanceof Error ? e.message : String(e),
                duration: 5000,
            });
        }
    };

    if (loading) {
        return (
            <div className="animate-pulse space-y-3">
                <div className="h-20 bg-gray-100 rounded-lg" />
                <div className="h-20 bg-gray-100 rounded-lg" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-800">Ошибка: {error}</p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            {/* GigaAM RNN-T — High Accuracy */}
            {gigaAmModels.map((model) => {
                const isAvailable = model.status === 'Available';
                const isSelected = selected?.engine === 'gigaam' && selected.name === model.name;
                const isThisLoading = loadingId === 'gigaam';

                return (
                    <ModelCard
                        key={`gigaam:${model.name}`}
                        icon="🎯"
                        title="GigaAM v3"
                        badge="High Accuracy"
                        badgeColor="bg-blue-100 text-blue-700"
                        subtitle="RNN-T E2E • Высокое качество распознавания"
                        size={`${model.size_mb} MB`}
                        isAvailable={isAvailable}
                        isSelected={isSelected}
                        isLoading={isThisLoading}
                        isDisabled={!!loadingId && !isThisLoading}
                        onClick={() => handleSelectGigaAm(model)}
                        missingHint="Поместите encoder.onnx, decoder.onnx, joint.onnx, vocab.json в папку gigaam/gigaam-v3-e2e-rnnt/"
                    />
                );
            })}

            {/* T-One CTC — Lightweight Fast */}
            {toneModels.map((model) => {
                const isAvailable = model.status === 'Available';
                const isSelected = selected?.engine === 'tone' && selected.name === model.name;
                const isThisLoading = loadingId === 'tone';

                return (
                    <ModelCard
                        key={`tone:${model.name}`}
                        icon="⚡"
                        title="T-One"
                        badge="Lightweight Fast ASR"
                        badgeColor="bg-green-100 text-green-700"
                        subtitle="CTC Streaming • Быстрое и лёгкое распознавание"
                        size={`${model.size_mb} MB`}
                        isAvailable={isAvailable}
                        isSelected={isSelected}
                        isLoading={isThisLoading}
                        isDisabled={!!loadingId && !isThisLoading}
                        onClick={() => handleSelectTone(model)}
                        missingHint="Поместите model.onnx и vocab.json в папку russianAsr/t-one/"
                    />
                );
            })}
        </div>
    );
}

interface ModelCardProps {
    icon: string;
    title: string;
    badge: string;
    badgeColor: string;
    subtitle: string;
    size: string;
    isAvailable: boolean;
    isSelected: boolean;
    isLoading: boolean;
    isDisabled: boolean;
    onClick: () => void;
    missingHint: string;
}

function ModelCard({
    icon, title, badge, badgeColor, subtitle, size,
    isAvailable, isSelected, isLoading, isDisabled, onClick, missingHint,
}: ModelCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            onClick={() => isAvailable && !isDisabled && onClick()}
            className={`
        relative rounded-lg border-2 p-4 transition-all
        ${isSelected && isAvailable ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white'}
        ${isAvailable && !isDisabled ? 'cursor-pointer hover:border-gray-300' : 'cursor-default opacity-60'}
      `}
        >
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">{icon}</span>
                    <div>
                        <div className="flex items-center gap-2 flex-wrap">
                            <h3 className="font-semibold text-gray-900">{title}</h3>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeColor}`}>
                                {badge}
                            </span>
                            {isSelected && isAvailable && (
                                <span className="bg-blue-600 text-white px-2 py-0.5 rounded-full text-xs font-medium">✓</span>
                            )}
                        </div>
                        <p className="text-sm text-gray-600 mt-0.5">{subtitle}</p>
                        <p className="text-xs text-gray-400 mt-0.5">{size} • Русский язык</p>
                    </div>
                </div>

                <div className="ml-4 flex items-center shrink-0">
                    {isLoading ? (
                        <span className="text-xs text-blue-600 font-medium animate-pulse">Загрузка…</span>
                    ) : isAvailable ? (
                        <div className="flex items-center gap-1.5 text-green-600">
                            <div className="w-2 h-2 bg-green-500 rounded-full" />
                            <span className="text-xs font-medium">Ready</span>
                        </div>
                    ) : (
                        <span className="text-xs text-gray-400">Не найдена</span>
                    )}
                </div>
            </div>

            {!isAvailable && (
                <p className="mt-2 text-xs text-amber-600 bg-amber-50 rounded p-2">
                    {missingHint}
                </p>
            )}
        </motion.div>
    );
}
