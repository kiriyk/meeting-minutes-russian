import React, { useState, useEffect, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import { GigaAmAPI, GigaAmModelInfo, GigaAmModelStatus } from '../lib/gigaam';
import { ToneAPI, ToneModelInfo, ToneModelStatus } from '../lib/tone';

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

function getDownloadProgress(status: GigaAmModelStatus | ToneModelStatus): number | null {
    if (typeof status === 'object' && 'Downloading' in status) return status.Downloading;
    return null;
}

function isAvailable(status: GigaAmModelStatus | ToneModelStatus): boolean {
    return status === 'Available';
}

function isMissing(status: GigaAmModelStatus | ToneModelStatus): boolean {
    return status === 'Missing';
}

function isError(status: GigaAmModelStatus | ToneModelStatus): boolean {
    return typeof status === 'object' && 'Error' in status;
}

function getErrorMessage(status: GigaAmModelStatus | ToneModelStatus): string {
    if (typeof status === 'object' && 'Error' in status) return status.Error;
    return '';
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
    const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());
    const [error, setError] = useState<string | null>(null);

    // Stable refs for callbacks
    const onModelSelectRef = useRef(onModelSelect);
    const autoSaveRef = useRef(autoSave);
    const progressThrottleRef = useRef<Map<string, { progress: number; timestamp: number }>>(new Map());

    useEffect(() => {
        onModelSelectRef.current = onModelSelect;
        autoSaveRef.current = autoSave;
    }, [onModelSelect, autoSave]);

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

    // GigaAM model loading events
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

    // T-One model loading events
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

    // GigaAM download progress events
    useEffect(() => {
        let unlistenProgress: (() => void) | null = null;
        let unlistenComplete: (() => void) | null = null;
        let unlistenError: (() => void) | null = null;

        const setup = async () => {
            unlistenProgress = await listen<{ modelName: string; progress: number }>(
                'gigaam-model-download-progress',
                (event) => {
                    const { modelName, progress } = event.payload;
                    const now = Date.now();
                    const throttleData = progressThrottleRef.current.get(`gigaam:${modelName}`);
                    const shouldUpdate = !throttleData ||
                        now - throttleData.timestamp > 300 ||
                        Math.abs(progress - throttleData.progress) >= 5;

                    if (shouldUpdate) {
                        progressThrottleRef.current.set(`gigaam:${modelName}`, { progress, timestamp: now });
                        setGigaAmModels(prev =>
                            prev.map(m => m.name === modelName
                                ? { ...m, status: { Downloading: progress } as GigaAmModelStatus }
                                : m
                            )
                        );
                    }
                }
            );

            unlistenComplete = await listen<{ modelName: string }>(
                'gigaam-model-download-complete',
                (event) => {
                    const { modelName } = event.payload;
                    setGigaAmModels(prev =>
                        prev.map(m => m.name === modelName ? { ...m, status: 'Available' as GigaAmModelStatus } : m)
                    );
                    setDownloadingModels(prev => { const s = new Set(prev); s.delete(`gigaam:${modelName}`); return s; });
                    progressThrottleRef.current.delete(`gigaam:${modelName}`);
                    toast.success('🎯 GigaAM v3 готов!', { description: 'Модель загружена и готова к использованию', duration: 4000 });
                    if (onModelSelectRef.current) saveAndSelect('gigaam', modelName);
                }
            );

            unlistenError = await listen<{ modelName: string; error: string }>(
                'gigaam-model-download-error',
                (event) => {
                    const { modelName, error } = event.payload;
                    setGigaAmModels(prev =>
                        prev.map(m => m.name === modelName ? { ...m, status: { Error: error } as GigaAmModelStatus } : m)
                    );
                    setDownloadingModels(prev => { const s = new Set(prev); s.delete(`gigaam:${modelName}`); return s; });
                    progressThrottleRef.current.delete(`gigaam:${modelName}`);
                    toast.error('Ошибка загрузки GigaAM', {
                        description: error,
                        duration: 6000,
                        action: { label: 'Повторить', onClick: () => downloadGigaAm(modelName) }
                    });
                }
            );
        };
        setup();
        return () => { unlistenProgress?.(); unlistenComplete?.(); unlistenError?.(); };
    }, []);

    // T-One download progress events
    useEffect(() => {
        let unlistenProgress: (() => void) | null = null;
        let unlistenComplete: (() => void) | null = null;
        let unlistenError: (() => void) | null = null;

        const setup = async () => {
            unlistenProgress = await listen<{ modelName: string; progress: number }>(
                'tone-model-download-progress',
                (event) => {
                    const { modelName, progress } = event.payload;
                    const now = Date.now();
                    const throttleData = progressThrottleRef.current.get(`tone:${modelName}`);
                    const shouldUpdate = !throttleData ||
                        now - throttleData.timestamp > 300 ||
                        Math.abs(progress - throttleData.progress) >= 5;

                    if (shouldUpdate) {
                        progressThrottleRef.current.set(`tone:${modelName}`, { progress, timestamp: now });
                        setToneModels(prev =>
                            prev.map(m => m.name === modelName
                                ? { ...m, status: { Downloading: progress } as ToneModelStatus }
                                : m
                            )
                        );
                    }
                }
            );

            unlistenComplete = await listen<{ modelName: string }>(
                'tone-model-download-complete',
                (event) => {
                    const { modelName } = event.payload;
                    setToneModels(prev =>
                        prev.map(m => m.name === modelName ? { ...m, status: 'Available' as ToneModelStatus } : m)
                    );
                    setDownloadingModels(prev => { const s = new Set(prev); s.delete(`tone:${modelName}`); return s; });
                    progressThrottleRef.current.delete(`tone:${modelName}`);
                    toast.success('⚡ T-One готов!', { description: 'Модель загружена и готова к использованию', duration: 4000 });
                    if (onModelSelectRef.current) saveAndSelect('tone', modelName);
                }
            );

            unlistenError = await listen<{ modelName: string; error: string }>(
                'tone-model-download-error',
                (event) => {
                    const { modelName, error } = event.payload;
                    setToneModels(prev =>
                        prev.map(m => m.name === modelName ? { ...m, status: { Error: error } as ToneModelStatus } : m)
                    );
                    setDownloadingModels(prev => { const s = new Set(prev); s.delete(`tone:${modelName}`); return s; });
                    progressThrottleRef.current.delete(`tone:${modelName}`);
                    toast.error('Ошибка загрузки T-One', {
                        description: error,
                        duration: 6000,
                        action: { label: 'Повторить', onClick: () => downloadTone(modelName) }
                    });
                }
            );
        };
        setup();
        return () => { unlistenProgress?.(); unlistenComplete?.(); unlistenError?.(); };
    }, []);

    const saveAndSelect = async (engine: ActiveEngine, modelName: string) => {
        const fullId = `${engine}:${modelName}`;
        if (autoSaveRef.current) {
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
        onModelSelectRef.current?.(fullId);
    };

    // GigaAM handlers
    const downloadGigaAm = async (modelName: string) => {
        const key = `gigaam:${modelName}`;
        if (downloadingModels.has(key)) return;
        setDownloadingModels(prev => new Set([...prev, key]));
        setGigaAmModels(prev => prev.map(m => m.name === modelName ? { ...m, status: { Downloading: 0 } as GigaAmModelStatus } : m));
        toast.info('Загрузка GigaAM v3...', { description: 'Это может занять несколько минут (~851 MB)', duration: 5000 });
        try {
            await GigaAmAPI.downloadModel(modelName);
        } catch (e) {
            setDownloadingModels(prev => { const s = new Set(prev); s.delete(key); return s; });
        }
    };

    const cancelGigaAm = async (modelName: string) => {
        try {
            await GigaAmAPI.cancelDownload(modelName);
            setDownloadingModels(prev => { const s = new Set(prev); s.delete(`gigaam:${modelName}`); return s; });
            setGigaAmModels(prev => prev.map(m => m.name === modelName ? { ...m, status: 'Missing' as GigaAmModelStatus } : m));
            toast.info('Загрузка GigaAM отменена', { duration: 3000 });
        } catch (e) {
            toast.error('Не удалось отменить загрузку', { duration: 3000 });
        }
    };

    const deleteGigaAm = async (modelName: string) => {
        try {
            await GigaAmAPI.deleteModel(modelName);
            setGigaAmModels(prev => prev.map(m => m.name === modelName ? { ...m, status: 'Missing' as GigaAmModelStatus } : m));
            toast.success('GigaAM удалён', { description: 'Файлы модели удалены для освобождения места', duration: 3000 });
            if (selected?.engine === 'gigaam' && selected.name === modelName) {
                onModelSelect?.('');
            }
        } catch (e) {
            toast.error('Не удалось удалить GigaAM', { description: e instanceof Error ? e.message : String(e), duration: 4000 });
        }
    };

    const handleSelectGigaAm = async (model: GigaAmModelInfo) => {
        if (!isAvailable(model.status) || loadingId) return;
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

    // T-One handlers
    const downloadTone = async (modelName: string) => {
        const key = `tone:${modelName}`;
        if (downloadingModels.has(key)) return;
        setDownloadingModels(prev => new Set([...prev, key]));
        setToneModels(prev => prev.map(m => m.name === modelName ? { ...m, status: { Downloading: 0 } as ToneModelStatus } : m));
        toast.info('Загрузка T-One...', { description: 'Это может занять несколько минут (~138 MB)', duration: 5000 });
        try {
            await ToneAPI.downloadModel(modelName);
        } catch (e) {
            setDownloadingModels(prev => { const s = new Set(prev); s.delete(key); return s; });
        }
    };

    const cancelTone = async (modelName: string) => {
        try {
            await ToneAPI.cancelDownload(modelName);
            setDownloadingModels(prev => { const s = new Set(prev); s.delete(`tone:${modelName}`); return s; });
            setToneModels(prev => prev.map(m => m.name === modelName ? { ...m, status: 'Missing' as ToneModelStatus } : m));
            toast.info('Загрузка T-One отменена', { duration: 3000 });
        } catch (e) {
            toast.error('Не удалось отменить загрузку', { duration: 3000 });
        }
    };

    const deleteTone = async (modelName: string) => {
        try {
            await ToneAPI.deleteModel(modelName);
            setToneModels(prev => prev.map(m => m.name === modelName ? { ...m, status: 'Missing' as ToneModelStatus } : m));
            toast.success('T-One удалён', { description: 'Файлы модели удалены для освобождения места', duration: 3000 });
            if (selected?.engine === 'tone' && selected.name === modelName) {
                onModelSelect?.('');
            }
        } catch (e) {
            toast.error('Не удалось удалить T-One', { description: e instanceof Error ? e.message : String(e), duration: 4000 });
        }
    };

    const handleSelectTone = async (model: ToneModelInfo) => {
        if (!isAvailable(model.status) || loadingId) return;
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
                <div className="h-24 bg-gray-100 rounded-lg" />
                <div className="h-24 bg-gray-100 rounded-lg" />
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
            {/* GigaAM RNN-T cards */}
            {gigaAmModels.map((model) => {
                const key = `gigaam:${model.name}`;
                const isSelected = selected?.engine === 'gigaam' && selected.name === model.name;
                const isThisLoading = loadingId === 'gigaam';
                return (
                    <ModelCard
                        key={key}
                        icon="🎯"
                        title="GigaAM v3"
                        badge="High Accuracy"
                        badgeColor="bg-blue-100 text-blue-700"
                        subtitle="RNN-T E2E • Высокое качество распознавания"
                        sizeMb={model.size_mb}
                        status={model.status}
                        isSelected={isSelected}
                        isLoading={isThisLoading}
                        isDisabled={!!loadingId && !isThisLoading}
                        onClick={() => handleSelectGigaAm(model)}
                        onDownload={() => downloadGigaAm(model.name)}
                        onCancel={() => cancelGigaAm(model.name)}
                        onDelete={() => deleteGigaAm(model.name)}
                    />
                );
            })}

            {/* T-One CTC cards */}
            {toneModels.map((model) => {
                const key = `tone:${model.name}`;
                const isSelected = selected?.engine === 'tone' && selected.name === model.name;
                const isThisLoading = loadingId === 'tone';
                return (
                    <ModelCard
                        key={key}
                        icon="⚡"
                        title="T-One"
                        badge="Lightweight Fast"
                        badgeColor="bg-green-100 text-green-700"
                        subtitle="CTC Streaming • Быстрое и лёгкое распознавание"
                        sizeMb={model.size_mb}
                        status={model.status}
                        isSelected={isSelected}
                        isLoading={isThisLoading}
                        isDisabled={!!loadingId && !isThisLoading}
                        onClick={() => handleSelectTone(model)}
                        onDownload={() => downloadTone(model.name)}
                        onCancel={() => cancelTone(model.name)}
                        onDelete={() => deleteTone(model.name)}
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
    sizeMb: number;
    status: GigaAmModelStatus | ToneModelStatus;
    isSelected: boolean;
    isLoading: boolean;
    isDisabled: boolean;
    onClick: () => void;
    onDownload: () => void;
    onCancel: () => void;
    onDelete: () => void;
}

function ModelCard({
    icon, title, badge, badgeColor, subtitle, sizeMb,
    status, isSelected, isLoading, isDisabled,
    onClick, onDownload, onCancel, onDelete,
}: ModelCardProps) {
    const [isHovered, setIsHovered] = useState(false);

    const available = isAvailable(status);
    const missing = isMissing(status);
    const hasError = isError(status);
    const downloadProgress = getDownloadProgress(status);
    const isDownloading = downloadProgress !== null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`
                relative rounded-lg border-2 transition-all
                ${isSelected && available
                    ? 'border-blue-500 bg-blue-50'
                    : available
                        ? 'border-gray-200 hover:border-gray-300 bg-white'
                        : missing || hasError
                            ? 'border-gray-200 bg-gray-50 hover:border-blue-300'
                            : 'border-gray-200 bg-gray-50'
                }
                ${(available && !isDisabled) || missing || hasError ? 'cursor-pointer' : 'cursor-default'}
            `}
            onClick={() => {
                if (available && !isDisabled) onClick();
                else if ((missing || hasError) && !isDownloading && !isLoading) onDownload();
            }}
        >
            <div className="p-4">
                <div className="flex items-start justify-between mb-1">
                    <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                            <span className="text-2xl">{icon}</span>
                            <h3 className="font-semibold text-gray-900">{title}</h3>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${badgeColor}`}>
                                {badge}
                            </span>
                            {isSelected && available && (
                                <motion.span
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    className="bg-blue-600 text-white px-2 py-0.5 rounded-full text-xs font-medium"
                                >
                                    ✓
                                </motion.span>
                            )}
                        </div>
                        <p className="text-sm text-gray-600 ml-9">{subtitle}</p>
                        <p className="text-xs text-gray-400 ml-9 mt-0.5">{sizeMb} MB • Русский язык</p>
                    </div>

                    {/* Status / Action area */}
                    <div className="ml-4 flex items-center gap-2 shrink-0">
                        {isLoading && (
                            <span className="text-xs text-blue-600 font-medium animate-pulse">Загрузка…</span>
                        )}

                        {!isLoading && available && (
                            <>
                                <div className="flex items-center gap-1.5 text-green-600">
                                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                                    <span className="text-xs font-medium">Ready</span>
                                </div>
                                <AnimatePresence>
                                    {isHovered && (
                                        <motion.button
                                            initial={{ opacity: 0, scale: 0.8 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.8 }}
                                            transition={{ duration: 0.15 }}
                                            onClick={(e) => { e.stopPropagation(); onDelete(); }}
                                            className="text-gray-400 hover:text-red-600 transition-colors p-1"
                                            title="Удалить модель"
                                        >
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                        </motion.button>
                                    )}
                                </AnimatePresence>
                            </>
                        )}

                        {!isLoading && !isDownloading && missing && (
                            <button
                                onClick={(e) => { e.stopPropagation(); onDownload(); }}
                                className="bg-blue-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
                            >
                                Скачать
                            </button>
                        )}

                        {!isLoading && !isDownloading && hasError && (
                            <button
                                onClick={(e) => { e.stopPropagation(); onDownload(); }}
                                className="bg-red-600 text-white px-3 py-1.5 rounded-md text-sm font-medium hover:bg-red-700 transition-colors"
                            >
                                Повторить
                            </button>
                        )}
                    </div>
                </div>

                {/* Download progress bar */}
                {isDownloading && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-3 pt-3 border-t border-gray-200"
                    >
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-blue-600">Загрузка...</span>
                                <span className="text-sm font-semibold text-blue-600">
                                    {Math.round(downloadProgress!)}%
                                </span>
                            </div>
                            <button
                                onClick={(e) => { e.stopPropagation(); onCancel(); }}
                                className="text-xs text-gray-600 hover:text-red-600 font-medium transition-colors px-2 py-1 rounded hover:bg-red-50"
                            >
                                Отмена
                            </button>
                        </div>
                        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${downloadProgress}%` }}
                                transition={{ duration: 0.3, ease: 'easeOut' }}
                            />
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                            {Math.round(sizeMb * downloadProgress! / 100)} MB / {sizeMb} MB
                        </p>
                    </motion.div>
                )}

                {/* Error message */}
                {hasError && !isDownloading && (
                    <p className="mt-2 text-xs text-red-600 bg-red-50 rounded p-2">
                        {getErrorMessage(status)}
                    </p>
                )}
            </div>
        </motion.div>
    );
}
