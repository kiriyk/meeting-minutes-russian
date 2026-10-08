import React from 'react';
import { act, create, type ReactTestRenderer } from 'react-test-renderer';
import { afterEach, beforeEach, expect, mock, test } from 'bun:test';

const vadReplies: Array<() => Promise<unknown>> = [];
const invokeMock = mock(async (command: string, _args?: unknown): Promise<unknown> => {
  if (command === 'set_vad_engine' && vadReplies.length) return vadReplies.shift()!();
  return command === 'get_ollama_models' ? [] : null;
});
mock.module('@tauri-apps/api/core', () => ({ invoke: invokeMock }));
mock.module('@tauri-apps/api/event', () => ({ listen: async () => () => {} }));
mock.module('../../src/services/configService', () => ({
  configService: {
    getTranscriptConfig: async () => null,
    getModelConfig: async () => null,
    getRecordingPreferences: async () => null,
  },
}));

const { ConfigProvider, useConfig } = await import('../../src/contexts/ConfigContext');
type Config = ReturnType<typeof useConfig>;
let config: Config;
let view: ReactTestRenderer | undefined;
let values: Map<string, string>;

function Probe() {
  config = useConfig();
  return <div>{config.vadEngine}</div>;
}

beforeEach(() => {
  values = new Map();
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { localStorage: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    } },
  });
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: globalThis.window.localStorage,
  });
  invokeMock.mockClear();
  vadReplies.length = 0;
});

afterEach(async () => {
  await act(async () => view?.unmount());
  view = undefined;
  delete (globalThis as { window?: Window }).window;
  delete (globalThis as { localStorage?: Storage }).localStorage;
});

test('a rejected VAD selection keeps the last applied engine visible', async () => {
  await act(async () => {
    view = create(<ConfigProvider><Probe /></ConfigProvider>);
  });
  expect(config.vadEngine).toBe('silero_v6');

  vadReplies.push(async () => { throw new Error('IPC unavailable'); });
  await act(async () => { config.setVadEngine('earshot'); });

  expect(config.vadEngine).toBe('silero_v6');
  expect(values.get('vadEngine')).toBe('silero_v6');
  expect(config.vadEngineError).toContain('Фактический выбор неизвестен');
});

test('a stored engine appears as active only after Rust confirms it', async () => {
  values.set('vadEngine', 'earshot');
  let confirm!: () => void;
  vadReplies.push(() => new Promise<void>(resolve => { confirm = resolve; }));

  await act(async () => {
    view = create(<ConfigProvider><Probe /></ConfigProvider>);
  });
  expect(config.vadEngine).toBe('silero_v6');

  await act(async () => { confirm(); });
  expect(config.vadEngine).toBe('earshot');
});

test('rapid changes reach Rust in order and leave the last choice applied', async () => {
  await act(async () => {
    view = create(<ConfigProvider><Probe /></ConfigProvider>);
  });

  let releaseFirst!: () => void;
  vadReplies.push(() => new Promise<void>(resolve => { releaseFirst = resolve; }));
  await act(async () => {
    config.setVadEngine('earshot');
    await Promise.resolve();
    config.setVadEngine('silero_v6');
  });
  expect(invokeMock.mock.calls.filter(([command]) => command === 'set_vad_engine')).toHaveLength(2);
  expect(config.vadEngine).toBe('silero_v6');

  await act(async () => { releaseFirst(); });
  expect(invokeMock.mock.calls.filter(([command]) => command === 'set_vad_engine').map(([, args]) => args)).toEqual([
    { engine: 'silero_v6' },
    { engine: 'earshot' },
    { engine: 'silero_v6' },
  ]);
  expect(config.vadEngine).toBe('silero_v6');
  expect(values.get('vadEngine')).toBe('silero_v6');
});
