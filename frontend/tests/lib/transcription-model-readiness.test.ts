import { describe, expect, test } from 'bun:test';

import {
  getProviderCommands,
  hasDownloadingModel,
} from '../../src/lib/transcription-model-readiness';

describe('transcription model readiness', () => {
  test('uses Whisper commands when Whisper is the configured provider', () => {
    expect(getProviderCommands('localWhisper')).toEqual({
      initialize: 'whisper_init',
      hasAvailableModels: 'whisper_has_available_models',
      getAvailableModels: 'whisper_get_available_models',
    });
  });

  test('uses Parakeet commands when Parakeet is the configured provider', () => {
    expect(getProviderCommands('parakeet')).toEqual({
      initialize: 'parakeet_init',
      hasAvailableModels: 'parakeet_has_available_models',
      getAvailableModels: 'parakeet_get_available_models',
    });
  });

  test('uses GigaAM commands for a GigaAM Russian ASR model', () => {
    const gigaam = {
      initialize: 'gigaam_init',
      hasAvailableModels: 'gigaam_has_available_models',
      getAvailableModels: 'gigaam_get_available_models',
    };
    expect(getProviderCommands('russianAsr', 'gigaam:gigaam-v3-e2e-rnnt')).toEqual(gigaam);
    expect(getProviderCommands('russianAsr', 'gigaam-v3-e2e-rnnt')).toEqual(gigaam);
  });

  test('uses T-One commands for a T-One Russian ASR model', () => {
    const tone = {
      initialize: 'tone_init',
      hasAvailableModels: 'tone_has_available_models',
      getAvailableModels: 'tone_get_available_models',
    };
    expect(getProviderCommands('russianAsr', 'tone:t-one')).toEqual(tone);
    expect(getProviderCommands('russianAsr', 't-one')).toEqual(tone);
  });

  test('rejects a Russian ASR model it cannot attribute to an engine', () => {
    expect(getProviderCommands('russianAsr', 'whisper-large')).toBeNull();
    expect(getProviderCommands('russianAsr')).toBeNull();
  });

  test('does not silently treat an unsupported provider as Parakeet', () => {
    expect(getProviderCommands('deepgram')).toBeNull();
  });

  test('recognizes only active downloads', () => {
    expect(hasDownloadingModel([{ status: 'Available' }])).toBeFalse();
    expect(hasDownloadingModel([{ status: 'Downloading' }])).toBeTrue();
    expect(hasDownloadingModel([{ status: { Downloading: { progress: 0 } } }])).toBeTrue();
    expect(hasDownloadingModel([{ status: { Downloading: { progress: 42 } } }])).toBeTrue();
  });
});
