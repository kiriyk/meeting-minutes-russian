import { afterAll, afterEach, describe, expect, spyOn, test } from "bun:test";
import { act, create, type ReactTestRenderer } from "react-test-renderer";

const core = await import("@tauri-apps/api/core");
const invoke = spyOn(core, "invoke").mockImplementation(
  async (command: string): Promise<any> => {
    const models: Record<string, string> = {
      whisper_get_available_models: "base",
      parakeet_get_available_models: "parakeet-test",
      gigaam_get_available_models: "gigaam-v3-e2e-rnnt",
      tone_get_available_models: "t-one",
    };
    return models[command]
      ? [{ name: models[command], size_mb: 100, status: "Available" }]
      : (undefined as any);
  },
);
const { useTranscriptionModels } = await import(
  "../../src/hooks/useTranscriptionModels"
);
let state: ReturnType<typeof useTranscriptionModels>;
let renderer: ReactTestRenderer | undefined;
function View({ config, providers }: any) {
  state = useTranscriptionModels(config, providers);
  return null;
}
afterEach(async () => {
  await act(async () => renderer?.unmount());
  renderer = undefined;
  invoke.mockClear();
});
afterAll(() => invoke.mockRestore());
async function fetch(config: any, providers?: string[]) {
  await act(async () => {
    renderer = create(<View config={config} providers={providers} />);
  });
  await act(async () => {
    await state.fetchModels();
  });
}
describe("batch transcription model selection", () => {
  test("retranscription selects the saved composite T-one model", async () => {
    await fetch({ provider: "russianAsr", model: "tone:t-one" }, [
      "whisper",
      "parakeet",
      "gigaam",
      "tone",
    ]);
    expect(state.availableModels.map((m) => m.provider)).toEqual([
      "whisper",
      "parakeet",
      "gigaam",
      "tone",
    ]);
    expect(state.selectedModelKey).toBe("tone:t-one");
  });
  test("recognizes the bare GigaAM onboarding setting", async () => {
    await fetch({ provider: "russianAsr", model: "gigaam-v3-e2e-rnnt" }, [
      "whisper",
      "parakeet",
      "gigaam",
      "tone",
    ]);
    expect(state.selectedModelKey).toBe("gigaam:gigaam-v3-e2e-rnnt");
  });
  test("import keeps its original model providers", async () => {
    await fetch({ provider: "russianAsr", model: "tone:t-one" });
    expect(state.availableModels.map((m) => m.provider)).toEqual([
      "whisper",
      "parakeet",
    ]);
    expect(
      invoke.mock.calls.some(
        ([command]) => command === "tone_get_available_models",
      ),
    ).toBe(false);
  });
  test("refresh does not override a manual model choice", async () => {
    await fetch({ provider: "localWhisper", model: "base" }, [
      "whisper",
      "parakeet",
      "gigaam",
      "tone",
    ]);
    await act(async () => {
      state.setSelectedModelKey("tone:t-one");
    });
    await act(async () => {
      await state.fetchModels();
    });
    expect(state.selectedModelKey).toBe("tone:t-one");
  });
});
