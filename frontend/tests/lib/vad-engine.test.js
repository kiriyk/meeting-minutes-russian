import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

const invokeMock = mock(async () => null);
mock.module("@tauri-apps/api/core", () => ({ invoke: invokeMock }));

function installLocalStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: (k) => values.get(k) ?? null,
        setItem: (k, v) => values.set(k, v),
        removeItem: (k) => values.delete(k),
      },
    },
  });
  return values;
}

const { readStoredVadEngine, applyVadEngine, DEFAULT_VAD_ENGINE } = await import("../../src/lib/vad");

describe("VAD engine preference", () => {
  beforeEach(() => invokeMock.mockClear());
  // These tests stub globalThis.window; remove it afterwards so it doesn't leak
  // into other test files that assume a Node-like (window-less) environment.
  afterEach(() => {
    delete globalThis.window;
  });

  test("defaults to Silero v6", () => {
    installLocalStorage();
    expect(DEFAULT_VAD_ENGINE).toBe("silero_v6");
    expect(readStoredVadEngine()).toBe("silero_v6");
  });

  test("reads a stored engine", () => {
    installLocalStorage({ vadEngine: "earshot" });
    expect(readStoredVadEngine()).toBe("earshot");
  });

  test("ignores unknown stored values such as the removed v4", () => {
    installLocalStorage({ vadEngine: "silero_v4" });
    expect(readStoredVadEngine()).toBe("silero_v6");
  });

  test("apply persists and syncs to Rust", async () => {
    const values = installLocalStorage();
    await applyVadEngine("earshot");
    expect(values.get("vadEngine")).toBe("earshot");
    expect(invokeMock).toHaveBeenCalledWith("set_vad_engine", { engine: "earshot" });
  });

  test("apply still syncs to Rust when storage throws", async () => {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: { localStorage: { getItem: () => null, setItem: () => { throw new Error("quota"); } } },
    });
    await applyVadEngine("earshot");
    expect(invokeMock).toHaveBeenCalledWith("set_vad_engine", { engine: "earshot" });
  });
});
