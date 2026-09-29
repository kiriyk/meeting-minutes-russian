import React from "react";
import {
  afterAll,
  afterEach,
  beforeEach,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";
import { act, create, type ReactTestRenderer } from "react-test-renderer";

const modules = [
  "@tauri-apps/api/core",
  "@tauri-apps/api/event",
  "../../src/contexts/ConfigContext",
  "../../src/components/ui/dialog",
  "../../src/components/ui/select",
  "../../src/lib/analytics",
  "sonner",
];
const originals = await Promise.all(
  modules.map(async (path) => ({ ...(await import(path)) })),
);
const handlers = new Map<string, (event: any) => void>();
let pendingListeners: (() => void)[] = [];
let delayListeners = false;
let available = true;
let statusReply: Promise<any> | null = null;
const closed = mock();
const warning = mock();
const core = await import("@tauri-apps/api/core");
const invoke = spyOn(core, "invoke").mockImplementation(
  async (
    command: string,
    _args?: Parameters<typeof core.invoke>[1],
  ): Promise<any> => {
    if (command.endsWith("_get_available_models"))
      return available
        ? [
            {
              name: command.startsWith("tone") ? "t-one" : "base",
              size_mb: 100,
              status: "Available",
            },
          ]
        : [];
    if (command === "diarization_get_model_status")
      return (
        statusReply ?? { available: false, downloading: false, progress: 0 }
      );
  },
);
const listen = mock(async (name: string, handler: (event: any) => void) => {
  if (delayListeners)
    await new Promise<void>((resolve) => pendingListeners.push(resolve));
  handlers.set(name, handler);
  return () => handlers.delete(name);
});
const Box = ({ children }: any) => <div>{children}</div>;
const config = { provider: "russianAsr", model: "tone:t-one" };
function installMocks() {
  mock.module(modules[1], () => ({ ...originals[1], listen }));
  mock.module(modules[2], () => ({
    ...originals[2],
    useConfig: () => ({
      selectedLanguage: "en",
      transcriptModelConfig: config,
    }),
  }));
  mock.module(modules[3], () =>
    Object.fromEntries(
      [
        "Dialog",
        "DialogContent",
        "DialogDescription",
        "DialogFooter",
        "DialogHeader",
        "DialogTitle",
      ].map((key) => [key, Box]),
    ),
  );
  mock.module(modules[4], () =>
    Object.fromEntries(
      [
        "Select",
        "SelectContent",
        "SelectItem",
        "SelectTrigger",
        "SelectValue",
      ].map((key) => [key, Box]),
    ),
  );
  mock.module(modules[5], () => ({
    default: { track: async () => {}, trackError: async () => {} },
  }));
  mock.module("sonner", () => ({
    ...originals[6],
    toast: { success: mock(), warning, info: mock(), error: mock() },
  }));
}
installMocks();
const { RetranscribeDialog } =
  await import("../../src/components/MeetingDetails/RetranscribeDialog");
let view: ReactTestRenderer | undefined;
beforeEach(() => {
  installMocks();
  handlers.clear();
  pendingListeners = [];
  delayListeners = false;
  available = true;
  statusReply = null;
  invoke.mockClear();
  closed.mockClear();
  warning.mockClear();
});
afterEach(async () => {
  await act(async () => view?.unmount());
  view = undefined;
  for (const resolve of pendingListeners) resolve();
});
afterAll(() => {
  invoke.mockRestore();
  modules
    .slice(1)
    .forEach((path, i) => mock.module(path, () => originals[i + 1]));
});
async function show() {
  await act(async () => {
    view = create(
      <RetranscribeDialog
        open
        onOpenChange={closed}
        meetingId="meeting"
        meetingFolderPath="/recording"
      />,
    );
  });
}
function textOf(node: any): string {
  return typeof node === "string" ? node : node.children.map(textOf).join("");
}
function button(text: string) {
  return view!.root
    .findAllByType("button")
    .find((b) => textOf(b).includes(text))!;
}
async function click(text: string) {
  await act(async () => {
    await button(text).props.onClick();
  });
}

test("waits for event listeners and a downloaded ASR model before enabling start", async () => {
  delayListeners = true;
  await show();
  expect(button("Start Retranscription").props.disabled).toBe(true);
  await act(async () => {
    delayListeners = false;
    for (const resolve of pendingListeners) resolve();
  });
  expect(button("Start Retranscription").props.disabled).toBe(false);
});

test("routes T-one with Russian language and speaker identification enabled", async () => {
  await show();
  await click("Start Retranscription");
  const call = invoke.mock.calls.find(
    ([command]) => command === "start_retranscription_command",
  );
  expect(call?.[1]).toMatchObject({
    provider: "tone",
    model: "t-one",
    language: "ru",
    diarizationEnabled: true,
    diarizationModel: "community-1",
  });
});

test("downloads the selected Community-1 bundle", async () => {
  await show();
  await click("Download speaker models");
  expect(
    invoke.mock.calls.find(
      ([command]) => command === "diarization_download_models",
    )?.[1],
  ).toMatchObject({ model: "community-1" });
});

test("ignores download completion for a different speaker model", async () => {
  await show();
  await act(async () => {
    handlers.get("diarization-model-download-complete")!({
      payload: { model: "legacy" },
    });
  });
  expect(button("Download speaker models")).toBeDefined();
});

test("keeps model status usable while another meeting reports transcription progress", async () => {
  let resolveStatus!: (status: any) => void;
  statusReply = new Promise((resolve) => {
    resolveStatus = resolve;
  });
  await show();
  await act(async () => {
    handlers.get("retranscription-progress")!({
      payload: { meeting_id: "another-meeting", progress_percentage: 50 },
    });
    resolveStatus({ available: false, downloading: false, progress: 0 });
  });
  expect(button("Download speaker models").props.disabled).toBe(false);
});

test("a late status query does not erase live download progress", async () => {
  let resolveStatus!: (status: any) => void;
  statusReply = new Promise((resolve) => {
    resolveStatus = resolve;
  });
  await show();
  await act(async () => {
    handlers.get("diarization-model-download-progress")!({
      payload: { model: "community-1", progress: 40 },
    });
    resolveStatus({ available: false, downloading: false, progress: 0 });
  });
  expect(button("Cancel download")).toBeDefined();
});

test("a late status query does not restore another model's completed download", async () => {
  let resolveStatus!: (status: any) => void;
  statusReply = new Promise((resolve) => {
    resolveStatus = resolve;
  });
  await show();
  await act(async () => {
    handlers.get("diarization-model-download-progress")!({
      payload: { model: "legacy", progress: 50 },
    });
    handlers.get("diarization-model-download-complete")!({
      payload: { model: "legacy" },
    });
    resolveStatus({
      available: false,
      downloading: false,
      progress: 0,
      active_download: "legacy",
    });
  });
  expect(button("Download speaker models").props.disabled).toBe(false);
});

test("cancellation stays busy until the native job emits its terminal event", async () => {
  await show();
  await click("Start Retranscription");
  await click("Cancel");
  expect(closed).not.toHaveBeenCalled();
  expect(button("Cancellation requested").props.disabled).toBe(true);
  await act(async () => {
    handlers.get("retranscription-error")!({
      payload: { meeting_id: "meeting", error: "Retranscription cancelled" },
    });
  });
  expect(closed).toHaveBeenCalledWith(false);
});

test("surfaces native speaker fallback warnings on successful completion", async () => {
  await show();
  await click("Start Retranscription");
  await act(async () => {
    handlers.get("retranscription-complete")!({
      payload: {
        meeting_id: "meeting",
        duration_seconds: 3,
        segments_count: 1,
        warnings: ["Speakers unavailable"],
      },
    });
  });
  expect(warning).toHaveBeenCalledWith("Speakers unavailable");
});

test("an empty list of downloaded ASR models disables start", async () => {
  available = false;
  await show();
  expect(button("Start Retranscription").props.disabled).toBe(true);
});
