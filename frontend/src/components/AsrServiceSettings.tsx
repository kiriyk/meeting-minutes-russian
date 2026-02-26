import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type EngineId = "t_one" | "gigaam";

interface ModelInfo {
  id: string;
  mode: string;
  status: string;
}

interface LocalModelInfo {
  model_id: string;
  engine: string;
  repo_id: string;
  filename: string;
  revision: string;
  local_path: string;
}

interface JobStatusResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  model_id: string;
  engine: string;
  repo_id: string;
  filename: string;
  revision: string;
  target_path: string;
  progress: number;
  error?: string | null;
}

interface AsrGatewayServiceStatus {
  port: number;
  managed: boolean;
  healthy: boolean;
  mode: "managed_running" | "managed_starting" | "external_running" | "stopped";
}

type ServiceState = "disconnected" | "connecting" | "connected";

export function AsrServiceSettings() {
  const [port, setPort] = useState("8765");
  const [serviceState, setServiceState] =
    useState<ServiceState>("disconnected");
  const [serviceHealthy, setServiceHealthy] = useState<boolean | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [localModels, setLocalModels] = useState<LocalModelInfo[]>([]);
  const [engines, setEngines] = useState<Record<EngineId, boolean>>({
    t_one: true,
    gigaam: true,
  });
  const [diarizationEnabled, setDiarizationEnabled] = useState(false);
  const [diarizationMode, setDiarizationMode] = useState("energy");
  const [diarizationToken, setDiarizationToken] = useState("");
  const [liveCaptionsEnabled, setLiveCaptionsEnabled] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const [processStatus, setProcessStatus] =
    useState<AsrGatewayServiceStatus | null>(null);

  const [hfEngine, setHfEngine] = useState<EngineId>("t_one");
  const [hfModelId, setHfModelId] = useState("t-one-base");
  const [hfRepoId, setHfRepoId] = useState("");
  const [hfFilename, setHfFilename] = useState("");
  const [hfRevision, setHfRevision] = useState("main");
  const [activeJobs, setActiveJobs] = useState<
    Record<string, JobStatusResponse>
  >({});

  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<number | null>(null);

  const baseHttpUrl = useMemo(() => `http://127.0.0.1:${port}`, [port]);
  const wsUrl = useMemo(() => `ws://127.0.0.1:${port}/ws`, [port]);

  useEffect(() => {
    const savedPort = localStorage.getItem("asrServicePort");
    if (savedPort) {
      setPort(savedPort);
    }

    const savedTOne = localStorage.getItem("asrEngineTOne");
    const savedGigaam = localStorage.getItem("asrEngineGigaam");
    const savedLiveCaptions = localStorage.getItem("asrLiveCaptionsEnabled");

    if (savedTOne !== null || savedGigaam !== null) {
      setEngines({
        t_one: savedTOne !== null ? savedTOne === "true" : true,
        gigaam: savedGigaam !== null ? savedGigaam === "true" : true,
      });
    }

    if (savedLiveCaptions !== null) {
      setLiveCaptionsEnabled(savedLiveCaptions === "true");
    }

    const savedDiarizationEnabled = localStorage.getItem("asrDiarizationEnabled");
    const savedDiarizationMode = localStorage.getItem("asrDiarizationMode");
    const savedDiarizationToken = localStorage.getItem("asrDiarizationToken");

    if (savedDiarizationEnabled !== null) {
      setDiarizationEnabled(savedDiarizationEnabled === "true");
    }
    if (savedDiarizationMode !== null) {
      setDiarizationMode(savedDiarizationMode);
    }
    if (savedDiarizationToken !== null) {
      setDiarizationToken(savedDiarizationToken);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("asrServicePort", port);
  }, [port]);

  useEffect(() => {
    localStorage.setItem("asrEngineTOne", String(engines.t_one));
    localStorage.setItem("asrEngineGigaam", String(engines.gigaam));
  }, [engines]);

  useEffect(() => {
    localStorage.setItem("asrLiveCaptionsEnabled", String(liveCaptionsEnabled));
  }, [liveCaptionsEnabled]);

  useEffect(() => {
    localStorage.setItem("asrDiarizationEnabled", String(diarizationEnabled));
    localStorage.setItem("asrDiarizationMode", diarizationMode);
    localStorage.setItem("asrDiarizationToken", diarizationToken);
  }, [diarizationEnabled, diarizationMode, diarizationToken]);

  useEffect(() => {
    const syncGatewayConfig = async () => {
      const numericPort = parseInt(port || "8765", 10);
      await invoke("set_asr_gateway_config", {
        enabled: liveCaptionsEnabled,
        port: Number.isFinite(numericPort) ? numericPort : 8765,
        tOneEnabled: engines.t_one,
        gigaamEnabled: engines.gigaam,
        diarizationEnabled: diarizationEnabled,
        diarizationMode: diarizationMode,
        diarizationToken: diarizationToken || null,
      });
    };
    syncGatewayConfig().catch((error) => {
      setLastError(
        error instanceof Error
          ? error.message
          : "Failed to apply ASR gateway configuration",
      );
    });
  }, [liveCaptionsEnabled, port, engines.t_one, engines.gigaam, diarizationEnabled, diarizationMode, diarizationToken]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    const numericPort = parseInt(port || "8765", 10);
    const resolvedPort = Number.isFinite(numericPort) ? numericPort : 8765;

    const poll = async () => {
      try {
        const status = (await invoke("get_asr_gateway_service_status", {
          port: resolvedPort,
        })) as AsrGatewayServiceStatus;
        if (!cancelled) {
          setProcessStatus(status);
        }
      } catch {
        if (!cancelled) {
          setProcessStatus(null);
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, 2000);
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [port, liveCaptionsEnabled]);

  const checkHealth = async () => {
    setLastError(null);
    try {
      const response = await fetch(`${baseHttpUrl}/health`);
      setServiceHealthy(response.ok);
      if (!response.ok) {
        setLastError(`Health check failed with status ${response.status}`);
      }
    } catch (error) {
      setServiceHealthy(false);
      setLastError(
        error instanceof Error
          ? error.message
          : "Failed to connect to ASR service",
      );
    }
  };

  const loadModels = async () => {
    setLastError(null);
    try {
      const response = await fetch(`${baseHttpUrl}/models`);
      if (!response.ok) {
        throw new Error(`Models API returned status ${response.status}`);
      }
      const data = (await response.json()) as { engines?: ModelInfo[] };
      setModels(data.engines || []);
    } catch (error) {
      setModels([]);
      setLastError(
        error instanceof Error ? error.message : "Failed to load models",
      );
    }
  };

  const loadLocalModels = async () => {
    setLastError(null);
    try {
      const response = await fetch(`${baseHttpUrl}/models/local`);
      if (!response.ok) {
        throw new Error(`Local models API returned status ${response.status}`);
      }
      const data = (await response.json()) as { models?: LocalModelInfo[] };
      setLocalModels(data.models || []);
    } catch (error) {
      setLocalModels([]);
      setLastError(
        error instanceof Error
          ? error.message
          : "Failed to load local ASR models",
      );
    }
  };

  const pollJobs = async () => {
    const jobIds = Object.entries(activeJobs)
      .filter(([, job]) => job.status === "queued" || job.status === "running")
      .map(([jobId]) => jobId);
    if (jobIds.length === 0) return;

    for (const jobId of jobIds) {
      try {
        const response = await fetch(`${baseHttpUrl}/jobs/${jobId}`);
        if (!response.ok) continue;
        const status = (await response.json()) as JobStatusResponse;

        setActiveJobs((prev) => ({ ...prev, [jobId]: status }));

        if (status.status === "completed" || status.status === "failed") {
          if (status.status === "completed") {
            await loadLocalModels();
          }
        }
      } catch {
        // ignore polling failure for a single tick
      }
    }
  };

  useEffect(() => {
    const hasInProgressJobs = Object.values(activeJobs).some(
      (job) => job.status === "queued" || job.status === "running",
    );

    if (!hasInProgressJobs) {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    if (!pollRef.current) {
      pollRef.current = window.setInterval(() => {
        void pollJobs();
      }, 1200);
    }

    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [activeJobs]);

  const startHfDownload = async () => {
    setLastError(null);
    if (!hfRepoId.trim() || !hfFilename.trim() || !hfModelId.trim()) {
      setLastError("Model ID, repo ID and filename are required");
      return;
    }

    try {
      const response = await fetch(`${baseHttpUrl}/models/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: hfModelId.trim(),
          engine: hfEngine,
          repo_id: hfRepoId.trim(),
          filename: hfFilename.trim(),
          revision: hfRevision.trim() || "main",
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Download request failed with status ${response.status}`,
        );
      }

      const data = (await response.json()) as { job_id: string };
      setActiveJobs((prev) => ({
        ...prev,
        [data.job_id]: {
          job_id: data.job_id,
          status: "queued",
          model_id: hfModelId.trim(),
          engine: hfEngine,
          repo_id: hfRepoId.trim(),
          filename: hfFilename.trim(),
          revision: hfRevision.trim() || "main",
          target_path: "",
          progress: 0,
          error: null,
        },
      }));
    } catch (error) {
      setLastError(
        error instanceof Error
          ? error.message
          : "Failed to start Hugging Face model download",
      );
    }
  };

  const connect = async () => {
    setLastError(null);
    setServiceState("connecting");
    try {
      await checkHealth();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setServiceState("connected");
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as {
            type?: string;
            message?: string;
          };
          if (parsed.type === "error") {
            setLastError(parsed.message || "ASR service returned an error");
          }
        } catch {
          // Ignore non-JSON messages
        }
      };

      ws.onerror = () => {
        setLastError("WebSocket connection error");
      };

      ws.onclose = () => {
        setServiceState("disconnected");
        wsRef.current = null;
      };
    } catch (error) {
      setServiceState("disconnected");
      setLastError(
        error instanceof Error ? error.message : "Failed to connect",
      );
    }
  };

  const disconnect = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setServiceState("disconnected");
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm space-y-5">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">ASR Service</h3>
          <p className="text-sm text-gray-600">
            Configure local ASR gateway connection for live captions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">
            Live Captions
          </label>
          <div className="flex items-center gap-3">
            <Switch
              checked={liveCaptionsEnabled}
              onCheckedChange={setLiveCaptionsEnabled}
            />
            <span className="text-sm text-gray-700">
              Enable ASR WS captions in main transcript view
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">Port</label>
          <Input
            value={port}
            onChange={(e) => setPort(e.target.value.replace(/[^\d]/g, ""))}
            placeholder="8765"
            className="max-w-[240px]"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">
            Service Status
          </label>
          <div className="flex items-center gap-3">
            <span
              className={`text-sm font-medium ${
                serviceState === "connected"
                  ? "text-green-600"
                  : serviceState === "connecting"
                    ? "text-amber-600"
                    : "text-gray-500"
              }`}
            >
              {serviceState}
            </span>
            <Button variant="outline" onClick={checkHealth}>
              Check Health
            </Button>
            <Button variant="outline" onClick={loadModels}>
              Load Engines
            </Button>
            <Button variant="outline" onClick={loadLocalModels}>
              Load Local Models
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">ASR Process</label>
          <div className="flex items-center gap-3 text-sm">
            <span className="font-medium text-gray-800">
              {processStatus
                ? processStatus.mode === "managed_running"
                  ? "managed"
                  : processStatus.mode === "managed_starting"
                    ? "starting"
                    : processStatus.mode === "external_running"
                      ? "external"
                      : "stopped"
                : "unknown"}
            </span>
            <span className="text-xs text-gray-500">
              health:{" "}
              {processStatus ? (processStatus.healthy ? "ok" : "failed") : "unknown"}
            </span>
            <span className="text-xs text-gray-500">
              port: {processStatus?.port ?? port ?? "8765"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-center">
          <label className="text-sm font-medium text-gray-700">
            Connection
          </label>
          <div className="flex items-center gap-3">
            <Button
              onClick={connect}
              disabled={
                serviceState === "connected" || serviceState === "connecting"
              }
            >
              Connect
            </Button>
            <Button
              variant="outline"
              onClick={disconnect}
              disabled={serviceState === "disconnected"}
            >
              Disconnect
            </Button>
            <span className="text-xs text-gray-500">WS: {wsUrl}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-start">
          <label className="text-sm font-medium text-gray-700">Engines</label>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Switch
                checked={engines.t_one}
                onCheckedChange={(checked) =>
                  setEngines((prev) => ({ ...prev, t_one: checked }))
                }
              />
              <span className="text-sm">T-One (live draft)</span>
            </div>
            <div className="flex items-center gap-3">
              <Switch
                checked={engines.gigaam}
                onCheckedChange={(checked) =>
                  setEngines((prev) => ({ ...prev, gigaam: checked }))
                }
              />
              <span className="text-sm">GigaAM (final segments)</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-3 items-start">
          <label className="text-sm font-medium text-gray-700">Speaker Diarization</label>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Switch
                checked={diarizationEnabled}
                onCheckedChange={setDiarizationEnabled}
              />
              <span className="text-sm">Enable speaker identification</span>
            </div>
            {diarizationEnabled && (
              <>
                <div className="flex items-center gap-3">
                  <Select
                    value={diarizationMode}
                    onValueChange={setDiarizationMode}
                  >
                    <SelectTrigger className="max-w-[200px]">
                      <SelectValue placeholder="Select mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="energy">Energy-based (fast)</SelectItem>
                      <SelectItem value="pyannote">PyAnnote (accurate)</SelectItem>
                    </SelectContent>
                  </Select>
                  <span className="text-xs text-gray-500">Mode</span>
                </div>
                {diarizationMode === "pyannote" && (
                  <div className="flex items-center gap-3">
                    <Input
                      value={diarizationToken}
                      onChange={(e) => setDiarizationToken(e.target.value)}
                      placeholder="HuggingFace token (optional)"
                      className="max-w-[300px]"
                    />
                    <span className="text-xs text-gray-500">Token for PyAnnote</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="text-xs text-gray-600">
          Health:{" "}
          {serviceHealthy === null
            ? "unknown"
            : serviceHealthy
              ? "ok"
              : "failed"}
        </div>

        {models.length > 0 && (
          <div className="p-3 border rounded-md bg-gray-50">
            <div className="text-sm font-medium mb-2">Available Engines</div>
            <div className="space-y-1">
              {models.map((model) => (
                <div key={model.id} className="text-xs text-gray-700">
                  {model.id} | mode: {model.mode} | status: {model.status}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Hugging Face Model Download
          </h3>
          <p className="text-sm text-gray-600">
            Download ASR models directly from Hugging Face for T-One/GigaAM.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-600">Engine</label>
            <Select
              value={hfEngine}
              onValueChange={(v) => setHfEngine(v as EngineId)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select engine" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="t_one">t_one</SelectItem>
                <SelectItem value="gigaam">gigaam</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-xs text-gray-600">Model ID</label>
            <Input
              value={hfModelId}
              onChange={(e) => setHfModelId(e.target.value)}
              placeholder="t-one-base"
            />
          </div>

          <div>
            <label className="text-xs text-gray-600">HF Repo ID</label>
            <Input
              value={hfRepoId}
              onChange={(e) => setHfRepoId(e.target.value)}
              placeholder="org/repo"
            />
          </div>

          <div>
            <label className="text-xs text-gray-600">Filename</label>
            <Input
              value={hfFilename}
              onChange={(e) => setHfFilename(e.target.value)}
              placeholder="model.onnx"
            />
          </div>

          <div>
            <label className="text-xs text-gray-600">Revision</label>
            <Input
              value={hfRevision}
              onChange={(e) => setHfRevision(e.target.value)}
              placeholder="main"
            />
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button onClick={startHfDownload}>Download from Hugging Face</Button>
        </div>

        {Object.keys(activeJobs).length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Download Jobs</div>
            {Object.values(activeJobs).map((job) => (
              <div
                key={job.job_id}
                className="p-3 border rounded-md bg-gray-50 text-xs"
              >
                <div>
                  {job.model_id} ({job.engine})
                </div>
                <div>
                  {job.repo_id} / {job.filename}
                </div>
                <div>
                  status: {job.status} | progress: {job.progress.toFixed(1)}%
                </div>
                {job.error && (
                  <div className="text-red-600">error: {job.error}</div>
                )}
              </div>
            ))}
          </div>
        )}

        {localModels.length > 0 && (
          <div className="space-y-2">
            <div className="text-sm font-medium">Local ASR Models</div>
            {localModels.map((model) => (
              <div
                key={`${model.engine}-${model.model_id}-${model.local_path}`}
                className="p-3 border rounded-md bg-gray-50 text-xs"
              >
                <div>
                  {model.model_id} ({model.engine})
                </div>
                <div>
                  {model.repo_id} / {model.filename} @ {model.revision}
                </div>
                <div className="text-gray-600 break-all">
                  {model.local_path}
                </div>
              </div>
            ))}
          </div>
        )}

        {lastError && (
          <div className="p-3 border rounded-md bg-red-50 text-red-700 text-sm">
            {lastError}
          </div>
        )}
      </div>
    </div>
  );
}
