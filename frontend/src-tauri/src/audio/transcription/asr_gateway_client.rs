use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use log::{info, warn};
use serde_json::json;
use tauri::{AppHandle, Emitter, Runtime};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::audio::AudioChunk;

pub struct AsrGatewayClient {
    sender: mpsc::UnboundedSender<String>,
    session_id: String,
    seq: u64,
    started: bool,
    engines: Vec<String>,
}

impl AsrGatewayClient {
    pub async fn connect<R: Runtime>(
        app: AppHandle<R>,
        port: u16,
        engines: Vec<String>,
    ) -> Option<Self> {
        let url = format!("ws://127.0.0.1:{}/ws", port);
        let ws_result = connect_async(&url).await;

        let (ws_stream, _) = match ws_result {
            Ok(v) => v,
            Err(e) => {
                warn!("ASR gateway connection failed ({}): {}", url, e);
                return None;
            }
        };

        let (mut write, mut read) = ws_stream.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();

        tokio::spawn(async move {
            while let Some(payload) = rx.recv().await {
                if let Err(e) = write.send(Message::Text(payload.into())).await {
                    warn!("ASR gateway write failed: {}", e);
                    break;
                }
            }
        });

        let app_for_events = app.clone();
        tokio::spawn(async move {
            while let Some(msg_result) = read.next().await {
                match msg_result {
                    Ok(Message::Text(text)) => {
                        if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) {
                            let msg_type = payload
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();

                            if msg_type == "partial_transcript" || msg_type == "final_segment" {
                                let _ = app_for_events.emit("asr-transcript-update", payload);
                            } else if msg_type == "error" {
                                let _ = app_for_events.emit("asr-service-error", payload);
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!("ASR gateway read failed: {}", e);
                        break;
                    }
                }
            }
        });

        let session_id = format!("rust-live-{}", uuid::Uuid::new_v4());
        info!(
            "Connected to ASR gateway on {} with session {}",
            url, session_id
        );

        Some(Self {
            sender: tx,
            session_id,
            seq: 0,
            started: false,
            engines,
        })
    }

    pub fn start_session(&mut self, sample_rate: u32) {
        if self.started {
            return;
        }

        let msg = json!({
            "type": "start_session",
            "session_id": self.session_id,
            "sample_rate": sample_rate,
            "format": "pcm_s16le",
            "channels": 1,
            "engines": self.engines.clone(),
            "vad": { "enabled": true, "mode": "silero", "aggressiveness": 2 },
            "chunking": {
                "live_frame_ms": 20,
                "t_one_emit_ms": 200,
                "gigaam_segment_target_s": 10,
                "gigaam_segment_max_s": 20
            }
        });

        let _ = self.sender.send(msg.to_string());
        self.started = true;
    }

    pub fn send_audio_chunk(&mut self, chunk: &AudioChunk) {
        if !self.started {
            self.start_session(chunk.sample_rate);
        }

        let pcm_s16le = convert_f32_to_pcm_s16le(&chunk.data);
        let msg = json!({
            "type": "audio_chunk",
            "session_id": self.session_id,
            "seq": self.seq,
            "timestamp_ms": (chunk.timestamp * 1000.0) as u64,
            "data_b64": BASE64_STANDARD.encode(pcm_s16le)
        });

        self.seq += 1;
        let _ = self.sender.send(msg.to_string());
    }

    pub fn end_session(&mut self) {
        if !self.started {
            return;
        }

        let msg = json!({
            "type": "end_session",
            "session_id": self.session_id,
        });

        let _ = self.sender.send(msg.to_string());
        self.started = false;
    }
}

fn convert_f32_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let scaled = (clamped * i16::MAX as f32) as i16;
        out.extend_from_slice(&scaled.to_le_bytes());
    }

    out
}
