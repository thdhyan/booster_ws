# k1-chat handoff (state as of 2026-10-27, branch `feat/k1-chatbot`)

## State: working
- P1 llm client + velocity arbiter; P2 local voice on all 6 K1s (A1-A3, B1-B3, fw 1.8.1.2): mic 6ch -> mean of live ch0-3 (mic gain 16/25)
  -> energy VAD -> faster-whisper tiny.en (CPU) -> OpenRouter free model -> Kokoro-82M (GPU) -> aplay. Bench: Kokoro ~1 s / 5 s sentence, ASR ~1-2 s.
- `scripts/k1voice.sh start|stop|smoke|log|fetch|calibrate|mic-test|status|restore <robot>`; `start` = async voice chat (new LLM call per utterance,
  stale replies dropped, mic muted while speaking + 0.5 s). Audio dumps: `k1voice.sh fetch` -> `audio_dumps/<robot>/`. Details: `docs/AUDIO_PIPELINE.md`.
- Model: `K1_CHAT_MODEL=cohere/north-mini-code:free` (tool-capable, ~1 s). nemotron-3.5-lightning is a reasoning model and returned empty content at low max_tokens.
  Free models 429 often; fallback `poolside/laguna-s-2.1:free`. System prompt: `src/k1_chat/system_prompt.md` (to be rewritten).
- Live-tested on A3: calibrate + smoke OK. Chat loop with the new model + async: NOT yet tested live.

## Known issues
- ASR sometimes takes 6-12 s on 3-5 s clips and invents text from noise ("...let us know in the comments below"). Not diagnosed: check `utt_*.wav`/`.txt` dumps; try faster-whisper `vad_filter`, `no_speech_threshold`, `condition_on_previous_text=False`, a minimum utterance rms, or `base.en`.
- Per-robot threshold: run `calibrate <robot>` for each (only A3 done). Gain is an ALSA setting (lost on reboot, also affects the robot's own audio).
- Orphan python on robot if ssh drops during `smoke` (run.sh restores audio on TERM; kill by hand otherwise).
- Never run `booster-cli upgrade`/`upload_log`; chat must never move the robot or change locomotion mode; free OpenRouter models only.

## Next steps
### 1. Tool calling (skipped on purpose for now)
- Model already supports `tools`. Plan: extend `OpenRouterClient.chat` to return the full message (tool_calls), add a tool registry in `conversation.py`,
  loop: LLM -> tool_call -> run -> `role: tool` result -> LLM -> speak. Rewrite `system_prompt.md` to describe the tools.
- `run_python` tool (user wants code from replies run): DECISION PENDING where. Recommended: sandboxed subprocess on the laptop (timeout, no robot libs, no SDK/ROS),
  result spoken by the robot. On-robot execution could import the Booster SDK and move the robot: only with SDK/ROS stripped from env, still weaker.
- Tests first (fake LLM returning tool_calls), per skill `k1-chat-tdd`.
### 2. Interruption (barge-in)
- Today: mic ignored while speaking (`Audio._speaking` + 0.5 s tail). Options: (a) energy gate well above own-echo level, measured with a calibration pass while the robot speaks;
  (b) software echo cancel (speexdsp/webrtc-audio-processing AEC using the played PCM as reference); (c) push-to-talk/keyword. On barge-in: `player.stop()`, drop pending speech, supersede the reply.
- Needs `Player.stop` to be callable mid-sentence from the listener thread (already `terminate()`), and speak() to abort remaining sentences.
### 3. OpenAI Realtime API (later runs; paid, not free: breaks the free-models-only rule, needs explicit approval + a budget)
- WebSocket speech-to-speech: server VAD, built-in interruption (`input_audio_buffer.speech_started` -> `response.cancel`), function calling, would replace ASR+LLM+TTS and cut latency.
- Reported pricing (verify on OpenAI's page): flagship audio ~$32/M input, ~$64/M output tokens; mini tier ~$10/$20; unofficial ~$0.04/min real use.
  Sources: https://qz.com/openai-realtime-voice-models-developers-050726 , https://aireiter.com/blog/openai-realtime-api-pricing ,
  https://community.openai.com/t/realtimeapi-randomly-cuts-off-the-connection/1102194 (WS drops after function calls) .
- Robot has 16 kHz mono capture; Realtime wants 24 kHz PCM16 (resample). Key must stay off disk on the robot (same `.chat_env` trick as `start`).
- Plan: spike on the laptop with the robot mic streamed over ssh, compare latency to the local pipeline before moving anything.
### 4. Housekeeping
- Calibrate every robot; run `start` on each; note smoke/test results on Slides/Drive (report cadence rule); decide `-m robot` test.
