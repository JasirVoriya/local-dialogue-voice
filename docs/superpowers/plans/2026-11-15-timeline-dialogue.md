# Timeline dialogue implementation plan

Goal: preserve free dubbing and add local, sentence-specific voice references and fixed-position audio exports.
Architecture: keep FastAPI task runner and vanilla UI; introduce timeline data/audio/service modules and a lazy Qwen backend with bounded in-memory prompt LRU. Persist versioned editable segments and generation metadata inside existing jobs. No remote model or media transport.
Spec: the user's detailed upgrade request of 2026-11-15.

Constraints: Python 3.12, uv, existing Mac MPS/CPU float32/eager loading, 127.0.0.1; no weights/media in Git; no invented TTS speed/emotion arguments. Existing ASR environment and model cache stay intact.

Design decisions:
- Qwen consumes exactly ONE local audio/transcript pair per segment. Global reference is an explicit, same-speaker fallback, never concatenated with the full source. Unknown speaker identity does not authorize expanding across neighboring sentences.
- Natural, fit, custom duration strategies; atempo limited to 0.8–1.25. Large mismatch stays natural, warns, and blocks timeline export if overflowing a slot. Never truncate words silently or shift subsequent slots.
- Timeline exports have source duration, preserve gaps, reject overlaps/overflow. Replacement removes the entire source mixture in replaced slots; no claim of music preservation.
- Failed alignment creates no invented timestamps; users can create manual segments and confirm matching references.
- IndexTTS/OmniVoice are extension points only until tested on actual Apple hardware. Remote inference and source separation are not installed or enabled.

## Task 1: timeline records and reference cache
- [x] Add failing tests for natural sentence boundaries, strict ranges, reference correspondence and LRU identity/invalidation.
- [x] Implement timeline/models.py and tts/base.py, reference.py, qwen3_tts.py, registry.py.
- [x] Verify tests without downloading model weights.

## Task 2: audio and generation engine
- [x] Add failing tests for 0–3, 5–8, 10–14 placement, atempo bounds, single regeneration and fallback/speaker safety.
- [x] Implement timeline/audio.py and service.py; make per-segment generation transactional and retain other files on regeneration.
- [x] Exercise real FFmpeg WAV/MP3 operations on synthetic authorized fixtures.

## Task 3: API and backward compatibility
- [x] Add API tests for upload, persisted migration, save/split/merge, task locks, exports, cleanup and free-generation regression.
- [x] Integrate timeline/api.py and app.py; lazy-load torch only with Qwen. Fix old full-source reference assembly and reuse backend/cache for free mode.
- [x] Run the full test suite.

## Task 4: UI and delivery
- [x] Extend existing HTML with accessible mode switch, sentence editing, speaker/reference controls, selection, split/merge, durations and exports.
- [x] Validate JavaScript and exercise browser workflows with a synthetic video and fake TTS only in tests.
- [x] Update README, compatibility and verification record; review diff, run all checks, commit, push branch, open PR. Do not merge.

Review focus: stale media after edits; malformed/NaN ranges; concurrent save/delete during inference; reference text after partial word selection; overflow clipping/replacement collisions.

## Execution / review ledger

Tasks 1–4 implemented and tested in the feature checkout; no main edits or merge.
Ruling: use a fresh feature clone/branch rather than an additional worktree; it already isolates this work from the user's local checkout.
Ruling: proceed from the user's complete specification without an additional design-approval round. Qwen local support and truthful compatibility boundaries are explicitly authorized.
Ruling: real neural inference is unverified because this environment has no Apple Silicon, weights, or authorized human speech fixture. No quality improvement claimed.
Ruling: browser automation could not run: Chromium failed at OS process singleton socket permissions; separate desktop browser could not reach the local server. Browser test remains opt-in locally. Subsequent GitHub Actions run 43686802780 executed it in real Chromium and passed (1 test); local permission limits do not apply to that runner.
Final review: fresh reviewer found three important issues (sample boundary overflow, incomplete numeric draft rendering, ASCII sentence terminators), reproduced RED and fixed GREEN.
Review follow-ups on dynamic busy controls, stale preview metadata, ID/raw filename collisions, and free-reference word boundaries were upgraded to correctness repairs and covered by Python/Node regressions. No reported finding remains intentionally deferred.
