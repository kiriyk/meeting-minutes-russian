# testdata

`jfk-speech-4s.wav` is the first 4.0 s of `jfk.wav` from the whisper.cpp sample set
(JFK's 1961 inaugural address, US government work, public domain). It was already
16 kHz mono PCM16, so it was cropped in place with Python's stdlib `wave` module
(`w.readframes(4.0 * 16000)` then written back with the same format) — no re-encoding,
no new dependencies.
