# speakrs 0.5.0 compatibility copy

Source: published crate `speakrs` 0.5.0, upstream revision
`96f3cec0756bc9df7f830a9ed31e782ddd445edd`.
Apache-2.0 license and upstream README are retained. Model weights have their own
CC-BY-4.0 attribution; they are downloaded separately, not vendored here.

Meetily's Silero dependency pins `ort = 2.0.0-rc.10`, which cannot coexist with
rc.12 in the same process (`ort-sys` has a unique native `links` value).
This copy preserves the upstream inference/clustering algorithms and adapts:

- Cargo dependencies to ort rc.10, ndarray 0.16 and corresponding NPY/LAPACK crates.
- ONNX Runtime execution-provider names and tensor/session API spellings.
- macOS linear algebra to SDK BLAS/LAPACK (Accelerate); Windows x86_64 uses
  static Intel MKL (the upstream x86 default), Linux uses static OpenBLAS.
  No Homebrew runtime dependency on macOS.
- Online model management is disabled; Meetily owns pinned, cancellable downloads.
- CPU observation hooks check progress and cancellation between segmentation
  windows and masked embedding chunks, then before/after clustering. Successful
  execution retains the upstream algorithms and results.

The integration uses CPU inference. CoreML/GPU source is retained for comparison
with upstream, but those features are not supported by this compatibility copy.
Upstream fixture-dependent tests are not part of the Meetily workspace suite;
Meetily tests the integration and real downloaded ONNX models separately.
