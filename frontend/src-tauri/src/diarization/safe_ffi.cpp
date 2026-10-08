// Opaque pointers keep the shim independent of header locations. Configuration
// and segment layouts are supplied by the pinned sherpa-onnx-sys Rust bindings.
#include <cstdio>
#include <exception>

extern "C" {
const void *SherpaOnnxCreateOfflineSpeakerDiarization(const void *config);
const void *SherpaOnnxOfflineSpeakerDiarizationProcess(const void *diarizer,
                                                     const float *samples,
                                                     int length);
const void *SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(const void *result);
}

static thread_local char error_message[1024];

template <typename F> const void *guard(F operation) noexcept {
  error_message[0] = '\0';
  try {
    return operation();
  } catch (const std::exception &error) {
    std::snprintf(error_message, sizeof(error_message), "%s", error.what());
  } catch (...) {
    std::snprintf(error_message, sizeof(error_message), "Unknown native speaker inference error");
  }
  return nullptr;
}

extern "C" const char *meetily_diarization_error() noexcept { return error_message; }
extern "C" const void *meetily_diarization_create(const void *config) noexcept {
  return guard([&]() { return SherpaOnnxCreateOfflineSpeakerDiarization(config); });
}
extern "C" const void *meetily_diarization_process(const void *diarizer,
                                                const float *samples,
                                                int length) noexcept {
  return guard([&]() { return SherpaOnnxOfflineSpeakerDiarizationProcess(diarizer, samples, length); });
}
extern "C" const void *meetily_diarization_sort(const void *result) noexcept {
  return guard([&]() { return SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result); });
}
