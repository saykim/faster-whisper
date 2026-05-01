"""MLX Whisper 엔진 선택/결과 정규화 자동 검증."""
from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from transcriber import Transcriber


class FakeMlxWhisper:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio_path: str, **kwargs):
        self.calls.append((audio_path, kwargs))
        return {
            "text": " 안녕하세요 ",
            "language": "ko",
            "language_probability": 0.91,
            "duration": 1.2,
            "segments": [{"text": "안녕하세요", "start": 0.0, "end": 1.2}],
        }


class FailingMlxWhisper:
    def transcribe(self, audio_path: str, **kwargs):  # noqa: ARG002
        raise RuntimeError("mlx failed")


class FakeFasterWhisperModule:
    class WhisperModel:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D107
            pass

        def transcribe(self, *args, **kwargs):  # noqa: D102
            segment = type("Segment", (), {"text": "폴백 결과"})()
            info = type(
                "Info",
                (),
                {"language": "ko", "language_probability": 1.0, "duration": 1.0},
            )()
            return iter([segment]), info


def test_auto_selects_mlx_on_apple_silicon() -> None:
    fake_mlx = FakeMlxWhisper()
    with patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                transcriber = Transcriber(model_size="small", language="ko")
                assert transcriber.active_engine == "mlx-whisper"
                assert transcriber.supports_vad is False
                text, segments, info = transcriber.transcribe_verbose(Path("sample.wav"))

    assert text == "안녕하세요"
    assert segments[0].text == "안녕하세요"
    assert info.language == "ko"
    assert info.language_probability == 0.91
    assert fake_mlx.calls[0][1]["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
    assert fake_mlx.calls[0][1]["condition_on_previous_text"] is False
    assert fake_mlx.calls[0][1]["verbose"] is False


def test_auto_falls_back_to_faster_whisper_without_mlx() -> None:
    with patch.dict(sys.modules, {"mlx_whisper": None}):
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                transcriber = Transcriber(model_size="small", language="ko")
                assert transcriber.active_engine == "faster-whisper"
                assert transcriber.supports_vad is True


def test_mlx_model_override() -> None:
    fake_mlx = FakeMlxWhisper()
    with patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
        transcriber = Transcriber(
            model_size="small",
            language="ko",
            engine="mlx-whisper",
            mlx_model="local/model",
        )
        text, _segments, _info = transcriber.transcribe_verbose("sample.wav", vad_filter=False)

    assert text == "안녕하세요"
    assert fake_mlx.calls[0][1]["path_or_hf_repo"] == "local/model"


def test_mlx_reads_pcm16_wav_without_ffmpeg() -> None:
    fake_mlx = FakeMlxWhisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        audio = (np.array([0, 16384, -16384], dtype=np.int16)).tobytes()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio)

        with patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
            transcriber = Transcriber(model_size="small", language="ko", engine="mlx-whisper")
            text, _segments, _info = transcriber.transcribe_verbose(wav_path, vad_filter=False)
    finally:
        wav_path.unlink(missing_ok=True)

    assert text == "안녕하세요"
    audio_arg = fake_mlx.calls[0][0]
    assert isinstance(audio_arg, np.ndarray)
    assert audio_arg.dtype == np.float32
    assert np.max(audio_arg) == 0.5


def test_mlx_keeps_non_16khz_wav_as_path() -> None:
    fake_mlx = FakeMlxWhisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        audio = (np.array([0, 16384, -16384], dtype=np.int16)).tobytes()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio)

        with patch.dict(sys.modules, {"mlx_whisper": fake_mlx}):
            transcriber = Transcriber(model_size="small", language="ko", engine="mlx-whisper")
            transcriber.transcribe_verbose(wav_path, vad_filter=False)
    finally:
        wav_path.unlink(missing_ok=True)

    assert fake_mlx.calls[0][0] == str(wav_path)


def test_mlx_runtime_failure_falls_back_to_faster_whisper() -> None:
    with patch.dict(
        sys.modules,
        {
            "mlx_whisper": FailingMlxWhisper(),
            "faster_whisper": FakeFasterWhisperModule(),
        },
    ):
        transcriber = Transcriber(model_size="small", language="ko", engine="mlx-whisper")
        text, segments, info = transcriber.transcribe_verbose("sample.wav")

    assert text == "폴백 결과"
    assert segments[0].text == "폴백 결과"
    assert info.language == "ko"
    assert transcriber.active_engine == "faster-whisper"


def main() -> int:
    checks = [
        ("Apple Silicon 자동 MLX 선택", test_auto_selects_mlx_on_apple_silicon),
        ("MLX 미설치 시 faster-whisper 폴백", test_auto_falls_back_to_faster_whisper_without_mlx),
        ("MLX 모델 override", test_mlx_model_override),
        ("MLX PCM16 WAV 직접 로드", test_mlx_reads_pcm16_wav_without_ffmpeg),
        ("MLX 비16kHz WAV 경로 유지", test_mlx_keeps_non_16khz_wav_as_path),
        ("MLX 런타임 실패 시 faster-whisper 폴백", test_mlx_runtime_failure_falls_back_to_faster_whisper),
    ]

    for name, check in checks:
        print(f"[검증] {name}")
        try:
            check()
        except Exception as exc:
            print(f"  실패: {exc!r}")
            return 1
        print("  통과")

    print()
    print("MLX Transcriber 자동 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
