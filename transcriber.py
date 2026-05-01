"""Whisper 엔진 래퍼.

QuickSTT 전체에서 사용할 음성 인식 엔진을 한 곳에서 관리한다.
Apple Silicon에서는 MLX Whisper를 우선 사용하고, 그 외 환경에서는
faster-whisper CPU 경로를 유지한다.
"""
from __future__ import annotations

import importlib
import platform
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np

EngineName = Literal["auto", "faster-whisper", "mlx-whisper"]

MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}


class Transcriber:
    """Whisper 모델 래퍼.

    모델은 첫 호출 시점에 lazy-load 되며, 이후 인스턴스 수명 동안 재사용된다.
    """

    def __init__(
        self,
        model_size: str = "medium",
        language: str = "ko",
        device: str = "cpu",
        compute_type: str = "int8",
        engine: EngineName = "auto",
        mlx_model: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.engine = engine
        self.mlx_model = mlx_model
        self._resolved_engine: Literal["faster-whisper", "mlx-whisper"] | None = None
        self._model: Optional[Any] = None
        self._mlx_module: Any | None = None
        self._warned_mlx_vad = False

    @property
    def active_engine(self) -> str:
        return self._resolved_engine or self._select_engine()

    @property
    def supports_vad(self) -> bool:
        return self.active_engine == "faster-whisper"

    def load(self):
        engine = self._select_engine()
        if engine == "mlx-whisper":
            if self._mlx_module is None:
                self._mlx_module = importlib.import_module("mlx_whisper")
            return self._mlx_module

        if self._model is None:
            faster_whisper = importlib.import_module("faster_whisper")
            self._model = faster_whisper.WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _select_engine(self) -> Literal["faster-whisper", "mlx-whisper"]:
        if self._resolved_engine is not None:
            return self._resolved_engine

        if self.engine == "faster-whisper":
            self._resolved_engine = "faster-whisper"
            return self._resolved_engine

        if self.engine == "mlx-whisper":
            if self._mlx_available():
                self._resolved_engine = "mlx-whisper"
                return self._resolved_engine
            print("mlx-whisper를 사용할 수 없어 faster-whisper로 전환합니다.", flush=True)
            self._resolved_engine = "faster-whisper"
            return self._resolved_engine

        if self._is_apple_silicon() and self._mlx_available():
            self._resolved_engine = "mlx-whisper"
        else:
            self._resolved_engine = "faster-whisper"
        return self._resolved_engine

    def _mlx_available(self) -> bool:
        try:
            importlib.import_module("mlx_whisper")
        except ImportError:
            return False
        return True

    @staticmethod
    def _is_apple_silicon() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def transcribe(
        self,
        audio_path: str | Path,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 2000,
    ) -> str:
        """주어진 오디오 파일을 텍스트로 변환하여 합쳐진 문자열을 반환."""
        text, _segments, _info = self.transcribe_verbose(
            audio_path,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        )
        return text

    def transcribe_verbose(
        self,
        audio_path: str | Path,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 2000,
    ):
        """변환 결과 + 세그먼트 리스트 + info(언어 감지 등)."""
        if self.active_engine == "mlx-whisper":
            try:
                return self._transcribe_mlx(audio_path, vad_filter=vad_filter)
            except Exception as exc:
                print(f"mlx-whisper 변환 실패, faster-whisper로 전환합니다: {exc}", flush=True)
                self._resolved_engine = "faster-whisper"
        return self._transcribe_faster_whisper(
            audio_path,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        )

    def _transcribe_faster_whisper(
        self,
        audio_path: str | Path,
        *,
        beam_size: int,
        vad_filter: bool,
        vad_min_silence_duration_ms: int,
    ):
        model = self.load()
        vad_parameters = (
            {"min_silence_duration_ms": vad_min_silence_duration_ms}
            if vad_filter
            else None
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
        )
        segments = list(segments_iter)
        text = "".join(seg.text for seg in segments).strip()
        return text, segments, info

    def _transcribe_mlx(self, audio_path: str | Path, *, vad_filter: bool):
        if vad_filter and not self._warned_mlx_vad:
            print("MLX 엔진에서는 Silero VAD가 적용되지 않습니다.", flush=True)
            self._warned_mlx_vad = True

        mlx_whisper = self.load()
        model_path = self.mlx_model or MLX_MODEL_MAP.get(
            self.model_size, f"mlx-community/whisper-{self.model_size}"
        )
        audio_input = self._mlx_audio_input(audio_path)
        if isinstance(audio_input, np.ndarray):
            print(
                f"MLX 오디오 준비 완료: {audio_input.size / 16000:.1f}s, "
                f"samples={audio_input.size}",
                flush=True,
            )
        else:
            print(f"MLX 오디오 경로 입력: {audio_input}", flush=True)
        print(f"MLX 변환 시작: model={model_path}", flush=True)
        result = mlx_whisper.transcribe(
            audio_input,
            path_or_hf_repo=model_path,
            language=self.language,
            condition_on_previous_text=False,
            verbose=False,
        )
        text = str(result.get("text", "")).strip()
        segments = [self._normalize_mlx_segment(seg) for seg in result.get("segments", [])]
        info = SimpleNamespace(
            language=result.get("language", self.language),
            language_probability=result.get("language_probability", 0.0),
            duration=result.get("duration", 0.0),
            engine="mlx-whisper",
            model=model_path,
        )
        return text, segments, info

    @staticmethod
    def _normalize_mlx_segment(segment):
        if not isinstance(segment, dict):
            return segment
        return SimpleNamespace(
            text=segment.get("text", ""),
            start=segment.get("start", 0.0),
            end=segment.get("end", 0.0),
        )

    @staticmethod
    def _mlx_audio_input(audio_path: str | Path):
        """QuickSTT가 만든 PCM16 WAV는 직접 읽어 ffmpeg 의존을 피한다."""
        path = Path(audio_path)
        if path.suffix.lower() != ".wav":
            return str(path)
        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        except (OSError, wave.Error):
            return str(path)

        if sample_width != 2 or sample_rate != 16000:
            return str(path)
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return audio.astype(np.float32) / 32768.0
