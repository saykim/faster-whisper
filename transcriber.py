"""faster-whisper 래퍼.

QuickSTT 전체에서 사용할 음성 인식 엔진을 한 곳에서 관리한다.
Apple Silicon에서는 CTranslate2가 Metal/CoreML을 직접 지원하지 않으므로
기본값으로 CPU + int8 양자화를 사용한다(메모리/속도 균형).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


class Transcriber:
    """faster-whisper 모델 래퍼.

    모델은 첫 호출 시점에 lazy-load 되며, 이후 인스턴스 수명 동안 재사용된다.
    """

    def __init__(
        self,
        model_size: str = "medium",
        language: str = "ko",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def load(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

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
        """변환 결과 + 세그먼트 리스트 + faster-whisper info(언어 감지 등)."""
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
