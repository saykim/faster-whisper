"""마이크 녹음 모듈.

`sounddevice.InputStream` 콜백으로 프레임을 누적하고, ``start()`` ~ ``stop()``
구간의 PCM 데이터를 numpy 배열로 반환한다. wav 저장은 표준 라이브러리 ``wave``
를 사용해 scipy 의존성을 회피한다.
"""
from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class RecordingChunk:
    """닫힌 wav 청크의 메타데이터."""

    path: Path
    started_at: float
    ended_at: float
    frames: int
    sample_rate: int
    peak: float
    rms: float

    @property
    def duration_sec(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def size_kb(self) -> float:
        return self.path.stat().st_size / 1024 if self.path.exists() else 0.0


class Recorder:
    """무한 길이 녹음을 시작/정지로 제어하는 단순 래퍼.

    macOS에서는 첫 녹음 시 시스템에서 마이크 권한 요청 팝업이 발생하며,
    터미널/IDE 단위로 권한이 부여되어야 한다.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        # status는 xrun/overflow 등 누락 경고. 디버깅 시 활성화.
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        if self.is_recording:
            raise RuntimeError("이미 녹음이 진행 중입니다.")
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """녹음을 종료하고 누적된 PCM 데이터를 반환."""
        if not self.is_recording:
            raise RuntimeError("녹음이 시작되지 않았습니다.")
        assert self._stream is not None
        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            if not self._frames:
                return np.zeros((0, self.channels), dtype=self.dtype)
            return np.concatenate(self._frames, axis=0)

    def save_wav(self, audio: np.ndarray, path: str | Path) -> None:
        """PCM int16 numpy 배열을 wav 파일로 저장."""
        if audio.dtype != np.int16:
            raise ValueError(f"int16 PCM만 지원합니다 (현재 dtype={audio.dtype}).")
        n_channels = audio.shape[1] if audio.ndim > 1 else 1
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())


class StreamingWavRecorder:
    """마이크 입력을 메모리에 쌓지 않고 wav 파일에 바로 기록하는 녹음기."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._stream: Optional[sd.InputStream] = None
        self._wave_file: Optional[wave.Wave_write] = None
        self._current_path: Optional[Path] = None
        self._chunk_started_at = 0.0
        self._frames_written = 0
        self._peak = 0.0
        self._sum_squares = 0.0
        self._sample_count = 0
        self._lock = threading.RLock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self, path: str | Path) -> None:
        """새 wav 파일에 스트리밍 녹음을 시작."""
        if self.is_recording:
            raise RuntimeError("이미 녹음이 진행 중입니다.")
        with self._lock:
            self._open_chunk(Path(path))
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._callback,
            )
            self._stream.start()

    def rotate_chunk(self, next_path: str | Path) -> RecordingChunk:
        """현재 청크를 닫고 다음 청크를 연 뒤, 닫힌 청크 메타데이터를 반환."""
        if not self.is_recording:
            raise RuntimeError("녹음이 시작되지 않았습니다.")
        with self._lock:
            closed = self._close_current_chunk()
            self._open_chunk(Path(next_path))
            return closed

    def stop(self) -> RecordingChunk:
        """녹음을 종료하고 마지막 청크를 닫아 반환."""
        if not self.is_recording:
            raise RuntimeError("녹음이 시작되지 않았습니다.")
        with self._lock:
            assert self._stream is not None
            self._stream.stop()
            self._stream.close()
            self._stream = None
            return self._close_current_chunk()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if indata.dtype != np.int16:
            raise ValueError(f"int16 PCM만 지원합니다 (현재 dtype={indata.dtype}).")
        with self._lock:
            if self._wave_file is None:
                return
            self._wave_file.writeframes(indata.tobytes())
            self._frames_written += frames
            normalized = indata.astype(np.float32) / 32768.0
            self._peak = max(self._peak, float(np.max(np.abs(normalized))))
            self._sum_squares += float(np.sum(normalized * normalized))
            self._sample_count += int(normalized.size)

    def _open_chunk(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._wave_file = wave.open(str(path), "wb")
        self._wave_file.setnchannels(self.channels)
        self._wave_file.setsampwidth(2)
        self._wave_file.setframerate(self.sample_rate)
        self._current_path = path
        self._chunk_started_at = time.time()
        self._frames_written = 0
        self._peak = 0.0
        self._sum_squares = 0.0
        self._sample_count = 0

    def _close_current_chunk(self) -> RecordingChunk:
        if self._wave_file is None or self._current_path is None:
            raise RuntimeError("열린 wav 청크가 없습니다.")
        path = self._current_path
        started_at = self._chunk_started_at
        frames = self._frames_written
        peak = self._peak
        rms = (
            float(np.sqrt(self._sum_squares / self._sample_count))
            if self._sample_count
            else 0.0
        )
        self._wave_file.close()
        self._wave_file = None
        self._current_path = None
        return RecordingChunk(
            path=path,
            started_at=started_at,
            ended_at=time.time(),
            frames=frames,
            sample_rate=self.sample_rate,
            peak=peak,
            rms=rms,
        )
