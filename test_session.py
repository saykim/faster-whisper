"""장시간 세션 자동 검증.

실제 마이크와 faster-whisper 모델을 사용하지 않고, fake recorder/transcriber로
청크 큐 처리와 transcript 누적 저장을 검증한다.
"""
from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recorder import RecordingChunk
from session import LongRecordingSession


class FakeRecorder:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._current_path: Path | None = None
        self._started_at = 0.0

    def start(self, path: str | Path) -> None:
        self._current_path = Path(path)
        self._started_at = time.time()
        self._write_dummy_wav(self._current_path, duration_sec=0.2)

    def rotate_chunk(self, next_path: str | Path) -> RecordingChunk:
        closed = self._close_current()
        self.start(next_path)
        return closed

    def stop(self) -> RecordingChunk:
        return self._close_current()

    def _close_current(self) -> RecordingChunk:
        if self._current_path is None:
            raise RuntimeError("열린 청크가 없습니다.")
        path = self._current_path
        self._current_path = None
        return RecordingChunk(
            path=path,
            started_at=self._started_at,
            ended_at=time.time(),
            frames=int(self.sample_rate * 0.2),
            sample_rate=self.sample_rate,
            peak=0.3,
            rms=0.1,
        )

    def _write_dummy_wav(self, path: Path, duration_sec: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.sample_rate * duration_sec)
        t = np.linspace(0.0, duration_sec, n, endpoint=False)
        audio = (np.sin(2 * np.pi * 440.0 * t) * 32767 * 0.2).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())


class FakeTranscriber:
    def transcribe_verbose(
        self,
        audio_path: str | Path,
        *,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 2000,
    ):
        stem = Path(audio_path).stem
        return (
            f"{stem} 변환 결과",
            [object()],
            SimpleNamespace(language="ko", language_probability=1.0),
        )


def test_session_writes_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = LongRecordingSession(
            transcriber=FakeTranscriber(),  # type: ignore[arg-type]
            recorder=FakeRecorder(),
            output_root=tmp,
            chunk_seconds=1,
            keep_wav=True,
        )
        session.start()
        closed = session.recorder.rotate_chunk(session.session_dir / "chunk_0002.wav")
        session._enqueue_if_not_empty(closed)  # 테스트 전용: rotator 대기 없이 큐 투입
        session.stop(wait=True)

        transcript = session.transcript_path.read_text(encoding="utf-8")
        assert "QuickSTT Transcript" in transcript
        assert "chunk_0001 변환 결과" in transcript
        assert "chunk_0002 변환 결과" in transcript
        assert session.completed_chunks == 2


def main() -> int:
    print("[검증] 장시간 세션 transcript 누적")
    try:
        test_session_writes_transcript()
    except Exception as exc:
        print(f"  실패: {exc!r}")
        return 1
    print("  통과")
    print()
    print("장시간 세션 자동 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
