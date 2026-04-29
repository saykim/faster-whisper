"""장시간 녹음 세션 관리.

단축키로 시작된 하나의 세션을 5분 단위 wav 청크로 자동 분할하고,
닫힌 청크를 백그라운드 워커가 순서대로 변환해 transcript 파일에 누적한다.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from recorder import RecordingChunk, StreamingWavRecorder
from transcriber import Transcriber


class ChunkRecorder(Protocol):
    def start(self, path: str | Path) -> None: ...
    def rotate_chunk(self, next_path: str | Path) -> RecordingChunk: ...
    def stop(self) -> RecordingChunk: ...


@dataclass(frozen=True)
class TranscribedChunk:
    chunk: RecordingChunk
    text: str
    elapsed_sec: float
    segment_count: int


class LongRecordingSession:
    """장시간 녹음/자동 청크 변환 세션."""

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        recorder: ChunkRecorder | None = None,
        output_root: str | Path = "~/QuickSTT/recordings",
        chunk_seconds: int = 300,
        keep_wav: bool = False,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 2000,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds는 1 이상이어야 합니다.")
        self.transcriber = transcriber
        self.recorder = recorder or StreamingWavRecorder(sample_rate=16000, channels=1)
        self.output_root = Path(output_root).expanduser()
        self.chunk_seconds = chunk_seconds
        self.keep_wav = keep_wav
        self.vad_filter = vad_filter
        self.vad_min_silence_ms = vad_min_silence_ms

        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_root / self.session_id
        self.transcript_path = self.session_dir / "transcript.txt"

        self._queue: queue.Queue[RecordingChunk | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self._chunk_index = 0
        self._rotator_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._completed_chunks = 0

    @property
    def completed_chunks(self) -> int:
        return self._completed_chunks

    @property
    def queued_chunks(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("세션이 이미 시작되었습니다.")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._write_session_header()
            self._chunk_index = 1
            self.recorder.start(self._chunk_path(self._chunk_index))
            self._started = True
            self._stop_event.clear()

            self._worker_thread = threading.Thread(
                target=self._worker_loop, name="quickstt-transcriber", daemon=True
            )
            self._rotator_thread = threading.Thread(
                target=self._rotator_loop, name="quickstt-chunk-rotator", daemon=True
            )
            self._worker_thread.start()
            self._rotator_thread.start()

    def stop(self, wait: bool = True) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop_event.set()

        if self._rotator_thread is not None:
            self._rotator_thread.join(timeout=2.0)

        try:
            final_chunk = self.recorder.stop()
            self._enqueue_if_not_empty(final_chunk)
        except Exception as exc:
            self._append_system_line(f"마지막 청크 종료 실패: {exc}")

        self._queue.put(None)
        if wait and self._worker_thread is not None:
            self._worker_thread.join()
        self._append_system_line("세션 종료")

    def _rotator_loop(self) -> None:
        while not self._stop_event.wait(self.chunk_seconds):
            try:
                with self._lock:
                    if not self._started:
                        return
                    closed = self.recorder.rotate_chunk(
                        self._chunk_path(self._chunk_index + 1)
                    )
                    self._chunk_index += 1
                self._enqueue_if_not_empty(closed)
                print(
                    f"청크 저장: {closed.path.name} "
                    f"({closed.duration_sec:.1f}s, {closed.size_kb:.1f} KiB, "
                    f"queue={self.queued_chunks})",
                    flush=True,
                )
            except Exception as exc:
                self._append_system_line(f"청크 회전 실패: {exc}")
                print(f"청크 회전 실패: {exc}", flush=True)

    def _worker_loop(self) -> None:
        while True:
            chunk = self._queue.get()
            try:
                if chunk is None:
                    return
                self._transcribe_and_append(chunk)
            finally:
                self._queue.task_done()

    def _transcribe_and_append(self, chunk: RecordingChunk) -> None:
        print(f"변환 대기 청크 처리: {chunk.path.name}", flush=True)
        started_at = time.perf_counter()
        text, segments, info = self.transcriber.transcribe_verbose(
            chunk.path,
            vad_filter=self.vad_filter,
            vad_min_silence_duration_ms=self.vad_min_silence_ms,
        )
        elapsed = time.perf_counter() - started_at
        result = TranscribedChunk(
            chunk=chunk,
            text=text,
            elapsed_sec=elapsed,
            segment_count=len(segments),
        )
        self._append_transcript(result, detected_language=info.language)
        self._completed_chunks += 1
        print(
            f"변환 완료: {chunk.path.name} "
            f"({elapsed:.2f}s, segments={len(segments)}, detected={info.language})",
            flush=True,
        )
        if not self.keep_wav:
            chunk.path.unlink(missing_ok=True)

    def _enqueue_if_not_empty(self, chunk: RecordingChunk) -> None:
        if chunk.frames <= 0:
            if not self.keep_wav:
                chunk.path.unlink(missing_ok=True)
            return
        self._queue.put(chunk)

    def _chunk_path(self, index: int) -> Path:
        return self.session_dir / f"chunk_{index:04d}.wav"

    def _write_session_header(self) -> None:
        self.transcript_path.write_text(
            "\n".join(
                [
                    "# QuickSTT Transcript",
                    f"session_id: {self.session_id}",
                    f"chunk_seconds: {self.chunk_seconds}",
                    f"started_at: {datetime.now().isoformat(timespec='seconds')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _append_transcript(
        self, result: TranscribedChunk, *, detected_language: str
    ) -> None:
        started = datetime.fromtimestamp(result.chunk.started_at).strftime("%H:%M:%S")
        ended = datetime.fromtimestamp(result.chunk.ended_at).strftime("%H:%M:%S")
        text = result.text if result.text else "(빈 결과)"
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        f"## {result.chunk.path.name} [{started} - {ended}]",
                        (
                            f"- audio: {result.chunk.duration_sec:.1f}s, "
                            f"peak={result.chunk.peak:.3f}, rms={result.chunk.rms:.3f}"
                        ),
                        (
                            f"- transcribe: {result.elapsed_sec:.2f}s, "
                            f"segments={result.segment_count}, "
                            f"detected={detected_language}"
                        ),
                        "",
                        text,
                        "",
                    ]
                )
            )

    def _append_system_line(self, message: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[system] {datetime.now().isoformat(timespec='seconds')} {message}\n")
