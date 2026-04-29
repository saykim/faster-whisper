"""QuickSTT Phase 3: 글로벌 단축키로 녹음 시작/종료.

실행 후 백그라운드에서 대기하다가 `Cmd+Shift+R`을 누르면 녹음을 시작하고,
다시 누르면 녹음을 종료한 뒤 faster-whisper로 한국어 변환을 수행한다.

사용 예:
    python main.py
    python main.py --model small
    python main.py --hotkey '<cmd>+<shift>+r'
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from pynput import keyboard

from recorder import Recorder
from session import LongRecordingSession
from transcriber import Transcriber


def audio_levels(audio: np.ndarray) -> tuple[float, float]:
    """int16 PCM에서 peak(0~1), RMS(0~1) 정규화 값을 계산."""
    if audio.size == 0:
        return 0.0, 0.0
    a = audio.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    return peak, rms


class QuickSTTApp:
    """핫키 이벤트를 짧은 녹음/장시간 세션 상태 전이로 연결."""

    def __init__(
        self,
        transcriber: Transcriber,
        *,
        mode: str = "session",
        recorder: Recorder | None = None,
        keep_wav: bool = False,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 2000,
        hotkey_debounce_ms: int = 1000,
        output_dir: str | Path = "~/QuickSTT/recordings",
        chunk_seconds: int = 300,
    ) -> None:
        self.transcriber = transcriber
        self.mode = mode
        self.recorder = recorder or Recorder(sample_rate=16000, channels=1)
        self.keep_wav = keep_wav
        self.vad_filter = vad_filter
        self.vad_min_silence_ms = vad_min_silence_ms
        self.hotkey_debounce_sec = hotkey_debounce_ms / 1000
        self.output_dir = Path(output_dir).expanduser()
        self.chunk_seconds = chunk_seconds
        self._lock = threading.Lock()
        self._state = "idle"
        self._recording_started_at = 0.0
        self._last_hotkey_at = 0.0
        self._session: LongRecordingSession | None = None

    @property
    def state(self) -> str:
        return self._state

    def toggle_recording(self) -> None:
        """핫키 콜백: idle이면 녹음 시작, recording이면 녹음 종료."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_hotkey_at < self.hotkey_debounce_sec:
                return
            self._last_hotkey_at = now

            if self.mode == "once":
                self._toggle_once_locked()
                return
            self._toggle_session_locked()

    def _toggle_once_locked(self) -> None:
        if self._state == "idle":
            self._start_recording_locked()
            return
        if self._state == "recording":
            self._state = "transcribing"
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()
            return
        print("변환 중입니다. 잠시 후 다시 시도하세요.", flush=True)

    def _toggle_session_locked(self) -> None:
        if self._state == "idle":
            self._start_session_locked()
            return
        if self._state == "recording_session":
            self._state = "stopping"
            threading.Thread(target=self._stop_session, daemon=True).start()
            return
        print("세션 정리 중입니다. 잠시 후 다시 시도하세요.", flush=True)

    def shutdown(self) -> None:
        """종료 시 진행 중인 녹음/세션을 가능한 한 정리."""
        with self._lock:
            state = self._state
            session = self._session
            if state not in {"recording", "recording_session"}:
                return
            self._state = "idle"

        if state == "recording_session" and session is not None:
            print("진행 중인 장시간 세션을 마감합니다.", flush=True)
            session.stop(wait=True)
            print(f"transcript: {session.transcript_path}", flush=True)
            return

        try:
            audio = self.recorder.stop()
            print(f"진행 중이던 녹음을 버리고 종료합니다. ({len(audio)} samples)", flush=True)
        except Exception:
            pass

    def _start_session_locked(self) -> None:
        try:
            session = LongRecordingSession(
                transcriber=self.transcriber,
                output_root=self.output_dir,
                chunk_seconds=self.chunk_seconds,
                keep_wav=self.keep_wav,
                vad_filter=self.vad_filter,
                vad_min_silence_ms=self.vad_min_silence_ms,
            )
            session.start()
        except Exception as exc:
            print(f"장시간 세션 시작 실패: {exc}", flush=True)
            self._state = "idle"
            return

        self._session = session
        self._state = "recording_session"
        print(
            f"장시간 세션 시작. {self.chunk_seconds}s 단위로 자동 저장/변환합니다.",
            flush=True,
        )
        print(f"session_dir: {session.session_dir}", flush=True)
        print("다시 단축키를 누르면 세션을 종료합니다.", flush=True)

    def _stop_session(self) -> None:
        session = self._session
        if session is None:
            with self._lock:
                self._state = "idle"
            return
        try:
            print("장시간 세션 종료 중... 남은 청크 변환을 기다립니다.", flush=True)
            session.stop(wait=True)
            print(
                f"세션 종료. 변환 청크={session.completed_chunks}, "
                f"transcript={session.transcript_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"세션 종료 실패: {exc}", flush=True)
        finally:
            with self._lock:
                self._session = None
                self._state = "idle"

    def _start_recording_locked(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            print(f"녹음 시작 실패: {exc}", flush=True)
            self._state = "idle"
            return

        self._recording_started_at = time.perf_counter()
        self._state = "recording"
        print("녹음 시작. 다시 단축키를 누르면 종료합니다.", flush=True)

    def _stop_and_transcribe(self) -> None:
        wav_path: Path | None = None
        try:
            audio = self.recorder.stop()
            duration = time.perf_counter() - self._recording_started_at
            if audio.size == 0:
                print("녹음된 오디오가 없습니다.", flush=True)
                return

            peak, rms = audio_levels(audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = Path(f.name)
            self.recorder.save_wav(audio, wav_path)
            size_kb = wav_path.stat().st_size / 1024
            print(
                f"녹음 종료 ({duration:.1f}s, {size_kb:.1f} KiB, "
                f"peak={peak:.3f}, rms={rms:.3f})",
                flush=True,
            )
            if peak < 0.01:
                print("입력 신호가 거의 없습니다. 마이크 권한/입력 디바이스를 확인하세요.", flush=True)

            print(
                "변환 중... (vad_filter={}, min_silence={}ms)".format(
                    self.vad_filter,
                    self.vad_min_silence_ms if self.vad_filter else "off",
                ),
                flush=True,
            )
            started_at = time.perf_counter()
            text, segments, info = self.transcriber.transcribe_verbose(
                wav_path,
                vad_filter=self.vad_filter,
                vad_min_silence_duration_ms=self.vad_min_silence_ms,
            )
            elapsed = time.perf_counter() - started_at
            print(
                f"변환 완료 ({elapsed:.2f}s, segments={len(segments)}, "
                f"detected={info.language} p={info.language_probability:.2f})",
                flush=True,
            )
            print()
            print("--- 변환 결과 ---")
            print(text if text else "(빈 결과)")
            print("------------------")
            print()
            print("대기 중... 단축키로 다시 녹음할 수 있습니다.", flush=True)
        except Exception as exc:
            print(f"변환 처리 실패: {exc}", flush=True)
        finally:
            if wav_path is not None and not self.keep_wav:
                wav_path.unlink(missing_ok=True)
            with self._lock:
                self._state = "idle"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuickSTT Phase 3")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper 모델 크기 (기본 small)",
    )
    parser.add_argument(
        "--mode",
        default="session",
        choices=["session", "once"],
        help="실행 모드: session=장시간 자동 청크, once=기존 짧은 녹음 (기본 session)",
    )
    parser.add_argument(
        "--hotkey",
        default="<cmd>+<shift>+r",
        help="녹음 시작/종료 글로벌 단축키 (pynput 형식, 기본 '<cmd>+<shift>+r')",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="녹음된 임시 wav 파일을 보존",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Silero VAD 필터를 비활성화",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        help="VAD가 구간을 나누는 최소 무음 길이(ms, 기본 2000)",
    )
    parser.add_argument(
        "--hotkey-debounce-ms",
        type=int,
        default=1000,
        help="핫키 중복 입력 무시 시간(ms, 기본 1000)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=300,
        help="session 모드 청크 길이(초, 기본 300)",
    )
    parser.add_argument(
        "--output-dir",
        default="~/QuickSTT/recordings",
        help="session 모드 출력 루트 디렉터리 (기본 ~/QuickSTT/recordings)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(f"모델 로드 중: {args.model} (ko)")
    started_at = time.perf_counter()
    transcriber = Transcriber(model_size=args.model, language="ko")
    transcriber.load()
    print(f"준비 완료 ({time.perf_counter() - started_at:.1f}s)")

    app = QuickSTTApp(
        transcriber=transcriber,
        mode=args.mode,
        recorder=Recorder(sample_rate=16000, channels=1),
        keep_wav=args.keep_wav,
        vad_filter=(not args.no_vad),
        vad_min_silence_ms=args.vad_min_silence_ms,
        hotkey_debounce_ms=args.hotkey_debounce_ms,
        output_dir=args.output_dir,
        chunk_seconds=args.chunk_seconds,
    )

    stop_event = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"대기 중... 단축키: {args.hotkey}")
    print(f"모드: {args.mode}")
    if args.mode == "session":
        print(f"청크 길이: {args.chunk_seconds}s, 출력 폴더: {Path(args.output_dir).expanduser()}")
    print("종료하려면 Ctrl+C")
    print()

    listener = keyboard.GlobalHotKeys({args.hotkey: app.toggle_recording})
    listener.daemon = True

    try:
        listener.start()
        while not stop_event.is_set():
            time.sleep(0.1)
    except Exception as exc:
        print(f"핫키 리스너 시작 실패: {exc}")
        print("macOS 권한을 확인하세요: 시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용")
        return 1
    finally:
        app.shutdown()
        listener.stop()
        listener.join(timeout=1.0)

    print("종료합니다.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    os._exit(exit_code)
