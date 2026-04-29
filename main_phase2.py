"""Phase 2 진입점: 마이크 녹음 -> 임시 wav -> 변환 -> 콘솔 출력.

Enter 키로 녹음을 시작하고, 다시 Enter 키로 종료한다.
글로벌 단축키 통합은 Phase 3에서 추가된다.

사용 예:
    python main_phase2.py
    python main_phase2.py --model small   # 빠른 테스트용
    python main_phase2.py --keep-wav      # 임시 wav 보존(디버깅)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from recorder import Recorder
from transcriber import Transcriber


def _audio_levels(audio: np.ndarray) -> tuple[float, float]:
    """int16 PCM에서 peak(0~1), RMS(0~1) 정규화 값을 계산."""
    if audio.size == 0:
        return 0.0, 0.0
    a = audio.astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    return peak, rms


def main() -> int:
    parser = argparse.ArgumentParser(description="QuickSTT Phase 2")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper 모델 크기 (기본 medium)",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="녹음된 임시 wav 파일을 보존",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Silero VAD 필터를 비활성화 (조용한 녹음 디버깅용)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        help="VAD가 구간을 나누는 최소 무음 길이(ms, 기본 2000)",
    )
    args = parser.parse_args()

    print(f"==> 모델 로드 중: {args.model} (ko)")
    t0 = time.perf_counter()
    transcriber = Transcriber(model_size=args.model, language="ko")
    transcriber.load()
    print(f"    ✓ 준비 완료 ({time.perf_counter() - t0:.1f}s)")
    print()

    input("Enter 를 눌러 녹음 시작... ")
    recorder = Recorder(sample_rate=16000, channels=1)
    recorder.start()
    rec_started_at = time.perf_counter()
    print("    ● 녹음 중... (Enter 로 종료)")

    try:
        input()
    except KeyboardInterrupt:
        print()
        print("    ! 사용자 중단")

    audio = recorder.stop()
    duration = time.perf_counter() - rec_started_at

    if audio.size == 0:
        print("✗ 녹음된 오디오가 없습니다.")
        return 1

    peak, rms = _audio_levels(audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    recorder.save_wav(audio, wav_path)
    size_kb = wav_path.stat().st_size / 1024
    print(f"    ✓ 녹음 종료 ({duration:.1f}s, {size_kb:.1f} KiB, peak={peak:.3f}, rms={rms:.3f})")

    if peak < 0.01:
        print("    ⚠ 입력 신호가 거의 없음 — 마이크 권한/입력 디바이스 확인 필요")

    print(
        "==> 변환 중... (vad_filter={}, min_silence={}ms)".format(
            not args.no_vad,
            args.vad_min_silence_ms if not args.no_vad else "off",
        )
    )
    t0 = time.perf_counter()
    text, segments, info = transcriber.transcribe_verbose(
        wav_path,
        vad_filter=(not args.no_vad),
        vad_min_silence_duration_ms=args.vad_min_silence_ms,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"    ✓ 완료 ({elapsed:.2f}s, segments={len(segments)}, "
        f"detected={info.language} p={info.language_probability:.2f}, "
        f"audio_dur={info.duration:.2f}s)"
    )

    print()
    print("--- 변환 결과 ---")
    print(text if text else "(빈 결과 — segments=0 이면 VAD가 음성을 못 잡았을 가능성. --no-vad 로 재시도)")
    print("------------------")

    if args.keep_wav:
        print(f"(임시 wav 보존됨: {wav_path})")
    else:
        wav_path.unlink(missing_ok=True)

    return 0 if text else 1


if __name__ == "__main__":
    sys.exit(main())
