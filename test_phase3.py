"""Phase 3 자동 검증.

글로벌 핫키 자체는 macOS 권한과 실제 키 입력이 필요하므로 자동화하지 않는다.
대신 Phase 3 진입점에서 순수 함수/옵션 파싱이 정상인지 확인한다.
"""
from __future__ import annotations

import sys

import numpy as np

from main import audio_levels, build_parser


def check_default_options() -> None:
    args = build_parser().parse_args([])
    assert args.model == "small"
    assert args.mode == "session"
    assert args.hotkey == "<cmd>+<shift>+r"
    assert args.keep_wav is False
    assert args.no_vad is False
    assert args.vad_min_silence_ms == 2000
    assert args.hotkey_debounce_ms == 1000
    assert args.chunk_seconds == 300
    assert args.output_dir == "~/QuickSTT/recordings"


def check_custom_options() -> None:
    args = build_parser().parse_args(
        ["--model", "medium", "--no-vad", "--vad-min-silence-ms", "3000"]
    )
    assert args.model == "medium"
    assert args.no_vad is True
    assert args.vad_min_silence_ms == 3000


def check_audio_levels() -> None:
    audio = np.array([[0], [16384], [-16384], [32767]], dtype=np.int16)
    peak, rms = audio_levels(audio)
    assert 0.99 < peak <= 1.0
    assert rms > 0.0


def main() -> int:
    checks = [
        ("기본 옵션", check_default_options),
        ("커스텀 옵션", check_custom_options),
        ("오디오 레벨 계산", check_audio_levels),
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
    print("Phase 3 자동 검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
