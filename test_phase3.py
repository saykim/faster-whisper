"""Phase 3 자동 검증.

글로벌 핫키 자체는 macOS 권한과 실제 키 입력이 필요하므로 자동화하지 않는다.
대신 Phase 3 진입점에서 순수 함수/옵션 파싱이 정상인지 확인한다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main import (
    audio_levels,
    acquire_single_instance_lock,
    build_parser,
    check_hotkey_trust,
    copy_to_clipboard,
    notify,
    play_sound,
)


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
    assert args.engine == "auto"
    assert args.mlx_model is None
    assert args.no_clipboard is False
    assert args.no_notify is False
    assert args.no_sound is False


def check_custom_options() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "medium",
            "--engine",
            "mlx-whisper",
            "--mlx-model",
            "custom/model",
            "--no-vad",
            "--vad-min-silence-ms",
            "3000",
            "--no-clipboard",
            "--no-notify",
            "--no-sound",
        ]
    )
    assert args.model == "medium"
    assert args.engine == "mlx-whisper"
    assert args.mlx_model == "custom/model"
    assert args.no_vad is True
    assert args.vad_min_silence_ms == 3000
    assert args.no_clipboard is True
    assert args.no_notify is True
    assert args.no_sound is True


def check_audio_levels() -> None:
    audio = np.array([[0], [16384], [-16384], [32767]], dtype=np.int16)
    peak, rms = audio_levels(audio)
    assert 0.99 < peak <= 1.0
    assert rms > 0.0


def check_clipboard_helper() -> None:
    fake_pyperclip = SimpleNamespace(copied=None)

    def copy(value: str) -> None:
        fake_pyperclip.copied = value

    fake_pyperclip.copy = copy
    with patch.dict(sys.modules, {"pyperclip": fake_pyperclip}):
        assert copy_to_clipboard("hello") is True
        assert fake_pyperclip.copied == "hello"
    assert copy_to_clipboard("", enabled=True) is False
    assert copy_to_clipboard("hello", enabled=False) is False


def check_feedback_helpers_do_not_raise() -> None:
    with patch("subprocess.run", side_effect=RuntimeError("boom")):
        assert notify("title", "message") is False
        assert play_sound("start") is False
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=1)):
        assert notify("title", "message") is False
        assert play_sound("start") is False
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)):
        assert notify("title", "message") is True
        assert play_sound("start") is True
    assert notify("title", "message", enabled=False) is False
    assert play_sound("missing", enabled=True) is False


def check_hotkey_trust_preflight() -> None:
    fake_untrusted = SimpleNamespace(AXIsProcessTrusted=lambda: False)
    fake_trusted = SimpleNamespace(AXIsProcessTrusted=lambda: True)
    with patch("platform.system", return_value="Darwin"):
        with patch.dict(sys.modules, {"HIServices": fake_untrusted}):
            assert check_hotkey_trust() is False
        with patch.dict(sys.modules, {"HIServices": fake_trusted}):
            assert check_hotkey_trust() is True
        with patch.dict(sys.modules, {"HIServices": None}):
            assert check_hotkey_trust() is True
    with patch("platform.system", return_value="Linux"):
        assert check_hotkey_trust() is True


def check_single_instance_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / "quickstt.lock"
        first = acquire_single_instance_lock(lock_path)
        assert first is not None
        try:
            second = acquire_single_instance_lock(lock_path)
            assert second is None
        finally:
            first.close()


def main() -> int:
    checks = [
        ("기본 옵션", check_default_options),
        ("커스텀 옵션", check_custom_options),
        ("오디오 레벨 계산", check_audio_levels),
        ("클립보드 헬퍼", check_clipboard_helper),
        ("알림/사운드 헬퍼", check_feedback_helpers_do_not_raise),
        ("핫키 권한 preflight", check_hotkey_trust_preflight),
        ("단일 실행 락", check_single_instance_lock),
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
