"""Phase 2 사전 검증 (마이크 입력 없이 자동화 가능한 부분만).

실제 마이크 녹음 동작은 ``main_phase2.py`` 에서 사용자가 확인한다.

검증 항목:
    1. Recorder.save_wav 가 16kHz mono int16 wav 파일을 올바르게 만들고
       표준 ``wave`` 모듈로 다시 읽었을 때 샘플레이트/샘플수가 일치하는지.
    2. sounddevice 가 입력 디바이스(마이크)를 인식하는지.
"""
from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from recorder import Recorder


def _make_dummy_audio(sample_rate: int = 16000, duration_sec: float = 1.0) -> np.ndarray:
    n = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    waveform = np.sin(2 * np.pi * 440.0 * t) * 0.3
    pcm = (waveform * 32767).astype(np.int16)
    return pcm.reshape(-1, 1)


def check_save_wav() -> None:
    sample_rate = 16000
    audio = _make_dummy_audio(sample_rate=sample_rate, duration_sec=1.0)
    n_expected = audio.shape[0]

    rec = Recorder(sample_rate=sample_rate, channels=1, dtype="int16")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        rec.save_wav(audio, wav_path)

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

        assert sr == sample_rate, f"샘플레이트 불일치: {sr}"
        assert n_frames == n_expected, f"샘플 수 불일치: {n_frames} vs {n_expected}"
        assert n_channels == 1, f"채널 수 불일치: {n_channels}"
        assert sample_width == 2, f"샘플 폭 불일치: {sample_width} (int16=2)"
    finally:
        wav_path.unlink(missing_ok=True)


def check_input_device() -> None:
    import sounddevice as sd

    devices = sd.query_devices()
    has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
    assert has_input, "입력 디바이스가 감지되지 않았습니다 (마이크 미연결?)"


def main() -> int:
    print("[1/2] Recorder.save_wav 포맷 검증")
    try:
        check_save_wav()
        print("  ✓ 통과")
    except AssertionError as exc:
        print(f"  ✗ 실패: {exc}")
        return 1
    except Exception as exc:
        print(f"  ✗ 예외: {exc!r}")
        return 1

    print("[2/2] sounddevice 입력 디바이스 검증")
    try:
        check_input_device()
        print("  ✓ 통과")
    except AssertionError as exc:
        print(f"  ✗ 실패: {exc}")
        return 1
    except Exception as exc:
        print(f"  ✗ 예외 (sounddevice 설치/PortAudio 문제 가능): {exc!r}")
        return 1

    print()
    print("✓ Phase 2 사전 검증 통과")
    print("  실제 녹음 + 변환은 `python main_phase2.py` 로 확인하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
