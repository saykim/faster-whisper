"""Phase 1 검증 스크립트.

macOS의 ``say`` + ``afconvert``를 이용해 5초 분량의 한국어 샘플 wav를
즉석 생성한 뒤, ``transcriber.Transcriber``가 정상 동작하는지 확인한다.

사용 예:
    python test_phase1.py                  # medium 모델(기본, PRD 사양, 첫 실행 시 ~1.5GB)
    python test_phase1.py --model small    # 빠른 검증용(~480MB)
    python test_phase1.py --keep-sample    # 생성된 sample_ko.wav 보존
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from transcriber import Transcriber


SAMPLE_TEXT = "안녕하세요. 이것은 음성 인식 테스트입니다. 잘 들리시나요?"
EXPECTED_KEYWORDS = ("안녕", "음성", "테스트")


def generate_sample_wav(out_path: Path, text: str = SAMPLE_TEXT) -> None:
    """macOS ``say``로 한국어 음성을 만든 뒤 16kHz mono PCM wav로 변환."""
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        raise RuntimeError("macOS 환경에서만 동작합니다 (say/afconvert 필요).")

    with tempfile.TemporaryDirectory() as tmp:
        aiff_path = Path(tmp) / "sample.aiff"
        subprocess.run(
            ["say", "-v", "Yuna", "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            [
                "afconvert",
                "-f", "WAVE",
                "-d", "LEI16@16000",
                "-c", "1",
                str(aiff_path),
                str(out_path),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 transcriber 검증")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper 모델 크기 (기본 medium, PRD 사양)",
    )
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="생성한 sample_ko.wav 파일을 보존",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    sample_path = project_dir / "sample_ko.wav"

    print(f"[1/3] 샘플 wav 생성 중 -> {sample_path.name}")
    try:
        generate_sample_wav(sample_path)
    except Exception as exc:
        print(f"  ✗ 샘플 생성 실패: {exc}")
        return 1
    size_kb = sample_path.stat().st_size / 1024
    print(f"  ✓ 생성 완료 ({size_kb:.1f} KiB)")

    print(f"[2/3] 모델 로드: {args.model} (최초 1회 다운로드 발생)")
    t0 = time.perf_counter()
    transcriber = Transcriber(model_size=args.model, language="ko")
    transcriber.load()
    print(f"  ✓ 로드 완료 ({time.perf_counter() - t0:.1f}s)")

    print("[3/3] 변환 실행 중...")
    t0 = time.perf_counter()
    text = transcriber.transcribe(sample_path)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ 변환 완료 ({elapsed:.2f}s)")
    print()
    print(f"  기대 원문 : {SAMPLE_TEXT}")
    print(f"  변환 결과 : {text}")
    print()

    if not args.keep_sample:
        sample_path.unlink(missing_ok=True)

    if not text:
        print("✗ 변환 결과가 비어 있습니다.")
        return 1

    matched = [kw for kw in EXPECTED_KEYWORDS if kw in text]
    if not matched:
        print(f"✗ 기대 키워드 {list(EXPECTED_KEYWORDS)} 가 하나도 포함되지 않았습니다.")
        return 1

    print(f"✓ Phase 1 검증 통과 (매칭 키워드: {matched})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
