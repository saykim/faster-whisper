# QuickSTT 사용 방법

이 문서는 현재 구현된 Phase 1, Phase 2 기준 실행 방법을 정리한다.

개발 환경은 `conda activate web310`을 기준으로 한다.

---

## 1. 개발 환경 활성화

프로젝트 폴더로 이동한 뒤 conda 환경을 활성화한다.

```bash
cd /Users/kimsy/DataScience/01_Projects/Web_Applications/faster-whisper
conda activate web310
python --version
```

정상이라면 `Python 3.10.x`가 출력된다.

---

## 2. 의존성 설치

```bash
pip install -r requirements.txt
```

설치 확인:

```bash
pip list | grep -E 'faster-whisper|ctranslate2|sounddevice|numpy'
```

---

## 3. 모델 사전 다운로드

모델은 실행 시 자동 다운로드되지만, 미리 받아두면 첫 실행 지연을 줄일 수 있다.

### small 모델 다운로드

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('done')"
```

### medium 모델 다운로드

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8'); print('done')"
```

다운로드 확인:

```bash
du -sh ~/.cache/huggingface/hub/models--Systran--faster-whisper-small/
du -sh ~/.cache/huggingface/hub/models--Systran--faster-whisper-medium/
```

---

## 4. Phase 1: 샘플 wav 변환 테스트

macOS `say` 명령으로 5초 내외 한국어 샘플 wav를 생성한 뒤 변환한다.

```bash
python test_phase1.py
```

small 모델로 빠르게 검증:

```bash
python test_phase1.py --model small
```

샘플 wav 파일을 삭제하지 않고 남기기:

```bash
python test_phase1.py --model small --keep-sample
```

성공하면 다음과 비슷한 결과가 출력된다.

```text
✓ Phase 1 검증 통과
```

---

## 5. Phase 2: 마이크 녹음 후 변환

마이크로 직접 녹음하고, 녹음된 wav를 한국어 텍스트로 변환한다.

```bash
python main_phase2.py --model small
```

실행 흐름:

1. 모델 로드
2. `Enter 를 눌러 녹음 시작...` 문구가 나오면 Enter
3. 한국어로 말하기
4. 다시 Enter를 눌러 녹음 종료
5. 변환 결과가 콘솔에 출력됨

예상 출력:

```text
==> 모델 로드 중: small (ko)
    ✓ 준비 완료

Enter 를 눌러 녹음 시작...
    ● 녹음 중... (Enter 로 종료)
    ✓ 녹음 종료 (..., peak=..., rms=...)
==> 변환 중... (vad_filter=True, min_silence=2000ms)
    ✓ 완료 (..., segments=..., detected=ko ...)

--- 변환 결과 ---
...
------------------
```

---

## 6. 조정 가능한 실행 옵션

현재 `main_phase2.py`에서 직접 바꿀 수 있는 옵션은 다음과 같다.

전체 옵션 확인:

```bash
python main_phase2.py --help
```

### 6.1 모델 크기 변경

```bash
python main_phase2.py --model small
```

가능한 값:

```text
tiny, base, small, medium, large-v3
```

추천:

- 빠른 테스트: `small`
- 정확도 우선: `medium`
- 최고 정확도: `large-v3` (느리고 용량 큼)

---

### 6.2 VAD 끄기

```bash
python main_phase2.py --model small --no-vad
```

VAD는 `Voice Activity Detection`의 약자로, 음성 구간만 감지해서 변환하는 기능이다.

- 기본값: `vad_filter=True`
- `--no-vad` 사용 시: `vad_filter=False`

짧은 테스트나 조용한 목소리에서 인식이 안 되면 `--no-vad`로 재시도한다.

---

### 6.3 VAD 무음 기준 변경

```bash
python main_phase2.py --model small --vad-min-silence-ms 3000
```

기본값:

```text
2000ms
```

의미:

- `1000`: 1초 이상 무음이면 구간을 나눔
- `2000`: 현재 기본값
- `3000`: 3초 이상 무음이면 구간을 나눔
- `5000`: 더 긴 문단 단위로 처리

계속 켜놓고 받아쓰기 용도로 쓸 경우에는 VAD를 켜고, `2000~3000ms` 정도를 권장한다.

---

### 6.4 녹음 wav 파일 보존

```bash
python main_phase2.py --model small --keep-wav
```

기본은 변환 후 임시 wav 파일을 삭제한다.

`--keep-wav`를 사용하면 파일 경로가 출력되며, 실제 녹음 품질을 직접 들어볼 수 있다.

---

## 7. 자주 쓰는 실행 예시

### 빠른 테스트

```bash
python main_phase2.py --model small
```

### VAD 없이 테스트

```bash
python main_phase2.py --model small --no-vad
```

### 3초 무음 기준으로 테스트

```bash
python main_phase2.py --model small --vad-min-silence-ms 3000
```

### wav 파일을 남겨서 디버깅

```bash
python main_phase2.py --model small --keep-wav
```

### 여러 옵션 조합

```bash
python main_phase2.py --model small --vad-min-silence-ms 3000 --keep-wav
```

---

## 8. Phase 3: 글로벌 단축키 + 장시간 자동 청크 변환

Phase 3부터는 터미널에서 한 번 실행해 둔 뒤, 글로벌 단축키로 장시간 녹음 세션을 시작/종료한다.

기본 모드는 `session`이며, 앱이 5분 단위로 자동 저장/변환한다.

```bash
python main.py
```

기본 설정:

- 모델: `small`
- 단축키: `Cmd+Shift+R`
- 모드: `session`
- 청크 길이: `300초`
- VAD: 켜짐
- VAD 무음 기준: `2000ms`
- 출력 폴더: `~/QuickSTT/recordings`

실행 후 다음 문구가 보이면 대기 상태다.

```text
대기 중... 단축키: <cmd>+<shift>+r
모드: session
청크 길이: 300s, 출력 폴더: /Users/.../QuickSTT/recordings
종료하려면 Ctrl+C
```

사용 흐름:

1. `Cmd+Shift+R` 입력
2. 장시간 세션 시작
3. 앱이 5분마다 wav 청크를 자동 저장하고 백그라운드에서 변환
4. 변환 결과는 `transcript.txt`에 자동 누적
5. 다시 `Cmd+Shift+R` 입력
6. 현재 청크를 마감하고 남은 변환을 완료한 뒤 세션 종료

결과 파일 예시:

```text
~/QuickSTT/recordings/
└── 2026-04-29_22-30-00/
    ├── transcript.txt
    ├── chunk_0001.wav  # --keep-wav 사용 시 보존
    └── chunk_0002.wav
```

기본 설정에서는 변환이 끝난 wav 청크를 삭제한다. 원본 wav를 남기려면 `--keep-wav`를 사용한다.

---

## 9. Phase 3 실행 옵션

### 9.1 모델 변경

```bash
python main.py --model medium
```

빠른 반응이 필요하면 `small`, 정확도가 더 중요하면 `medium`을 사용한다.

---

### 9.2 단축키 변경

```bash
python main.py --hotkey '<cmd>+<shift>+space'
```

기본값:

```text
<cmd>+<shift>+r
```

`pynput` 형식을 사용한다.

예시:

```bash
python main.py --hotkey '<cmd>+<shift>+r'
python main.py --hotkey '<cmd>+<alt>+r'
python main.py --hotkey '<ctrl>+<shift>+r'
```

---

### 9.3 VAD 옵션 변경

VAD 끄기:

```bash
python main.py --no-vad
```

VAD 무음 기준을 3초로 변경:

```bash
python main.py --vad-min-silence-ms 3000
```

---

### 9.4 녹음 wav 보존

```bash
python main.py --keep-wav
```

실제 녹음 파일을 확인해야 할 때 사용한다.

---

### 9.5 장시간 세션 청크 길이 변경

기본값은 300초(5분)이다.

```bash
python main.py --chunk-seconds 300
```

디버깅할 때는 짧게 줄이면 빠르게 확인할 수 있다.

```bash
python main.py --chunk-seconds 30 --keep-wav
```

---

### 9.6 출력 폴더 변경

```bash
python main.py --output-dir ~/QuickSTT/recordings
```

각 실행마다 `YYYY-MM-DD_HH-MM-SS` 세션 폴더가 만들어진다.

---

### 9.7 짧은 녹음 모드 사용

기존처럼 단축키 한 번으로 녹음을 시작하고, 다시 눌러 한 번만 변환하려면 `once` 모드를 사용한다.

```bash
python main.py --mode once --model small
```

---

### 9.8 핫키 중복 입력 방지 시간

```bash
python main.py --hotkey-debounce-ms 1500
```

기본값:

```text
1000ms
```

`Cmd+Shift+R`을 한 번 눌렀는데 녹음이 바로 종료되거나 `변환 중입니다`가 즉시 출력되면 키 입력이 중복 감지된 것이다.

그럴 때는 `1500~2000ms`로 늘려서 실행한다.

---

### 9.9 추천 실행 명령

현재 가장 추천하는 실행 명령:

```bash
python main.py --model small --chunk-seconds 300 --vad-min-silence-ms 2000
```

정확도가 부족하면:

```bash
python main.py --model medium --chunk-seconds 300 --vad-min-silence-ms 2000
```

단축키가 중복으로 들어가는 환경이면:

```bash
python main.py --model small --chunk-seconds 300 --vad-min-silence-ms 2000 --hotkey-debounce-ms 1500
```

짧은 청크로 동작 확인:

```bash
python main.py --model small --chunk-seconds 30 --keep-wav
```

---

## 10. macOS 마이크/단축키 권한

처음 마이크 녹음을 실행하면 macOS가 마이크 권한을 요청할 수 있다.

변환 결과가 계속 비어 있거나 `peak=0.000`에 가깝다면 마이크 입력이 들어오지 않는 것이다.

확인 경로:

```text
시스템 설정 > 개인정보 보호 및 보안 > 마이크
```

현재 실행 중인 앱에 권한을 부여한다.

- Cursor 터미널에서 실행하면 Cursor 권한 필요
- iTerm에서 실행하면 iTerm 권한 필요
- Terminal.app에서 실행하면 Terminal 권한 필요

권한을 변경한 뒤에는 해당 앱을 재시작하는 것이 좋다.

글로벌 단축키가 반응하지 않으면 다음 권한도 확인한다.

```text
시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용
```

현재 실행 중인 앱(Cursor, iTerm, Terminal.app 등)에 권한을 부여한다.

일부 macOS 환경에서는 다음 경로도 필요할 수 있다.

```text
시스템 설정 > 개인정보 보호 및 보안 > 입력 모니터링
```

