# BeatFCOS Architecture Notes

---

## 왜 바꿨는가

기존 BeatFCOS의 FPN + 3 Heads + Soft-NMS 구조는 유지하되,
**Backbone만 dsTCN → DSA(Beat Transformer)로 교체**하는 것이 목표.

기존 dsTCN은 stride=2가 각 블록마다 있어서 자연스럽게 multi-scale feature(C4, C5)가 생겼고,
이걸 FPN에 바로 넣을 수 있었다.

DSA는 모든 레이어가 shape `(B, T, d)`를 유지한다 (stride 없음).
그래서 intermediate output을 tap해도 해상도 차이가 없고,
**FPN에 넣으려면 명시적 downsampling(rho)이 반드시 필요**하다.

---

## 기존 구조 요약

```
Mel-spec (B, T, 128)
    → dsTCN (C1~C8 생성, stride=2 per block)
    → C4, C5만 FPN에 전달   ← 마지막 2 레벨만 사용
    → FPN: P4(T), P5(T/2) + stride conv으로 P6, P7, P8 생성 (가짜 레벨)
    → 3 Heads → Soft-NMS → beat/downbeat 위치
```

P6/P7/P8은 C5에서 stride convolution만으로 만든 것 — semantic 차이 없는 가짜 multi-scale.

---

## 새 구조 요약

```
Mel-spec (B, T, 128)
    → BeatTransformerEncoder (Conv2d frontend + DSA 9 layers)
    → C1(layer 2), C2(layer 5), C3(layer 8) tap   ← 3그룹×3레이어, 각 (B,T,d)
    → rho downsampling: C1→(T), C2→(T/2), C3→(T/4)
    → FPN: P1(T), P2(T/2), P3(T/4)   ← 진짜 multi-scale 3레벨
    → 3 Heads → Soft-NMS → beat/downbeat 위치
```

---

## 핵심 결정들

**왜 9 layers?**
Beat Transformer 원 논문 설계. dilation = 2^i (i=0~8) → 최대 ~25초 컨텍스트 커버.

**왜 C3 (3-3-3)?**
9 layers를 3등분하면 딱 나눠떨어짐. C4는 9를 4로 나눠야 해서 비균등 — 일단 C3으로 시작.

**왜 rho가 필요한가?**
DSA는 shape 불변이라 C1/C2/C3 모두 T 해상도. FPN 입력은 서로 다른 해상도여야 하므로 명시적으로 T → T/2 → T/4 만들어줌.

---

## 구현 현황

| 컴포넌트 | 파일 | 상태 |
|----------|------|------|
| PyramidFeatures (P1/P2/P3, P6~P8 제거) | `model_module.py` | ✅ 완료 |
| DilatedTransformerLayer 복사 | `beatfcos/DilatedTransformerLayer.py` | ✅ 완료 |
| BeatTransformerEncoder | `beatfcos/beat_transformer_encoder.py` | ✅ 완료 |
| rho downsampling (rho1/2/3) + BeatFCOS.__init__ / .forward 교체 | `model_module.py` | ✅ 완료 |
| create_beatfcos_model 정리 (dsTCN 관련 제거) | `model_module.py` | ✅ 완료 |
| train.py 인수 정리 | `train.py` | ✅ 완료 |
| dataloader.py mel-spec 입력 | `beatfcos/dataloader.py` | ✅ 완료 |
| losses.py | `beatfcos/losses.py` | ✅ 수정 없음 (절대 유지) |

### BeatTransformerEncoder 역할
- Conv2d frontend: mel-spec `(B, T, 128)` → `(B, T, d)` 차원 변환
- DSA 9 layers: layer 2, 5, 8에서 tap → C1, C2, C3 반환
- rho downsampling은 여기서 하지 않음 (BeatFCOS에서 처리)

### dataloader mel-spec 변환 (`beatfcos/dataloader.py`)
- `load_data()` 에서 waveform 로드 후 `torchaudio.transforms.MelSpectrogram`으로 즉석 변환
- 파라미터: `n_fft=2048, hop_length=512, n_mels=128, f_min=30, f_max=11000`
- log 압축: `torch.log1p()` 적용
- 출력: `(T, 128)` → collate 후 `(B, T, 128)`
- `__getitem__` spectral 브랜치: 길이 초과 시 `audio[:length, :]` crop

### train.py 정리
- `audio_downsampling_factor` 기본값: 128 → 512 (hop_length 기준)
- `pretrained` 기본값: True → False (DSA는 처음부터 학습)
- `spectral=True` 고정 (train/val 모두)
- DSA encoder 인수 추가: `--dmodel`, `--nhead`, `--d_hid`, `--nlayers`, `--attn_len`, `--dropout`
- `BeatFCOS.__init__`에서 encoder_keys 필터링으로 dsTCN 전용 인수가 BeatTransformerEncoder에 넘어가지 않도록 처리

### create_beatfcos_model 정리
- 기존: dsTCN/tcn2019 pretrained weight 로딩 + freeze 로직 존재
- 변경: DSA는 처음부터 학습 → pretrained 블록 전체 제거, model 생성 후 바로 return

### rho downsampling 역할 (`BeatFCOS.__init__` 안에 구현)
- DSA 출력 C1/C2/C3은 모두 `(B, T, d)` — 해상도 동일
- transpose로 `(B, d, T)` 변환 후 rho 적용
- rho1: Identity → `(B, d, T)`
- rho2: Conv1d stride=2 → `(B, d, T/2)`
- rho3: Conv1d stride=2 × 2 → `(B, d, T/4)`
- 이후 FPN `[C1, C2, C3]` 순으로 전달

---

## 버그 수정 이력 (첫 실행 시 발견)

### 1. `dataloader.py` — spectral 브랜치 target_length 오류
- **증상**: Training OOM (32+ GiB 누적) → Eval 시 14+ GiB 추가 할당 시도
- **원인**: `spectral=True`일 때 `self.target_length = length`로 설정 — `length`는 오디오 샘플 수(153600)이지 mel frame 수(300)가 아님. T=153600 mel frame으로 학습되어 메모리 폭발.
- **수정**: `if/else` 제거하고 항상 `self.target_length = int(self.length / self.audio_downsampling_factor)` 사용

### 2. `model_module.py` — base_level_image_shape 타입 오류
- **증상**: `TypeError: 'int' object is not subscriptable` at anchors.py:37
- **원인**: `base_level_image_shape = C1.shape[-1]` → int를 넘김. `anchors.py`에서 `base_image_shape[2:]` 처리하므로 shape tuple이어야 함.
- **수정**: `base_level_image_shape = C1.shape` (torch.Size 그대로 전달)

### 3. `anchors.py` — pyramid_levels 5개 vs FPN 3개 불일치
- **증상**: `IndexError: shape of mask [7936] does not match indexed tensor [7168, 2]`
- **원인**: 기존 코드 `pyramid_levels = [0, 1, 2, 3, 4]` → anchor 7936개 생성 (eval T=4096 기준: 4096+2048+1024+512+256). 하지만 FPN은 P1/P2/P3만 출력 → regression head 출력은 7168개 (4096+2048+1024)
- **수정**: `self.pyramid_levels = [0, 1, 2]` → strides 자동으로 [1, 2, 4]

---

## 실험 계획

- **C3**: 9 layers → 3-3-3 → tap at layer 2, 5, 8 → P1/P2/P3
- **C4**: 9 layers → 비균등 4구간 → P1/P2/P3/P4 (ablation)
