# Hungarian(NMS-free) Head 실험 결과 정리

## 배경
BeatFCOS의 기존 anchor 기반 탐지(ClassificationModel/RegressionModel/Anchors/CombinedLoss +
Soft-NMS 후처리)를 대체하기 위해, 고정 개수의 learned query가 (class, interval)을
직접 예측하는 축소판 RT-DETR 스타일 헤드(`head_type="hungarian"`)를 시도함.
목표는 Soft-NMS 같은 후처리 없이 end-to-end로 중복 억제까지 학습되는 구조를 만드는 것.

일반 Hungarian 매칭 대신, beat/downbeat 이벤트의 두 가지 구조적 성질
(1) 곡 시작/끝을 제외하면 빈 공간 없이 조밀함, (2) 항상 시간 순서대로 나타남
을 이용해 O(Q*M) monotonic DP 매칭(OrderedMatcher)으로 단순화함 (수학적으로
이 조건 하에서는 일반 Hungarian과 동일한 결과).

## 시도 흐름과 원인 진단

| 버전 | 수정 내용 | 결과 |
|---|---|---|
| v3 (최초) | positional encoding 없음, DETR 고정 loss weight(bbox×5, giou×2) | query collapse (300개 query → 4개 위치로 뭉침), epoch 2 이후 Joint score 0.001로 완전 붕괴 |
| v4 | + sinusoidal positional encoding, + Kendall et al. 2018 learnable uncertainty weighting(log_var), + log_var clamp[-2,2] | query_embed 자체는 diverse(코사인 유사도 랜덤 초기화 수준)했지만 TransformerDecoder를 통과하면서 여전히 collapse (300개 중 7~10개 위치로 뭉침) → 여전히 Joint 0.001대 |
| v5 | + Pre-LN decoder(`norm_first=True`), + grad clip 완화(hungarian 경로만 0.1→1.0) | collapse 해결. Post-LN + 지나치게 타이트한 grad clip이 decoder 초반 representation collapse의 원인이었음. epoch 14 부근 최고 Joint **0.181** |
| v6 | + FPN 레벨별 실제 시간 좌표 기반 positional encoding + level embedding | P1/P2/P3를 이어붙인 시퀀스 순번으로 위치 인코딩을 계산하던 방식이, 서로 다른 stride를 가진 레벨 간 같은 시간대 토큰을 다른 위치로 취급하는 문제를 가지고 있었음 → 실제 시간 좌표(stride 반영) + 레벨 식별 임베딩으로 수정. 최고 Joint **0.195**(epoch 14), 이후 정체 |

## v6 (최종 run) epoch별 추이

```
Epoch 0~13: 0.12~0.19 사이 변동
Epoch 14:   0.195   ← 최고점
Epoch 15~17: 0.188~0.194
Epoch 18~32: 0.185~0.189 완전 정체 (15 epoch 연속 사실상 그대로)
```

같은 구간에서 학습 loss(CLS 0.486~0.488, GIOU 0.511~0.515, BBOX(L1) -0.184 고정)도 완전히 멈춤.
`ReduceLROnPlateau`가 여러 차례 LR을 낮췄을 것으로 추정되는 시점 이후에도 더 이상 움직이지 않음.

## 핵심 관찰: loss는 계속 개선되는데 F-measure는 정체/역행

여러 시점(v5, v6)에서 공통으로 관찰된 패턴: train loss(CLS/BBOX/GIOU)는 계속 매끄럽게
개선되는데 eval F-measure는 어느 시점부터 정체되거나 오히려 소폭 하락함. 이는:
- loss(L1/GIoU)는 "평균적으로 얼마나 가까운가"를 보는 연속 지표
- F-measure는 "허용 오차(±70ms) 안에 들어왔는가"를 보는 이진(hit/miss) 지표

이 둘의 간극 때문에, 모델이 평균적으로는 계속 좋아져도 F-measure가 요구하는 정밀도
문턱을 못 넘으면 점수가 안 오르는 현상으로 해석됨.

## 결론

1. 세 차례에 걸쳐 구조적 원인을 찾아 순차적으로 해결함: query collapse(positional
   encoding 부재) → decoder representation collapse(Post-LN + 과도한 grad clip) →
   FPN 레벨 간 위치 정렬 불일치. 그때마다 확실한 개선이 있었음(0.001 → 0.181 → 0.195).
2. 그럼에도 FCOS 베이스라인(~0.8)과는 여전히 큰 격차가 있고, 15 epoch 연속 정체를
   보면 이 구조/설정의 한계치에 도달한 것으로 판단됨.

## 관련 파일
- `beatfcos/hungarian_head.py` — SetPredictionHead, OrderedMatcher, SetCriterion, 위치 인코딩
- `beatfcos/model_module.py` — `head_type="hungarian"` 분기, `_forward_hungarian()`
- `train.py` — `--head_type`, `--num_queries`, `--decoder_layers` CLI 옵션, grad clip 분기
- `beatfcos/beat_eval.py` — hungarian 경로 eval 출력
- `diag_checkpoint_collapse.py` — 체크포인트의 query 위치 다양성/collapse 진단 스크립트

---

## FCOS+DSA 백본 (baseline 재확인) 실험 현황

hungarian 결과가 안 좋아서, "DSA+FPN 백본 교체만 하고 기존 FCOS 헤드는 그대로 둔"
조합이 잘 도는지 별도로 재확인 중.

### 현재 학습 환경/설정
- `head_type`: `fcos` (기본값, 안 건드림 — anchor 기반 ClassificationModel/RegressionModel/
  Anchors/CombinedLoss + Soft-NMS 그대로)
- 백본: DSA(BeatTransformerEncoder) + FPN (dmodel=128, nhead=2, d_hid=512, nlayers=9,
  attn_len=5, dropout=0.1 — 전부 기본값)
- 데이터셋 4개: ballroom, beatles, rwc_popular, hainsworth (train.py에 gtzan/smc는
  CLI 인자 자체가 없어서 연결 안 돼 있음)
- `--lr`: 기본값 1e-3
- `--patience`: **10** (기본값 3에서 변경 — 이전 run이 patience=3일 때 15 epoch
  넘게 완전히 정체돼서, LR이 너무 일찍/자주 깎여서 그런 것 아닌지 확인하려고 변경)
- `--downbeat_weight`: 기본값 0.6, **이번부터 실제로 loss에 반영됨** (아래 참고)
- `--epochs`: 100 (기본값)
- `--batch_size`: 32 (기본값)

### 이번에 같이 반영된 코드 수정: downbeat_weight 연결
- 문제: `downbeat_weight` 인자가 모델에 저장만 되고 `losses.py`의 `FocalLoss`
  어디에도 실제로 안 쓰이고 있었음 (죽은 파라미터). Beat는 downbeat보다 인스턴스가
  ~4배 많은데 loss는 두 클래스에 동일한 `alpha=0.25`를 써서, downbeat 채널이
  상대적으로 약한 학습 신호만 받음 — 실제로 이전 run에서 Beat F(0.85~0.88, 안정적)
  대비 Downbeat F(0.30~0.37, epoch마다 크게 출렁임)가 훨씬 불안정하게 관찰됨.
- 수정: `FocalLoss.__init__(downbeat_weight=0.5)`가 "중립"(=기존 동작과 100%
  동일, no-op)이 되도록 하고, 0.5보다 크면 downbeat 채널 loss를 그만큼 키우고
  beat 채널은 그만큼 줄이게 구현 (가중치 합은 항상 2.0으로 고정). 프로젝트 기본값
  0.6이 이제 실제로 [downbeat 1.2, beat 0.8] 가중치로 적용됨.
- 검증: `downbeat_weight=0.5`일 때 결과가 수정 전 FocalLoss를 수동으로 재현한
  값과 정확히 일치함을 확인 (회귀 없음).

### 이전(patience=3, downbeat_weight 미반영) run 결과 요약
- epoch 1부터 Beat F 0.8 돌파, epoch 9에서 Joint 최고 **0.633**
- 이후 20+ epoch 동안 그 최고점을 못 넘고 0.57~0.63 사이에서 정체
- 뜯어보니 Beat F는 안정적(0.85~0.88)이고 Downbeat F만 불안정(0.30~0.37)해서
  Joint 전체가 그 노이즈에 흔들리는 것으로 확인 → 위 downbeat_weight 수정의 근거

### 지금 이 run의 목적
1. `patience=10`이 "LR이 너무 일찍 깎여서 생기는 조기 정체"를 완화하는지
2. `downbeat_weight` 수정이 Downbeat F의 epoch간 변동폭을 줄이고 Joint 최고점을
   0.633보다 끌어올리는지

진행 상황은 이 섹션에 계속 업데이트 예정.