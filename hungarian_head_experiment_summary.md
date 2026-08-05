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

---

## FPN Ablation 실험 (안재훈 박사님 제안, 2026-07)

박사님이 epoch 79 비교표를 보고 제안: FPN의 lower layer(P1) 제거 + classification/regression
head 통합 구조, 그리고 FPN 자체(cross-scale fusion)를 완전히 제거하고 head를 encoder
마지막 레이어에 직결하는 구조. 두 실험 모두 `head_type` 분기로 구현 (`model_module.py`,
`losses.py`, `anchors.py`, `train.py`).

### 실험군 (validation_fold 0, epochs 100, patience 10, nhead 8, 6개 데이터셋 공통)

| head_type | GPU | 구조 | 상태 |
|---|---|---|---|
| `fcos` (baseline) | 0 | 원래 구조 (P1/P2/P3 + FPN fusion + 분리된 head) | **완료** (epoch 99) |
| `fcos_lite` | 1 | P1 제거(P2/P3만 사용, FPN fusion은 유지), classification+regression head 통합 | 진행 중 (epoch 28~) |
| `fcos_no_fpn` | 2 | FPN cross-scale fusion 자체를 끔(P1만, 단일 레벨), head는 분리 구조 유지 | 진행 중 (epoch 20~) |

### 학습 중 자체 평가 최고점 (train.py 로그 기준, 세 실험 모두 100 epoch 완료)

| head_type | 체크포인트(최고점) | Beat score | Downbeat score | Joint score |
|---|---|---|---|---|
| `fcos` (baseline) | retinanet_79.pt | 0.917 | 0.816 | 0.868 |
| `fcos_lite` | retinanet_92.pt | 0.925 | 0.838 | 0.885 |
| `fcos_no_fpn` | retinanet_35.pt | 0.858 | 0.520 | 0.690 |

### 정식 평가 (evaluate_all_datasets.py, score_threshold=downbeat_score_threshold=0.20,
데이터셋별 val fold 개별 평가 — train.py의 macro-average와 별개로 재확인)

**fcos (baseline)**

| dataset | Beat F | Beat CMLt | Beat AMLt | Downbeat F | Downbeat CMLt | Downbeat AMLt | Joint F |
|---|---|---|---|---|---|---|---|
| ballroom | 0.923 | 0.831 | 0.855 | 0.819 | 0.795 | 0.834 | 0.871 |
| beatles | 0.958 | 0.923 | 0.923 | 0.896 | 0.833 | 0.863 | 0.927 |
| hainsworth | 0.866 | 0.777 | 0.830 | 0.574 | 0.540 | 0.683 | 0.720 |
| rwc_popular | 0.921 | 0.831 | 0.840 | 0.880 | 0.897 | 0.897 | 0.900 |
| carnatic | 0.886 | 0.765 | 0.779 | 0.837 | 0.945 | 0.945 | 0.861 |
| harmonix | 0.945 | 0.897 | 0.904 | 0.916 | 0.932 | 0.936 | 0.930 |
| gtzan(held-out) | 0.834 | 0.696 | 0.776 | 0.606 | 0.539 | 0.711 | 0.720 |
| smc(held-out) | 0.495 | 0.293 | 0.395 | 0.279 | 0.052 | 0.214 | 0.387 |

**fcos_lite (P1 제거 + head 통합)**

| dataset | Beat F | Beat CMLt | Beat AMLt | Downbeat F | Downbeat CMLt | Downbeat AMLt | Joint F |
|---|---|---|---|---|---|---|---|
| ballroom | 0.940 | 0.865 | 0.888 | 0.865 | 0.845 | 0.869 | 0.902 |
| beatles | 0.946 | 0.872 | 0.899 | 0.901 | 0.814 | 0.888 | 0.923 |
| hainsworth | 0.888 | 0.806 | 0.835 | 0.615 | 0.532 | 0.625 | 0.751 |
| rwc_popular | 0.919 | 0.847 | 0.868 | 0.920 | 0.928 | 0.928 | 0.920 |
| carnatic | 0.909 | 0.793 | 0.804 | 0.851 | 0.948 | 0.951 | 0.880 |
| harmonix | 0.949 | 0.899 | 0.905 | 0.916 | 0.929 | 0.932 | 0.932 |
| gtzan(held-out) | 0.839 | 0.691 | 0.773 | 0.614 | 0.539 | 0.702 | 0.726 |
| smc(held-out) | 0.491 | 0.234 | 0.359 | 0.288 | 0.046 | 0.252 | 0.389 |

**fcos_no_fpn (FPN fusion 제거)**

| dataset | Beat F | Beat CMLt | Beat AMLt | Downbeat F | Downbeat CMLt | Downbeat AMLt | Joint F |
|---|---|---|---|---|---|---|---|
| ballroom | 0.888 | 0.747 | 0.778 | 0.579 | 0.321 | 0.390 | 0.734 |
| beatles | 0.879 | 0.743 | 0.765 | 0.660 | 0.372 | 0.407 | 0.769 |
| hainsworth | 0.847 | 0.733 | 0.761 | 0.364 | 0.148 | 0.225 | 0.605 |
| rwc_popular | 0.841 | 0.700 | 0.738 | 0.637 | 0.430 | 0.462 | 0.739 |
| carnatic | 0.767 | 0.493 | 0.631 | 0.401 | 0.165 | 0.240 | 0.584 |
| harmonix | 0.883 | 0.770 | 0.793 | 0.537 | 0.233 | 0.370 | 0.710 |
| gtzan(held-out) | 0.817 | 0.646 | 0.703 | 0.454 | 0.207 | 0.322 | 0.636 |
| smc(held-out) | 0.500 | 0.235 | 0.332 | 0.308 | 0.078 | 0.159 | 0.404 |

### 결론
- fcos_lite는 거의 전 데이터셋에서 baseline과 동등하거나 소폭 앞섬 → P1 제거+head 통합은
  성능 저하 요인이 아님.
- fcos_no_fpn은 전 데이터셋에서 확실히 큰 폭으로 하락, 특히 Downbeat CMLt(정확한 metric
  level 일치 비율)가 가장 크게 무너짐(예: carnatic 0.945→0.165, ballroom 0.795→0.321) →
  FPN의 cross-scale fusion이 downbeat의 정확한 위상/주기 판단에 핵심적인 역할을 한다는
  근거로 해석됨.
- carnatic/harmonix(우리가 vanilla BT 대비 이기던 데이터셋)에서 fcos_no_fpn의 낙폭이
  특히 큼 — FPN fusion이 그 우위의 구조적 원인 중 하나일 가능성을 뒷받침.

### 미해결
Vanilla Beat Transformer(우리 데이터로 from-scratch 재학습, `01_Beat_Transformer/Fold_0/
model/trf_param_008.pt`)의 동일 breakdown(Beat/Downbeat F·CMLt·AMLt, 데이터셋별)은 아직
계산된 적 없음 — 이전에 박사님께 보낸 "epoch 79 비교표"는 어디에도 파일로 안 남아있어서
찾을 수 없었음. 필요하면 vanilla BT 전용 평가 스크립트를 새로 만들어서 돌려야 함.

### 다음 할 일
GPU 0, 1, 2 세 실험 모두 완료 및 정식 평가 완료. 남은 건 vanilla BT와의 직접 비교
(위 "미해결" 참고) 및 박사님께 이 FPN ablation 결과 공유.