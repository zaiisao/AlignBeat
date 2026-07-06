"""
BeatFCOS를 위한 축소판 RT-DETR 스타일 set-prediction 헤드.

[무엇] model_module.py의 조밀한 anchor 기반 파이프라인(ClassificationModel /
RegressionModel / Anchors / CombinedLoss)을 대체하는, 고정 개수의 learned
query가 (class, interval) 쌍을 직접 예측하는 헤드. 예측 개수가 고정되고
중복 억제가 학습 과정에서 end-to-end로 이루어지므로(DETR과 동일한 원리),
추론 시 NMS/Soft-NMS가 필요 없다.

[왜] Soft-NMS 같은 후처리 자체를 없애야 함. 후처리(NMS)를
없애려면 "조밀한 anchor 예측 + NMS로 중복 제거" 구조를 버리고, 고정 개수
query가 직접 최종 예측을 내놓는 구조로 가야 하기 때문에 이 새 헤드를 추가함.
기존 FCOS 경로(model_module.py의 head_type="fcos", 기본값)는 전혀 건드리지
않았고, 이 헤드는 head_type="hungarian"일 때만 대신 사용됨 — 두 경로를
나중에 나란히 비교할 수 있도록 하기 위함.

[매칭 방식이 왜 일반 Hungarian이 아닌가] 일반 객체 탐지와 달리 beat/downbeat
이벤트는 (1) [곡 시작, 첫 비트] / [마지막 비트, 곡 끝] 두 구간을 제외하면
빈 공간 없이 곡 전체를 조밀하게 덮고, (2) 항상 시간 순서대로 나타난다
(beat_1 < beat_2 < ... < beat_n). 1차원 직선 위의 점을 위치 기반 비용으로
매칭할 때, 이 두 성질이 성립하면 최적 매칭은 항상 순서를 보존하는 매칭과
일치한다(1차원 optimal transport의 고전적 결과, rearrangement inequality).
그래서 일반적인 O(n^3) Hungarian 알고리즘 대신, 예측/정답을 위치 순으로
정렬한 뒤 순서를 지키는(monotonic) 부분집합 매칭을 O(Q*M) DP로 찾는다 —
이 조건 하에서는 결과가 Hungarian과 동일하며, scipy 의존성도 없앤다.

[한계] 이건 완전한 RT-DETR가 아니다 — deformable attention, IoU 기반 query
selection, denoising training 등은 없음. NMS-free/anchor-free 방향이
실제로 동작하는지 먼저 확인해보기 위한 축소판이다.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(length, dim, device):
    """표준 sinusoidal positional encoding (Transformer 원 논문 방식).

    query가 cross-attention으로 FPN memory를 볼 때 "내용(무슨 소리인지)"만
    보고 "시간축 몇 번째 위치인지"는 전혀 알 수 없었던 문제(query 300개가
    거의 같은 위치/클래스로 뭉치는 collapse의 원인으로 추정)를 고치기 위해
    추가함. concat된 memory 시퀀스 전체 길이에 대해 계산하므로, P1/P2/P3의
    어느 레벨에 있든 각 토큰이 고유한 위치 신호를 갖게 된다.
    """
    position = torch.arange(length, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def monotonic_match(cost):
    """정렬된 Q개 후보 중 M개 target을 순서를 지키며 최적으로 배정하는
    O(Q*M) DP (일반 할당 문제의 1차원 monotonic 특수 케이스 — 왜 이게
    여기서 유효한지는 파일 상단 docstring 참고).

    cost: (Q, M) numpy 배열, 행/열이 이미 위치 순으로 정렬돼 있다고 가정.
    Q >= M을 가정함 (남는 행은 그냥 매칭 안 됨으로 처리).

    반환값 (query_indices, target_indices): 둘 다 (M,) int64 배열이고,
    Q/M의 *정렬된* 순서 안에서의 인덱스임. target 하나당 query 하나씩,
    인덱스는 항상 증가하는 순서.
    """
    Q, M = cost.shape
    if M == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    dp = np.full((Q + 1, M + 1), np.inf, dtype=np.float64)
    dp[:, 0] = 0.0
    choice = np.zeros((Q + 1, M + 1), dtype=np.int8)  # 1 = matched query j-1 to target k-1

    for j in range(1, Q + 1):
        max_k = min(j, M)
        for k in range(1, max_k + 1):
            skip_cost = dp[j - 1, k]
            match_cost = dp[j - 1, k - 1] + cost[j - 1, k - 1]
            if match_cost <= skip_cost:
                dp[j, k] = match_cost
                choice[j, k] = 1
            else:
                dp[j, k] = skip_cost

    query_indices = []
    j, k = Q, M
    while k > 0:
        if choice[j, k] == 1:
            query_indices.append(j - 1)
            j -= 1
            k -= 1
        else:
            j -= 1

    query_indices.reverse()
    target_indices = list(range(M))
    return np.asarray(query_indices, dtype=np.int64), np.asarray(target_indices, dtype=np.int64)


def pairwise_giou_1d(a, b):
    """a와 b의 모든 구간 쌍에 대한 1D GIoU(generalized IoU).

    a: (N, 2), b: (M, 2), 둘 다 (l, r) 형태이고 l <= r. (N, M) 반환.
    """
    area_a = (a[:, 1] - a[:, 0]).unsqueeze(1)  # (N, 1)
    area_b = (b[:, 1] - b[:, 0]).unsqueeze(0)  # (1, M)

    inter_l = torch.max(a[:, 0].unsqueeze(1), b[:, 0].unsqueeze(0))
    inter_r = torch.min(a[:, 1].unsqueeze(1), b[:, 1].unsqueeze(0))
    intersection = (inter_r - inter_l).clamp(min=0)

    union = area_a + area_b - intersection
    iou = intersection / union.clamp(min=1e-8)

    enclosing_l = torch.min(a[:, 0].unsqueeze(1), b[:, 0].unsqueeze(0))
    enclosing_r = torch.max(a[:, 1].unsqueeze(1), b[:, 1].unsqueeze(0))
    enclosing = (enclosing_r - enclosing_l).clamp(min=1e-8)

    return iou - (enclosing - union) / enclosing


class SetPredictionHead(nn.Module):
    """
    Input : FPN feature map 리스트 [P1, P2, P3], 각각 (B, feature_size, L_i)
    Output : class_logits (B, num_queries, num_classes + 1), boxes (B, num_queries, 2)
             boxes는 base(P1) 시퀀스 길이 기준 [0, 1]로 정규화된 (l, r).
             마지막 class index는 "no object"(배경) 클래스.

    ClassificationModel/RegressionModel(FCOS)을 대체: FPN을 조밀하게 훑는
    대신, query_embed로 만든 고정 개수(num_queries)의 학습된 query가
    TransformerDecoder를 통해 FPN 전체를 한 번에 cross-attention으로 보고
    각자 (class, l, r)을 직접 예측한다. anchor가 없으므로 NMS도 필요 없다.
    """
    def __init__(self, feature_size=256, num_classes=2, num_queries=300,
                 decoder_layers=3, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(num_queries, feature_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        self.class_head = nn.Linear(feature_size, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 2),  # (center, length), sigmoid로 정규화
        )

    def forward(self, feature_maps):
        # 멀티스케일 FPN 토큰들을 하나의 memory 시퀀스로 이어붙임
        # (deformable attention 없이 단순 concat - 축소판이라 생략함)
        memory = torch.cat([f.transpose(1, 2) for f in feature_maps], dim=1)  # (B, sum(L_i), feature_size)

        # positional encoding 추가: 이게 없으면 query가 "무슨 소리인지"만 보고
        # "시간축 몇 번째 위치인지"를 알 방법이 없어서, 음향적으로 비슷한
        # 여러 위치를 구분 못 하고 query들끼리 서로 뭉치는 원인이 됨.
        pos_embed = sinusoidal_position_encoding(memory.shape[1], memory.shape[2], memory.device)
        memory = memory + pos_embed.unsqueeze(0)

        batch_size = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, feature_size)

        decoded = self.decoder(tgt=queries, memory=memory)  # (B, Q, feature_size)

        class_logits = self.class_head(decoded)  # (B, Q, num_classes + 1)

        center_length = torch.sigmoid(self.bbox_head(decoded))  # (B, Q, 2) in [0, 1]
        center, length = center_length[..., 0], center_length[..., 1]
        l = (center - length / 2).clamp(0, 1)
        r = (center + length / 2).clamp(0, 1)
        boxes = torch.stack([l, r], dim=-1)

        return class_logits, boxes


class OrderedMatcher(nn.Module):
    """순서를 보존하는 매칭기 (자세한 근거는 파일 상단 docstring 참고).
    query를 예측 위치로, target을 실제 위치로 각각 정렬한 뒤, 일반 Hungarian
    대신 O(Q*M) DP로 순서를 지키는 최적 부분집합 매칭을 찾는다.

    class/bbox/giou 비용을 섞는 가중치는 고정 숫자(예: DETR의 5.0, 2.0)를
    갖고 있지 않다 — 이 가중치도 결국 "각 항목이 매칭 결정에 얼마나
    중요한가"를 정하는 문제라, SetCriterion.forward()가 이번 학습에서
    학습 중인 log_var로부터 매 스텝 실제 가중치(exp(-log_var))를 계산해
    forward() 호출 시 넘겨준다. 매칭과 최종 loss가 항상 같은(학습된) 기준을
    쓰게 되고, 어디에도 다른 task에서 가져온 임의의 숫자가 안 남는다.
    """
    @torch.no_grad()
    def forward(self, class_logits, boxes, targets, weight_class=1.0, weight_bbox=1.0, weight_giou=1.0):
        """
        class_logits: (B, Q, num_classes + 1)
        boxes: (B, Q, 2), (l, r)로 [0,1] 정규화됨
        targets: 길이 B인 리스트, 각 원소는 {"labels": (M_b,), "boxes": (M_b, 2)}
        weight_class/bbox/giou: 매칭 비용을 합칠 때 쓸 상대적 가중치.
        SetCriterion이 학습 중인 log_var 기반 값을 넘겨줌 (기본값 1.0은
        독립적으로 테스트할 때를 위한 것일 뿐, 실제 학습에서는 항상 호출
        쪽에서 값을 넘김).
        반환값: (query_idx, target_idx) 인덱스 쌍의 리스트(원본, 정렬 전
        인덱스 기준), 샘플 하나당 한 쌍씩.
        """
        bs, num_queries = class_logits.shape[:2]
        out_prob = class_logits.softmax(-1)  # (B, Q, num_classes+1)

        indices = []
        for b in range(bs):
            tgt_boxes = targets[b]["boxes"]
            tgt_labels = targets[b]["labels"]
            num_targets = tgt_boxes.shape[0]

            if num_targets == 0:
                indices.append((
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64),
                ))
                continue

            pred_boxes = boxes[b]  # (Q, 2)
            pred_centers = pred_boxes.mean(dim=-1)
            tgt_centers = tgt_boxes.mean(dim=-1)

            # 위치 순으로 정렬 - 이게 있어야 monotonic DP 지름길이 성립함
            q_order = torch.argsort(pred_centers)
            t_order = torch.argsort(tgt_centers)  # 원래도 시간 순 정렬돼 있어야 하지만 방어적으로 한 번 더 정렬

            sorted_pred_boxes = pred_boxes[q_order]
            sorted_tgt_boxes = tgt_boxes[t_order]
            sorted_tgt_labels = tgt_labels[t_order]

            cost_bbox = torch.cdist(sorted_pred_boxes, sorted_tgt_boxes, p=1)
            cost_giou = -pairwise_giou_1d(sorted_pred_boxes, sorted_tgt_boxes)
            cost_class = -out_prob[b][q_order][:, sorted_tgt_labels]

            C = (weight_bbox * cost_bbox + weight_giou * cost_giou + weight_class * cost_class)

            # num_queries는 한 곡당 실제 beat/downbeat 개수보다 넉넉하게 잡아뒀음.
            # 혹시라도 target이 query보다 많은 경우, 우선순위가 낮은(정렬 순서상
            # 뒤쪽) target들은 매칭 안 되고 남겨짐.
            num_targets_eff = min(num_queries, num_targets)
            C_np = C[:, :num_targets_eff].cpu().numpy()

            sorted_q_idx, sorted_t_idx = monotonic_match(C_np)

            query_idx = q_order[torch.as_tensor(sorted_q_idx, dtype=torch.int64, device=q_order.device)]
            target_idx = t_order[torch.as_tensor(sorted_t_idx, dtype=torch.int64, device=t_order.device)]

            indices.append((query_idx.cpu(), target_idx.cpu()))

        return indices


class SetCriterion(nn.Module):
    """DETR 스타일 set prediction loss: OrderedMatcher가 정한 매칭 결과를 바탕으로
    분류 loss(CE, "no object" 클래스는 eos_coef로 가중치를 낮춤)와 매칭된 쌍에
    대해서만 계산하는 L1 + GIoU 회귀 loss를 합친다. 기존 CombinedLoss(FCOS,
    losses.py)를 대체하는 역할.

    분류/bbox/GIoU 세 loss를 합칠 때 쓰던 고정 가중치(bbox×5, giou×2)는
    DETR 논문의 2D 이미지 박스(cx,cy,w,h 4개 값) 기준으로 튜닝된 숫자라,
    1D 구간(l,r 2개 값)인 우리 task에 그대로 옮겨올 근거가 없다(실제로 GIoU
    loss가 raw 상태에서도 이미 다른 loss보다 커서, 거기에 원본 가중치를 또
    곱하면 불균형이 더 커지는 것으로 확인됨). 그렇다고 사람이 임의로 다른
    숫자를 새로 골라도 똑같이 근거 없는 선택이 되므로, 대신 Kendall et al.
    2018(homoscedastic uncertainty weighting)을 적용해 각 loss의 상대적
    가중치(log_var) 자체를 학습 파라미터로 두어 모델이 학습 중 스스로
    찾아가게 한다 — 별도의 사전 체크포인트 없이 이번 학습에서 처음부터
    (0으로 초기화되어 균등한 상태에서 시작해) 다른 파라미터들과 같이 학습됨.
    """
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # [class, bbox, giou] 순서. 0으로 초기화 = exp(-0)=1, 즉 처음엔 세 loss를
        # 동등하게 취급하다가, 학습되면서 모델이 알아서 상대적 중요도를 조정함.
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, class_logits, boxes, targets):
        # 매칭 비용도 지금 학습 중인 상대적 중요도(log_var)를 그대로 써서,
        # 매칭 기준과 최종 loss 기준이 항상 일치하게 함(고정된 별도 숫자 없음).
        with torch.no_grad():
            implicit_weights = torch.exp(-self.log_vars)
        indices = self.matcher(
            class_logits, boxes, targets,
            weight_class=implicit_weights[0].item(),
            weight_bbox=implicit_weights[1].item(),
            weight_giou=implicit_weights[2].item(),
        )

        no_object_class = self.num_classes
        target_classes = torch.full(
            class_logits.shape[:2], no_object_class,
            dtype=torch.int64, device=class_logits.device
        )
        for b, (query_idx, tgt_idx) in enumerate(indices):
            if query_idx.numel() > 0:
                target_classes[b, query_idx] = targets[b]["labels"][tgt_idx].to(class_logits.device)

        loss_class = F.cross_entropy(
            class_logits.transpose(1, 2), target_classes, weight=self.empty_weight.to(class_logits.device)
        )

        matched_pred_boxes = torch.cat([
            boxes[b, query_idx] for b, (query_idx, _) in enumerate(indices) if query_idx.numel() > 0
        ], dim=0) if any(qi.numel() > 0 for qi, _ in indices) else torch.zeros(0, 2, device=boxes.device)

        matched_tgt_boxes = torch.cat([
            targets[b]["boxes"][tgt_idx].to(boxes.device) for b, (_, tgt_idx) in enumerate(indices) if tgt_idx.numel() > 0
        ], dim=0) if any(ti.numel() > 0 for _, ti in indices) else torch.zeros(0, 2, device=boxes.device)

        num_boxes = max(matched_tgt_boxes.shape[0], 1)

        if matched_tgt_boxes.shape[0] > 0:
            loss_bbox = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction='sum') / num_boxes
            giou = torch.diagonal(pairwise_giou_1d(matched_pred_boxes, matched_tgt_boxes))
            loss_giou = (1 - giou).sum() / num_boxes
        else:
            loss_bbox = torch.zeros((), device=class_logits.device)
            loss_giou = torch.zeros((), device=class_logits.device)

        # Kendall et al. 2018 homoscedastic uncertainty weighting:
        # 각 loss_i를 exp(-log_var_i)*loss_i + log_var_i 형태로 감싸서 반환.
        # log_var_i가 커지면(=이 loss가 불확실/신뢰도 낮다고 학습되면) 그 loss의
        # 실제 기여도는 자동으로 줄고, 대신 +log_var_i 항이 그만큼 커져서 "그냥
        # 다 무시해버리는" 손쉬운 회피는 못 하게 막아준다 — 이 regularizer 항을
        # 각 항에 이미 포함해서 반환하므로, train.py에서 셋을 단순히 더하기만
        # 해도(train.py는 4개 반환값을 그대로 sum함) 올바른 uncertainty-weighted
        # 총 loss가 됨.
        weighted_class = torch.exp(-self.log_vars[0]) * loss_class + self.log_vars[0]
        weighted_bbox = torch.exp(-self.log_vars[1]) * loss_bbox + self.log_vars[1]
        weighted_giou = torch.exp(-self.log_vars[2]) * loss_giou + self.log_vars[2]

        return {
            "loss_class": weighted_class,
            "loss_bbox": weighted_bbox,
            "loss_giou": weighted_giou,
        }
