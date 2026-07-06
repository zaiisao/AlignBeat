"""
BeatFCOS를 위한 축소판 RT-DETR 스타일 set-prediction 헤드.

[무엇] model_module.py의 조밀한 anchor 기반 파이프라인(ClassificationModel /
RegressionModel / Anchors / CombinedLoss)을 대체하는, 고정 개수의 learned
query가 (class, interval) 쌍을 직접 예측하는 헤드. 예측 개수가 고정되고
중복 억제가 학습 과정에서 end-to-end로 이루어지므로(DETR과 동일한 원리),
추론 시 NMS/Soft-NMS가 필요 없다.

[왜] Soft-NMS 같은 후처리 자체를 없애고 싶다는 요청이 있었음. 후처리(NMS)를
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def monotonic_match(cost):
    """Optimal order-preserving assignment of M targets into Q sorted
    candidates, via O(Q*M) DP (a 1D monotonic special case of the assignment
    problem - see module docstring for why this is valid here).

    cost: (Q, M) numpy array where rows/cols are already sorted by position.
    Assumes Q >= M (excess rows are simply left unmatched).

    Returns (query_indices, target_indices): both (M,) int64 arrays, indices
    into the Q/M *sorted* order, one query per target, strictly increasing.
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
    """1D generalized IoU between every pair of intervals in a and b.

    a: (N, 2), b: (M, 2), both (l, r) with l <= r. Returns (N, M).
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
            nn.Linear(feature_size, 2),  # (center, length), sigmoid-normalized
        )

    def forward(self, feature_maps):
        # concat multi-scale FPN tokens into a single memory sequence (no
        # deformable attention / positional encoding per level - simplified)
        memory = torch.cat([f.transpose(1, 2) for f in feature_maps], dim=1)  # (B, sum(L_i), feature_size)

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
    대신 O(Q*M) DP로 순서를 지키는 최적 부분집합 매칭을 찾는다."""
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, class_logits, boxes, targets):
        """
        class_logits: (B, Q, num_classes + 1)
        boxes: (B, Q, 2) normalized (l, r)
        targets: list of length B, each {"labels": (M_b,), "boxes": (M_b, 2)}
        returns: list of (query_idx, target_idx) index pairs (original,
        unsorted indices), one per sample.
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

            # sort by position - this is what makes the monotonic-DP shortcut valid
            q_order = torch.argsort(pred_centers)
            t_order = torch.argsort(tgt_centers)  # annotations should already be time-ordered; sorted defensively

            sorted_pred_boxes = pred_boxes[q_order]
            sorted_tgt_boxes = tgt_boxes[t_order]
            sorted_tgt_labels = tgt_labels[t_order]

            cost_bbox = torch.cdist(sorted_pred_boxes, sorted_tgt_boxes, p=1)
            cost_giou = -pairwise_giou_1d(sorted_pred_boxes, sorted_tgt_boxes)
            cost_class = -out_prob[b][q_order][:, sorted_tgt_labels]

            C = (self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class)

            # num_queries is set high relative to typical beat/downbeat counts per
            # clip; if a clip somehow has more targets than queries, the excess
            # (lowest-priority, i.e. last in sorted order) targets are left unmatched.
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
    losses.py)를 대체하는 역할."""
    def __init__(self, num_classes, matcher, eos_coef=0.1, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, class_logits, boxes, targets):
        indices = self.matcher(class_logits, boxes, targets)

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

        return {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox * self.weight_bbox,
            "loss_giou": loss_giou * self.weight_giou,
        }
