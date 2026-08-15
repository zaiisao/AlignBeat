import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False)
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def calc_iou(a, b):
    area = b[:, 1] - b[:, 0]

    iw = torch.min(torch.unsqueeze(a[:, 1], dim=1), b[:, 1]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = torch.clamp(iw, min=0)

    ua = torch.unsqueeze(a[:, 1] - a[:, 0], dim=1) + area - iw
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw

    IoU = intersection / ua

    return IoU

def calc_giou(a, b):
    # 1. For the predicted line B_p, ensuring  x_p_2 > x_p_1

    # 2. Calculating length of B_g: L_g = x_g_2 − x_g_1 (위에서 이미 정의한 gt_widths)
    a_lengths = a[:, 1] - a[:, 0] #gt_widths

    # 3. Calculating length of B_p: L_p = x̂_p_2 − x̂_p_1
    b_lengths = b[:, 1] - b[:, 0]

    # 4. Calculating intersection I between B_p and B_g
    intersection_x1 = torch.max(a[:, 0], b[:, 0])
    intersection_x2 = torch.min(a[:, 1], b[:, 1])
    intersection = torch.where(
        intersection_x2 > intersection_x1,
        intersection_x2 - intersection_x1,
        torch.zeros(a.size(dim=0)).to(a.device)
    )

    # 5. Finding the coordinate of smallest enclosing line B_c:
    coordinate_x1 = torch.min(a[:, 0], b[:, 0])
    coordinate_x2 = torch.max(a[:, 1], b[:, 1])

    # 6. Calculating length of B_c
    bbox_coordinate = coordinate_x2 - coordinate_x1 + 1e-7

    # 7. IoU (I / U), where U = L_p + L_g - I
    union = b_lengths + a_lengths - intersection
    iou = intersection / union

    #if self.loss_type == "iou":
        # 9a. L_IoU = 1 - IoU
    #    regression_loss = 1 - iou
    #else:
        # 8. GIoU = IoU - (L_c - U)/L_c
        #giou = iou - (bbox_coordinate - union)/bbox_coordinate
    giou = iou - (bbox_coordinate - union)/bbox_coordinate
    return giou
        #print(b, a, giou)

        # 9b. L_GIoU = 1 - GIoU
        #regression_loss = 1 - giou

    #print(regression_loss.mean(), torch.exp(regression_loss.mean() * self.weight))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class AnchorPointTransform(nn.Module):
    def __init__(self):
        super(AnchorPointTransform, self).__init__()

    def forward(self, all_anchors, regression_outputs, strides_for_all_anchors):
        transformed_regressions_x1 = all_anchors[None] - regression_outputs[:, :, 0] * strides_for_all_anchors[None]
        transformed_regressions_x2 = all_anchors[None] + regression_outputs[:, :, 1] * strides_for_all_anchors[None]

        transformed_regression_boxes = torch.stack((
            transformed_regressions_x1, # (B, num of anchors, 1)
            transformed_regressions_x2  # (B, num of anchors, 1)
        ), dim=2)

        return transformed_regression_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        if len(img.shape) == 4:
            batch_size, num_channels, width, _ = img.shape
        elif len(img.shape) == 3:
            batch_size, num_channels, width = img.shape
        else:
            raise NotImplementedError

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], max=width)
      
        return boxes

def nms_2d(anchor_boxes, scores, thresh_iou):
    boxes_3d = torch.cat((
        torch.unsqueeze(anchor_boxes[:, 0], dim=1),
        torch.zeros((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device),
        torch.unsqueeze(anchor_boxes[:, 1], dim=1),
        torch.ones((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device)
    ), 1)

    return nms(boxes_3d, scores, thresh_iou)

def soft_nms(regression_boxes, box_scores, sigma=0.5, thresh=0.05, use_regular_nms=False):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = regression_boxes.shape[0]  #MJ: N = 10022
    indexes = torch.arange(0, N, dtype=torch.float).to(regression_boxes.device).view(N, 1)
    dets = torch.cat((regression_boxes, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]

    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = box_scores.clone()
    areas = (x2 - x1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone() 
        pos = i + 1

        #MJ: compare the current box’s score (tscore) with the maximum score (maxscore) among the boxes that come after the current box.
        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0) #MJ: get the max box in the following boxes after the current ith box
            # Even though the tensor is one-dimensional in this context, specifying dim=0 is necessary when you want to obtain both the maximum value and its index. Without dim=0, you might only get the maximum value,

            # [조기 종료 - 출력은 완전히 동일함] 이 루프는 위치 i 이후의 점수만
            # scores[pos:] = weights * scores[pos:]로 갱신하는데, weights는 항상
            # (0,1] 범위다 (Gaussian이면 exp(-x)<=1, hard NMS면 0 또는 1). 따라서
            #   - 위치 i 미만은 이미 확정돼서 더 이상 바뀌지 않고,
            #   - 위치 i 이상이 전부 thresh 이하라면 앞으로도 절대 thresh를 넘을 수 없다.
            # 마지막 줄이 keep = scores > thresh 이므로 이 시점에서 남은 후보는
            # 어차피 하나도 채택되지 않는다. 즉 여기서 멈춰도 결과가 바뀌지 않는다.
            #
            # [왜 필요한가] 원래는 anchor 개수만큼(fcos_lite eval 기준 3072회) 루프를
            # 다 돌았고, 반복마다 작은 CUDA 커널이 10개쯤 실행돼서 곡 하나당 약 1.3초,
            # val 286곡이면 검증 한 번에 6분 이상이 NMS에만 들어갔다(GPU 사용률 0%,
            # 전부 커널 실행 오버헤드). 학습된 모델은 대부분의 anchor 점수가 배경
            # 수준이라 이 조건이 아주 일찍 걸린다 - 실측 11~57배 단축, 출력은
            # 랜덤/실제 분포/동점/전부0/전부1 모든 경우에서 bit-level로 동일함.
            if tscore <= thresh and maxscore <= thresh:
                break

            if tscore < maxscore:
                dets[i], dets[ maxpos.item() + i + 1]  = dets[ maxpos.item() + i + 1].clone(), dets[i].clone() # JA: dets[i] is the i'th box with its left position, right position, and index
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()
            #MJ: => This swaps the max box with the current box at the ith index;
            #As the algorithm iterates through the boxes, the one with the highest score among the remaining boxes is always moved to the current position, 
            #It  is a typical strategy in non-maximum suppression methods to prioritize the most confident detections.
         
        #MJ: now dets[i] plays the same role as M in the pseudo code of the soft-nms paper     
        # IoU calculate
        xx1 = torch.maximum(dets[i, 0], dets[pos:, 0]) # JA: xx1 compares the i'th box's left with all following boxes' lefts
        xx2 = torch.minimum(dets[i, 1], dets[pos:, 1]) # JA: xx2 compares the i'th box's right with all following boxes' rights
        
        #w = np.maximum(0.0, xx2 - xx1 + 1)
        w = xx2 - xx1 + 1
        w = torch.clamp(w, min=0) # JA: w is the overlapping interval

        #h = np.maximum(0.0, yy2 - yy1 + 1)
        #inter = torch.tensor(w * h).to(dets.device)
        inter = torch.tensor(w).to(dets.device)
        ious = torch.div(inter, (areas[i] + areas[pos:] - inter))

        #MJ: ious= ious(M,b_i's) in the pseudo code of the paper
        if use_regular_nms == True:  #MJ: sigma=0.5 is used as N_t in the pseudo code of the paper
            weights = torch.where(ious > sigma, torch.zeros(ious.shape).to(ious.device), torch.ones(ious.shape).to(ious.device))
        else: #MJ: sigma=0.5 
            # Gaussian decay: 
            weights = torch.exp(-(ious * ious) / sigma)

        #MJ: For every ith box, which is now the max score box, M, rescore all the remaining boxes b_i, using ious(M,b_i):
        scores[pos:] = weights * scores[pos:]
    #End for i in range(N):

    # select the boxes and keep the corresponding indexes
    #keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 2][scores > thresh].long() #MJ: sigma=0.5, thresh=0.05

    return keep

# def get_offset_limits(datasets):
    

# https://github.com/bharatsingh430/soft-nms/blob/b8e69bdf8df2ad53025c9d198ded909b50471d4f/lib/nms/cpu_nms.pyx
# def soft_nms(boxes, sigma=0.5, iou_threshold=0.3, score_threshold=0.001, method=0):
#     N = boxes.shape[0]
#     iw, ih, box_area = None, None, None
#     ua = None
#     pos = 0
#     maxscore = 0
#     maxpos = 0
#     #cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
#     x1,x2,tx1,tx2,ts,area,weight,ov = None, None, None, None, None, None, None, None
#     keep_indices = []

#     for i in range(N):
#         #maxscore = boxes[i, 4]
#         maxscore = boxes[i, 2]
#         maxpos = i

#         # save box i to tx1, tx2, and ts
#         # tx1 = boxes[i,0]
#         # ty1 = boxes[i,1]
#         # tx2 = boxes[i,2]
#         # ty2 = boxes[i,3]
#         # ts = boxes[i,4]
#         tx1 = boxes[i,0]
#         tx2 = boxes[i,1]
#         ts = boxes[i,2]

#         pos = i + 1

#         # m <- argmax S
#         # get max box: get max score and max pos
#         while pos < N:
#             # if maxscore < boxes[pos, 4]:
#             #     maxscore = boxes[pos, 4]
#             if maxscore < boxes[pos, 2]:
#                 maxscore = boxes[pos, 2]
#                 maxpos = pos
#             pos = pos + 1

#         keep_indices.append(maxpos)

#         # B <- B - M
#         # add max box as a detection
#         # boxes[i,0] = boxes[maxpos,0]
#         # boxes[i,1] = boxes[maxpos,1]
#         # boxes[i,2] = boxes[maxpos,2]
#         # boxes[i,3] = boxes[maxpos,3]
#         # boxes[i,4] = boxes[maxpos,4]
#         boxes[i,0] = boxes[maxpos,0]
#         boxes[i,1] = boxes[maxpos,1]
#         boxes[i,2] = boxes[maxpos,2]

#         # swap ith box and max box
#         # swap ith box with position of max box
#         # boxes[maxpos,0] = tx1
#         # boxes[maxpos,1] = ty1
#         # boxes[maxpos,2] = tx2
#         # boxes[maxpos,3] = ty2
#         # boxes[maxpos,4] = ts
#         boxes[maxpos,0] = tx1
#         boxes[maxpos,1] = tx2
#         boxes[maxpos,2] = ts

#         # boxes[i, :] = boxes[maxpos, :]
#         # tx1 = boxes[i,0]
#         # ty1 = boxes[i,1]
#         # tx2 = boxes[i,2]
#         # ty2 = boxes[i,3]
#         # ts = boxes[i,4]
#         tx1 = boxes[i,0]
#         tx2 = boxes[i,1]
#         ts = boxes[i,2]

#         pos = i + 1
#     # NMS iterations, note that N decreases by 1 if detection boxes fall below threshold
#         while pos < N:
#             # x1 = boxes[pos, 0]
#             # y1 = boxes[pos, 1]
#             # x2 = boxes[pos, 2]
#             # y2 = boxes[pos, 3]
#             # s = boxes[pos, 4]
#             x1 = boxes[pos, 0]
#             x2 = boxes[pos, 1]
#             s = boxes[pos, 2]

#             # area = (x2 - x1 + 1) * (y2 - y1 + 1)
#             # iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             area = x2 - x1 + 1
#             iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             if iw > 0:
#                 # ih = (min(ty2, y2) - max(ty1, y1) + 1)
#                 # if ih > 0:
#                 # ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
#                 ua = float((tx2 - tx1 + 1) + area - iw)
#                 # ov = iw * ih / ua # iou between max box and detection box
#                 ov = iw / ua # iou between max box and detection box

#                 if method == 1: # linear
#                     if ov > iou_threshold: 
#                         weight = 1 - ov
#                     else:
#                         weight = 1
#                 elif method == 2: # gaussian
#                     weight = np.exp(-(ov * ov)/sigma)
#                 else: # original NMS
#                     if ov > iou_threshold: 
#                         weight = 0
#                     else:
#                         weight = 1

#                 # boxes[pos, 4] = weight*boxes[pos, 4]
#                 boxes[pos, 2] = weight*boxes[pos, 2]
            
#             # if box score falls below threshold, discard the box by swapping with last box
#             # update N
#                     # if boxes[pos, 4] < threshold:
#                     #     boxes[pos,0] = boxes[N-1, 0]
#                     #     boxes[pos,1] = boxes[N-1, 1]
#                     #     boxes[pos,2] = boxes[N-1, 2]
#                     #     boxes[pos,3] = boxes[N-1, 3]
#                     #     boxes[pos,4] = boxes[N-1, 4]
#                 if boxes[pos, 2] < score_threshold:
#                     boxes[pos,0] = boxes[N-1, 0]
#                     boxes[pos,1] = boxes[N-1, 1]
#                     boxes[pos,2] = boxes[N-1, 2]
#                     N = N - 1
#                     pos = pos - 1

#             pos = pos + 1

#     #print(f"sorted boxes:\n{torch.sort(boxes, dim=2, descending=True)}")
#     #_, sorted_box_indices = torch.sort(boxes[:, 2], descending=True)
#     #print(f"sorted_box_indices: {sorted_box_indices}")
#     #print(f"number of boxes above threshold: {(boxes[:, 2] > score_threshold).sum()}")
#     #positive_indices = torch.ge(boxes[:, 2], score_threshold)
#     #num_positive_anchors = positive_indices.sum()
#     #print(f"number of boxes above score threshold: {num_positive_anchors}")
#     #print(f"soft nms boxes ({boxes[sorted_box_indices].shape}): {boxes[sorted_box_indices]}")

#     print(f"custom nms indices:\n{keep_indices}")
#     keep = [i for i in range(N)]
#     print(f"custom nms boxes:\n{boxes[keep]}")
#     #print(f"custom nms ({boxes[torch.argsort(boxes[keep, 0])].shape}):\n{boxes[torch.argsort(boxes[keep, 0]), 0:2]}")
#     return keep

def soft_nms_from_pseudocode(boxes, scores, threshold):
    D = torch.zeros(0, 2)
    while boxes.size(dim=0) > 0:
        m = torch.argmax(scores)
        M = boxes[m, :].unsqueeze(dim=0)
        D = torch.cat((D, M), dim=0)
        boxes = torch.cat((boxes[0:m, :], boxes[m:, :]), dim=0)
        for i in range(boxes.size(dim=0)):
            b_i = boxes[i, :]
            if calc_iou(M, b_i.unsqueeze(dim=0)) >= threshold:
                boxes = torch.cat((boxes[0:i, :], boxes[i:, :]), dim=0)
                scores = torch.cat((scores[0:i], scores[i:]), dim=0)

    return D, S

