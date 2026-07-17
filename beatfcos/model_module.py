from collections import OrderedDict
from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from beatfcos.utils import AnchorPointTransform, ClipBoxes, nms_2d, soft_nms
from beatfcos.anchors import Anchors
from beatfcos import losses
from beatfcos.losses import CombinedLoss
from beatfcos.beat_transformer_encoder import BeatTransformerEncoder


model_urls = {
    'wavebeat8': './backbone/wavebeat8.pth',
    'wavebeat_fold_0': './backbone/wavebeat_folds/fold_0.pth',
    'wavebeat_fold_1': './backbone/wavebeat_folds/fold_1.pth',
    'wavebeat_fold_2': './backbone/wavebeat_folds/fold_2.pth',
    'wavebeat_fold_3': './backbone/wavebeat_folds/fold_3.pth',
    'wavebeat_fold_4': './backbone/wavebeat_folds/fold_4.pth',
    'wavebeat_fold_5': './backbone/wavebeat_folds/fold_5.pth',
    'wavebeat_fold_6': './backbone/wavebeat_folds/fold_6.pth',
    'wavebeat_fold_7': './backbone/wavebeat_folds/fold_7.pth',
    'gnet': './backbone/gnet.pth',
    'tcn2019': './backbone/tcn2019.pth',
}

class PyramidFeatures(nn.Module):
    # feature_size is the number of channels in each feature map
    # >256 => 288 =>  320: C3=256, C=288, C5 = 320

    #def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    def __init__(self, C1_size, C2_size, C3_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        # C3 (coarsest, T/4) → P3
        self.P3_1 = nn.Conv1d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # C2 (T/2) + P3_upsampled → P2
        self.P2_1 = nn.Conv1d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P2_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # C1 (finest, T) + P2_upsampled → P1
        self.P1_1 = nn.Conv1d(C1_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)



    def forward(self, inputs):
        C1, C2, C3 = inputs  # C1=finest(T), C2=(T/2), C3=coarsest(T/4)

        P3_x = self.P3_1(C3)
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P3_upsampled_x + P2_x
        P2_upsampled_x = self.P2_upsampled(P2_x)
        P2_x = self.P2_2(P2_x)

        P1_x = self.P1_1(C1)
        P1_x = P2_upsampled_x + P1_x
        P1_x = self.P1_2(P1_x)

        return [P1_x, P2_x, P3_x]




class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        self.regression = nn.Conv1d(feature_size, 2, kernel_size=3, padding=1)
        self.leftness = nn.Conv1d(feature_size, 1, kernel_size=3, padding=1)
        self.leftness_act = nn.Sigmoid()

        # phase/주기성 auxiliary target: downbeat 박스끼리 겹치는 문제
        # (final_boxes.png, 평균 IoU 0.101 vs beat 0.015)가 모델이 마디 내
        # 위상을 명시적으로 학습하지 못해서라는 가설을 검증하려고 추가함.
        # 각 anchor가 속한 downbeat 구간(마디) 안에서의 위상을 (sin, cos) 단위원
        # 좌표로 예측하게 함 - 자세한 배경은 losses.py의
        # get_phase_targets/PhaseLoss 참고. tanh로 [-1,1] 범위로 bound.
        self.phase = nn.Conv1d(feature_size, 2, kernel_size=3, padding=1)
        self.phase_act = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        regression = self.regression(out)

        # regression is B x C x L, with C = 2*num_anchors
        regression = regression.permute(0, 2, 1)
        # (B, L, 2) where L is the locations of the feature map
        regression = regression.contiguous().view(regression.shape[0], -1, 2)
        # (B, L/2, 2, 2)

        leftness = self.leftness(out)
        leftness = self.leftness_act(leftness)
        leftness = leftness.permute(0, 2, 1)
        leftness = leftness.contiguous().view(leftness.shape[0], -1, 1)

        phase = self.phase(out)
        phase = self.phase_act(phase)
        phase = phase.permute(0, 2, 1)
        phase = phase.contiguous().view(phase.shape[0], -1, 2)

        return regression, leftness, phase

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x L, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 1)

        #batch_size, width, height, channels = out1.shape
        batch_size, length, channels = out1.shape

        out2 = out1.view(batch_size, length, 1, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

#MJ: https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
class BeatFCOS(nn.Module): #MJ: blcok, layers = Bottleneck, [3, 4, 6, 3]: not defined in our code using tcn
    def __init__(
        self,
        num_classes,
        clusters,
        downbeat_weight=0.6,
        # beat_radius/downbeat_radius: FCOS positive anchor 할당 반경 (그대로
        # CombinedLoss -> get_fcos_positives로 전달됨). 자세한 배경 설명은
        # train.py의 --beat_radius/--downbeat_radius 주석 참고.
        beat_radius=2.5,
        downbeat_radius=4.5,
        # phase_weight: 그대로 CombinedLoss -> PhaseLoss로 전달됨. 배경 설명은
        # train.py의 --phase_weight 주석 참고.
        phase_weight=1.0,
        audio_downsampling_factor=32,
        centerness=False,
        postprocessing_type="soft_nms",
        backbone_type="wavebeat",
        audio_sample_rate=22050,
        head_type="fcos",
        num_queries=300,
        decoder_layers=3,
        **kwargs
    ):
        self.inplanes = 256

        super(BeatFCOS, self).__init__()

        self.downbeat_weight = downbeat_weight
        self.audio_downsampling_factor = audio_downsampling_factor  #MJ: 220 with tcn2019, this factor comparaible with 2^7 = 128
        self.postprocessing_type = postprocessing_type

        self.backbone_type = backbone_type
        self.head_type = head_type

        dmodel = kwargs.get('dmodel', 128)
        encoder_keys = ['dmodel', 'nhead', 'd_hid', 'nlayers', 'attn_len', 'dropout', 'norm_first']
        encoder_kwargs = {k: kwargs[k] for k in encoder_keys if k in kwargs}
        self.encoder = BeatTransformerEncoder(**encoder_kwargs)


        self.rho1 = nn.Identity()
        self.rho2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)
        self.rho3 = nn.Sequential(
            nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)
        )

        self.fpn = PyramidFeatures(dmodel, dmodel, dmodel)

        if self.head_type == "hungarian":
            # [무엇] Soft-NMS 후처리를 없애기 위해 추가한 축소판 RT-DETR 경로.
            # 조밀한 anchor(Anchors/ClassificationModel/RegressionModel/CombinedLoss)
            # 대신, 고정 개수 query가 FPN을 보고 (class, interval)을 직접 예측하고
            # 순서 보존 매칭(OrderedMatcher)으로 학습한다. 자세한 근거는
            # beatfcos/hungarian_head.py 파일 상단 docstring 참고.
            # [왜 옵션인가] 기존 head_type="fcos"(기본값) 경로는 전혀 안 건드려서,
            # 두 방식을 나중에 나란히 비교할 수 있게 남겨둠.
            from beatfcos.hungarian_head import SetPredictionHead, OrderedMatcher, SetCriterion
            self.set_prediction_head = SetPredictionHead(
                feature_size=256, num_classes=num_classes,
                num_queries=num_queries, decoder_layers=decoder_layers
            )
            self.set_criterion = SetCriterion(num_classes, OrderedMatcher())
        else:
            self.classificationModel = ClassificationModel(256, num_classes=num_classes)
            self.regressionModel = RegressionModel(256)

            self.anchors = Anchors(clusters, audio_downsampling_factor, audio_sample_rate)
             #MJ: The audio base level is changed from 8 to 7, allowing a more fine-grained audio input
             #  => The target sampling level in wavebeat should be changed to 2^7 from 2^8 as well

            self.anchor_point_transform = AnchorPointTransform()

            self.clipBoxes = ClipBoxes()

            # downbeat_weight: 예전엔 self.downbeat_weight로 저장만 되고 실제 loss
            # 계산에는 전달이 안 되고 있었음(생성자 인자만 있고 미사용 상태) - FocalLoss가
            # beat/downbeat에 같은 alpha를 쓰다 보니 인스턴스가 훨씬 적은 downbeat
            # 채널이 상대적으로 약한 신호만 받아 F-measure가 유독 불안정했던 것으로
            # 관찰되어(자세한 내용은 losses.py의 FocalLoss 참고), 실제로 연결함.
            self.combined_loss = CombinedLoss(clusters, audio_downsampling_factor, audio_sample_rate, centerness=centerness, downbeat_weight=downbeat_weight, beat_radius=beat_radius, downbeat_radius=downbeat_radius, phase_weight=phase_weight)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                # nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)): # The batch normalization will become an identity transformation when
                                                # its weight parameters and bias parameters are set to 1 and 0 respectively
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # End of for m in self.modules()

        if self.head_type != "hungarian":
            # The reinitialization of the final layer of the classification head
            prior = 0.01

            self.classificationModel.output.weight.data.fill_(0)
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

            self.regressionModel.regression.weight.data.fill_(0)
            self.regressionModel.regression.bias.data.fill_(0)

            self.regressionModel.leftness.weight.data.fill_(0)
            self.regressionModel.leftness.bias.data.fill_(0)

            self.regressionModel.phase.weight.data.fill_(0)
            self.regressionModel.phase.bias.data.fill_(0)
        # self.freeze_bn() # If we do not freeze the batch normalization layers, the layers will be trained as was done in WaveBeat

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

    def _forward_hungarian(self, feature_maps, base_level_image_shape, audio_batch, inputs):
        # 참고: 함수명은 hungarian_head.py와의 일관성을 위해 유지했지만, 실제
        # 매칭은 OrderedMatcher(순서 보존 DP)를 쓴다 — 일반 Hungarian 아님.
        # 자세한 근거는 beatfcos/hungarian_head.py 파일 상단 docstring 참고.
        class_logits, boxes = self.set_prediction_head(feature_maps)  # boxes normalized (l, r) in [0, 1]
        T = base_level_image_shape[-1]

        annotations = inputs[1] if len(inputs) == 2 else None

        targets = None
        if annotations is not None:
            targets = []
            for j in range(annotations.shape[0]):
                jth = annotations[j]
                jth = jth[jth[:, 2] != -1]
                targets.append({
                    "labels": jth[:, 2].long(),
                    "boxes": (jth[:, 0:2].float() / T).clamp(0, 1),
                })
            losses = self.set_criterion(class_logits, boxes, targets)

        if self.training:
            zero = torch.zeros((), device=class_logits.device)
            # train.py의 5-tuple 언패킹/출력("CLS/REG/LFT/ADJ/PHA")을 그대로
            # 재사용하기 위해 같은 모양으로 반환함: CLS<-분류loss, REG<-bbox(L1),
            # LFT<-giou, ADJ/PHA<-미사용(0). 출력에 찍히는 라벨 이름은 FCOS
            # 시절 그대로라 의미가 안 맞지만, 값 자체는 실제 set-prediction
            # loss가 맞다. hungarian head에는 anchor 자체가 없어서 phase/마디
            # 위상 auxiliary target 개념도 적용 대상이 아님.
            return (
                losses["loss_class"].unsqueeze(0),
                losses["loss_bbox"].unsqueeze(0),
                losses["loss_giou"].unsqueeze(0),
                zero.unsqueeze(0),
                zero.unsqueeze(0),
            )
        else:
            probs = class_logits.softmax(-1)
            scores, labels = probs[..., :-1].max(-1)  # "no object" 클래스는 제외
            boxes_raw = boxes * T  # 정규화된 [0,1]을 다시 frame-index 스케일로 (FCOS 경로와 동일한 단위로 맞춤)

            # 참고: eval 시 batch_size==1을 가정함 (위 FCOS 경로에서 beat_eval.py가
            # 평가를 돌리는 방식과 동일).
            finalScores = scores[0]
            finalAnchorBoxesIndexes = labels[0]
            finalAnchorBoxesCoordinates = boxes_raw[0]

            if annotations is not None:
                eval_losses = (
                    losses["loss_class"].item(),
                    losses["loss_bbox"].item(),
                    losses["loss_giou"].item(),
                    0.0,
                )
            else:
                eval_losses = (0.0, 0.0, 0.0, 0.0)

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, eval_losses]

    # downbeat_score_threshold: 안 주면(None) score_threshold가 beat/downbeat
    # 둘 다에 적용됨(기존 동작과 동일). 값을 주면 downbeat 클래스만 그 threshold로
    # 독립적으로 후처리됨 (아래 soft_nms 부분의 class_score_threshold 참고).
    # return_raw_scores: True면 soft-NMS/threshold를 전부 건너뛰고, anchor별
    # 원본 점수(leftness 곱한 값)와 위치를 그대로 반환함 - madmom DBN처럼 NMS가
    # 아닌 다른 디코딩 방식에 넣기 위한 진단용 경로 (beat_eval.py의 DBN 관련
    # 주석/evaluate_with_dbn.py 참고).
    def forward(self, inputs, iou_threshold=0.5, score_threshold=0.05, max_thresh=1, downbeat_score_threshold=None, downbeat_sigma=None, return_raw_scores=False): #:forward_call = forward
        # inputs = audio, target
        # self.training = len(inputs) == 2

        if len(inputs) == 2:
            audio_batch, annotations = inputs  #MJ: audio_batch: shape = (16/1,1,3000,81); annotations: shape=(16/1,128,3)
        else:
            audio_batch = inputs


        C1, C2, C3 = self.encoder(audio_batch)

        C1 = C1.transpose(1, 2)
        C2 = C2.transpose(1, 2)
        C3 = C3.transpose(1, 2)

        C1 = self.rho1(C1)
        C2 = self.rho2(C2)
        C3 = self.rho3(C3)

        base_level_image_shape = C1.shape

        feature_maps = self.fpn([C1, C2, C3])

        if self.head_type == "hungarian":
            return self._forward_hungarian(feature_maps, base_level_image_shape, audio_batch, inputs)

        classification_outputs = torch.cat([self.classificationModel(feature_map) for feature_map in feature_maps], dim=1)
        regression_outputs = []
        leftness_outputs = []
        phase_outputs = []

        for feature_map in feature_maps:
            bbx_regression_output, leftness_regression_output, phase_output = self.regressionModel(feature_map)

            regression_outputs.append(bbx_regression_output)
            leftness_outputs.append(leftness_regression_output)
            phase_outputs.append(phase_output)

        regression_outputs = torch.cat(regression_outputs, dim=1)
        leftness_outputs = torch.cat(leftness_outputs, dim=1)
        phase_outputs = torch.cat(phase_outputs, dim=1)

        anchors_list = self.anchors(base_level_image_shape)

        # All classification outputs should be the same so we just pick the 0th one
        number_of_classes = classification_outputs.size(dim=2)

        focal_losses_batch_all_classes, regression_losses_batch_all_classes, leftness_losses_batch_all_classes = [], [], []
        class_one_cls_targets, class_one_reg_targets = None, None
        class_one_positive_indicators, class_two_positive_indicators = None, None

        # This combined loss will eventually replace the legacy losses we have been using
        classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss, phase_loss = self.combined_loss(
            classification_outputs, regression_outputs, leftness_outputs, phase_outputs, anchors_list, annotations
        )

        if self.training:
            return classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss, phase_loss
        else:
            # Start of evaluation mode

            all_anchors = torch.cat(anchors_list, dim=0)

            if return_raw_scores:
                # class_id 0 = downbeat, class_id 1 = beat (get_jth_targets 컨벤션과 동일)
                raw_downbeat_scores = classification_outputs[:, :, 0] * leftness_outputs[:, :, 0]
                raw_beat_scores = classification_outputs[:, :, 1] * leftness_outputs[:, :, 0]
                # all_anchors: anchor point 위치 (base-level, audio_downsampling_factor
                # 단위 - 초 단위 변환은 audio_downsampling_factor / audio_sample_rate 곱하면 됨)
                return raw_beat_scores, raw_downbeat_scores, all_anchors

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])
            strides_for_all_anchors = torch.zeros(0)

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
                strides_for_all_anchors = strides_for_all_anchors.cuda()

            for i, anchors_per_level in enumerate(anchors_list):    # i ranges over the level of feature maps.
                stride_per_level = torch.tensor(2**i).to(strides_for_all_anchors.device) #stride_per_level =2**1, 2**2, 2**3, 2**4,2**5
                stride_per_level_for_anchors = stride_per_level[None].expand(anchors_per_level.size(dim=0)) #MJ:anchors_per_level.size(dim=0)=188
                strides_for_all_anchors = torch.cat((strides_for_all_anchors, stride_per_level_for_anchors), dim=0)

            transformed_regression_boxes = self.anchor_point_transform(all_anchors, regression_outputs, strides_for_all_anchors)
            transformed_regression_boxes = self.clipBoxes(transformed_regression_boxes, audio_batch.transpose(1, 2))

            for class_id in range(classification_outputs.shape[2]): # the shape of classification_output is (B, number of anchor points per level, class ID)
                scores = classification_outputs[:, :, class_id] * leftness_outputs[:, :, 0] # We predict the max number for beats will be less than the num of anchors

                # class_id 0 = downbeat, class_id 1 = beat (see get_jth_targets in losses.py).
                # beat와 downbeat은 confidence 분포가 달라서 최적 threshold도 다름
                # (실측: beat=0.2, downbeat=0.05가 최적) - downbeat_score_threshold를
                # 안 주면 기존처럼 score_threshold 하나로 두 클래스에 동일 적용됨.
                class_score_threshold = downbeat_score_threshold if (class_id == 0 and downbeat_score_threshold is not None) else score_threshold

                scores_over_thresh = torch.logical_and(scores > class_score_threshold, scores <= max_thresh)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]

                regression_boxes = transformed_regression_boxes[scores_over_thresh]

                if self.postprocessing_type == 'gnet':
                    from gossipnet.model.gnet import GNet
                    gnet = GNet(numBlocks=4)
                    gnet.cuda()
                    checkpoint = torch.load(model_urls['gnet'])
                    gnet.load_state_dict(checkpoint['model_state_dict'])
                    gnet.eval()

                    data = torch.stack((  #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                        regression_boxes[:, 0],
                        torch.zeros(regression_boxes.size(dim=0)).to(regression_boxes.device),
                        regression_boxes[:, 1],
                        torch.ones(regression_boxes.size(dim=0)).to(regression_boxes.device),
                        torch.ones(scores.shape).to(scores.device) * class_id,
                        scores
                    ), dim=1).unsqueeze(dim=0) #MJ: detections refer to predicted bboxes by beat-fcos
                    
                    logit_scores = gnet(batch=data)  #MJ: scores for each detection/anchor point
                    logit_scores = logit_scores[0]
                    scores = torch.sigmoid(logit_scores)

                    num_remaining_scores = torch.sum(scores > 0.2)

                    anchors_nms_idx = torch.argsort(scores, descending=True)[:num_remaining_scores]
                elif self.postprocessing_type == 'nms':
                    # During NMS, if the IoU of two adjacent predicted boxes is less than IoU threshold, the two boxes are considered to be different beats
                    # Otherwise both predictions are considered redundant so that one is removed.

                    anchors_nms_idx = nms_2d(regression_boxes, scores, iou_threshold)
                    #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                    #    Get all the filtered detections and store them for use in training gnet.
                elif self.postprocessing_type == 'soft_nms':
                    # 원래 여기 thresh=0.2로 하드코딩되어 있어서, score_threshold를
                    # 밖에서 아무리 낮춰도(0.05, 0.02 등) 최종 결과는 항상 바뀌지
                    # 않았음 - score_threshold는 soft_nms에 들어갈 후보군만 거르는
                    # 역할이고, 실제 최종 컷은 이 하드코딩된 값이었기 때문. 이제
                    # class_score_threshold(클래스별로 다를 수 있음)를 그대로 써서
                    # 밖에서 지정한 threshold가 실제로 최종 결과에 반영되게 함.
                    #
                    # downbeat_sigma: downbeat box가 beat box보다 폭이 훨씬 넓어서
                    # (박마디 길이) 연속된 두 박스가 겹치는 경우가 많음(final_boxes.png
                    # 시각화로 확인됨, 평균 IoU 0.101 vs beat 0.015). soft-NMS의
                    # sigma는 겹치는 이웃 박스의 점수를 얼마나 강하게 깎을지 결정하는
                    # 파라미터라서, downbeat만 별도로 조정해 겹침을 더 강하게
                    # 억제할 수 있는지 실험.
                    class_sigma = downbeat_sigma if (class_id == 0 and downbeat_sigma is not None) else 0.5
                    anchors_nms_idx = soft_nms(regression_boxes, scores, sigma=class_sigma, thresh=class_score_threshold)
                elif self.postprocessing_type == 'none':
                    anchors_nms_idx = torch.arange(0, regression_boxes.size(dim=0))

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([class_id] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(regression_boxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([class_id] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, regression_boxes[anchors_nms_idx]))
            #END for class_id in range(classification_outputs.shape[2])

            eval_losses = (
                classification_loss.item(),
                regression_loss.item(),
                leftness_loss.item(),
                adjacency_constraint_loss.item()
            )

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, eval_losses]

def create_beatfcos_model(num_classes, clusters, args, **kwargs):
    model = BeatFCOS(num_classes, clusters, **kwargs)
    return model
