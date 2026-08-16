import math
import os
import glob
import torch
import torchsummary
import re
import random
import numpy as np
import collections
from itertools import product
from argparse import ArgumentParser
import traceback
import sys
from os.path import join as ospj
from kmeans_pytorch import kmeans, kmeans_predict

from beatfcos import model_module
from beatfcos.dataloader import BeatDataset, collater, BEAT_ONLY_DATASETS
from beatfcos.beat_eval import evaluate_beat_f_measure, evaluate_beat_f_measure_subset

class Logger(object):
    """Log stdout messages."""
    def __init__(self, outfile, mode="w"):
        self.terminal = sys.stdout
        self.log = open(outfile, mode)
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

def configure_log(log_file_name, mode="w"):
    Logger(log_file_name, mode)

# 원래는 log.log/GPU/checkpoints 경로가 전부 하드코딩이라, 8-fold CV처럼 여러 run을
# 동시에/순차적으로 돌리면 서로 같은 log.log와 checkpoints/ 폴더를 덮어써버림.
# --log_file, --gpu, --checkpoint_dir로 fold별로 분리할 수 있게 argparse 이후로
# 옮김 (아래 args 파싱 직후 참고).

torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--dataset_dir', type=str, default=None)
parser.add_argument('--beatles_audio_dir', type=str, default=None)
parser.add_argument('--beatles_annot_dir', type=str, default=None)
parser.add_argument('--ballroom_audio_dir', type=str, default=None)
parser.add_argument('--ballroom_annot_dir', type=str, default=None)
parser.add_argument('--hainsworth_audio_dir', type=str, default=None)
parser.add_argument('--hainsworth_annot_dir', type=str, default=None)
parser.add_argument('--rwc_popular_audio_dir', type=str, default=None)
parser.add_argument('--rwc_popular_annot_dir', type=str, default=None)
parser.add_argument('--carnatic_audio_dir', type=str, default=None)
parser.add_argument('--carnatic_annot_dir', type=str, default=None)
parser.add_argument('--harmonix_audio_dir', type=str, default=None)
parser.add_argument('--harmonix_annot_dir', type=str, default=None)
# SMC: downbeat 라벨이 없는 beat-only 데이터셋. Beat Transformer가 학습에 쓰는
# 7개 중 하나라 pool을 맞추려면 필요함. dataloader가 beat interval만 class_id=2로
# 내보내고(make_intervals), FCOS 쪽은 losses.py가 downbeat 채널을 -1 sentinel로
# 마스킹, subset head는 논문 7.1절(eq. 9 marginal / eq. 10-11 EM)로 처리한다.
parser.add_argument('--smc_audio_dir', type=str, default=None)
parser.add_argument('--smc_annot_dir', type=str, default=None)
parser.add_argument('--preload', default=False, action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=22050)
parser.add_argument('--audio_downsampling_factor', type=int, default=512)  # 128 → 512 (hop_length)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
# default=None: head_type에 따라 아래에서 결정됨 (명시하면 그 값이 항상 우선).
# fcos 계열은 기존과 동일하게 2097152샘플(=4096 mel frame, 95.1초)을 그대로 쓰고,
# subset head는 논문 규약대로 D=29.72초(=1280 frame = num_candidates*8)를 씀.
parser.add_argument('--train_length', type=int, default=None)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=2097152)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', default=True, action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
# default=None: head별로 아래에서 결정 (명시하면 그 값이 우선). fcos 계열은 기존과
# 동일하게 1e-3. subset head는 3e-4 - lambda_L1 = 1/b가 200이라 시간 항의 gradient가
# 크고(실측 grad norm 중앙값 50~160, 최대 7.7e4), 1e-3에서는 단일 fragment
# overfit조차 발산했음(loss 2.77 -> 2093). 3e-4/1e-4에서는 63/63 이벤트를 ±70ms
# 안에 맞추며 정상 수렴.
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--ninputs', type=int, default=1)
parser.add_argument('--noutputs', type=int, default=2)
parser.add_argument('--nblocks', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=15)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--dilation_growth', type=int, default=8)
parser.add_argument('--channel_growth', type=int, default=32)
parser.add_argument('--channel_width', type=int, default=32)
parser.add_argument('--stack_size', type=int, default=4)
parser.add_argument('--grouped', default=False, action='store_true')
parser.add_argument('--causal', default=False, action="store_true")
parser.add_argument('--skip_connections', default=False, action="store_true")
parser.add_argument('--norm_type', type=str, default='BatchNorm')
parser.add_argument('--act_type', type=str, default='PReLU')
parser.add_argument('--downbeat_weight', type=float, default=0.6)
# beat_radius/downbeat_radius: FCOS positive anchor 할당 시 GT 위치로부터 이
# 배수*stride 이내의 anchor를 positive로 인정하는 반경. downbeat_radius가
# beat_radius보다 넓으면(기존 기본값 4.5 vs 2.5) downbeat GT 하나당 학습되는
# positive anchor 범위가 더 넓어져서, 학습 신호가 흐릿해지고(진짜 downbeat의
# confidence도 낮아짐) 인접 anchor들도 downbeat로 오검출되는 원인이 됨(2~3배
# 과다예측 관찰됨). beat 수준(2.5~3.0)으로 낮춰서 재학습 검증 중.
parser.add_argument('--beat_radius', type=float, default=2.5)
parser.add_argument('--downbeat_radius', type=float, default=4.5)
# phase_weight: downbeat 박스끼리 겹치는 문제(final_boxes.png로 시각화 확인,
# 평균 IoU 0.101 vs beat 0.015)가 모델이 마디 내 위상/주기성을 명시적으로
# 학습하지 못해서라는 가설을 테스트하기 위한 auxiliary target의 loss 가중치.
# 각 anchor가 속한 downbeat 구간(마디) 안에서의 위상을 (sin, cos)로 인코딩해
# regression head 옆에 새로 붙인 phase head가 예측하도록 함 (DBN 같은 수작업
# 후처리나 실패했던 NMS-free 방식 대신, end-to-end로 리듬 일관성을 학습시키려는
# 시도 - losses.py의 get_phase_targets/PhaseLoss 참고).
parser.add_argument('--phase_weight', type=float, default=1.0)
parser.add_argument('--pretrained', default=False, action="store_true")  # True → False
parser.add_argument('--freeze_backbone', default=False, action="store_true")
parser.add_argument('--centerness', default=False, action="store_true")
parser.add_argument('--postprocessing_type', type=str, default='soft_nms')
parser.add_argument('--no_adj', default=False, action="store_true")
parser.add_argument('--validation_fold', type=int, default=None)
parser.add_argument('--backbone_type', type=str, default="wavebeat")
parser.add_argument('--hop_length_in_seconds', type=float, default=0.01) # This is from Spectral TCN
parser.add_argument('--dmodel', type=int, default=128)
# DilatedTransformerLayer.py의 attention head 분할(k[:,0:4]=symmetric 4개,
# k[:,4:5]/k[:,5:6]/k[:,6:7]x2=skewed/dilated 4개, 총 8개 전제)이 하드코딩돼
# 있는데 nhead 기본값이 2였음 - head가 2개뿐이면 저 skewed 4개 슬라이스가 전부
# 빈 텐서가 되어 dilated attention이 사실상 죽고 symmetric(shift=0)만 남는
# 버그가 있었음 (Beat Transformer 백본의 핵심 메커니즘이 꺼진 상태로 지금까지
# 모든 실험이 돌아간 것). 8로 맞춰서 실제로 dilated head가 동작하게 함.
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--d_hid', type=int, default=512)
parser.add_argument('--nlayers', type=int, default=9)
parser.add_argument('--attn_len', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.1)
# Soft-NMS 후처리를 없앤 축소판 RT-DETR 헤드(순서 보존 매칭 기반, 자세한 내용은
# beatfcos/hungarian_head.py 참고)를 쓰려면 --head_type hungarian으로 지정.
# 기본값 'fcos'는 기존 anchor+Soft-NMS 파이프라인을 그대로 유지함(비교용).
parser.add_argument('--head_type', type=str, default='fcos', choices=['fcos', 'hungarian', 'fcos_lite', 'fcos_no_fpn', 'subset'])
parser.add_argument('--num_queries', type=int, default=300)
parser.add_argument('--decoder_layers', type=int, default=3)

# --- order-constrained subset selection head (--head_type subset) ---------------
# 논문: beat_dp_matching-1.pdf. 구현은 beatfcos/subset_head.py 참고.
# num_candidates: 논문 eq. N = BPM_max * D_min. 논문 기본값(BPM_max=200 -> N=100)은
# 이 데이터에 대해 실측상 부족함 - 8개 데이터셋 전체 annotation을 30초 창으로 훑어본
# 결과 최대 밀도가 150 events/30s(gtzan jazz ~300BPM)였고, 학습셋도 carnatic 147 /
# harmonix 135 / ballroom 107로 100을 넘는 파일이 124개 있었다. M > N이면 DP가
# 애초에 불가능(D[M,N]=inf)하고 추론 시에도 recall이 구조적으로 막히므로 여유를 둠.
parser.add_argument('--num_candidates', type=int, default=160)
# b_scale: eq. (2) Laplace 관측 모형의 scale. lambda_L1 = 1/b이고, 이건 자유 가중치가
# 아니라 시간 관측 노이즈의 정밀도임. 시간이 window 전체 기준 (0,1)로 정규화돼 있어서
# ±70ms 허용오차가 정규화 단위로는 70ms/29.72s ~ 0.0024밖에 안 됨 - b를 그 근처로
# 잡으면 lambda_L1이 수백이 되어 DP 선택이 사실상 "시간상 최근접"이 되어버린다.
# 학습 로그의 [subset] 진단 줄(one-slot time cost vs class spread)을 보고 조정할 것.
parser.add_argument('--b_scale', type=float, default=0.005)
# learn_b: eq. (5) closed-form MLE(매칭된 쌍의 평균 절대 잔차)로 b를 EMA 갱신.
# 기본값 off - 먼저 고정 b로 학습이 안정적인지 확인한 뒤 켜는 것을 권장.
# [주의/실측] 멀티 GPU DataParallel에서는 replica의 buffer 갱신이 버려지기 때문에
# GPU 0의 half-batch 잔차만 EMA에 반영됨(재현 확인됨). 단일 GPU(--gpu N 하나)에서는
# 정상 동작. learn_b를 켤 거면 단일 GPU로 돌릴 것.
parser.add_argument('--learn_b', action='store_true', default=False)
# gamma: eq. (8)에서 background(미매칭 후보) 항의 가중치. N=160 후보 중 실제 이벤트가
# 보통 60~70개라 나머지 ~100개가 전부 background 신호를 냄.
parser.add_argument('--gamma', type=float, default=0.5)
# omega_db: eq. (8)의 클래스별 가중치. downbeat은 beat보다 약 1/L배로 드물어서
# (section 9.3) 매칭된 downbeat 분류 오차를 더 세게 반영함. omega_beat = 1.0 고정.
parser.add_argument('--omega_db', type=float, default=2.0)
# tau: Algorithm 3의 신뢰도 threshold. 학습 후 val에서 sweep하는 값이고, beat와
# downbeat의 confidence 분포가 달라 따로 둘 수 있게 함(FCOS 경로에서도 그랬음).
parser.add_argument('--tau_beat', type=float, default=0.2)
parser.add_argument('--tau_downbeat', type=float, default=0.2)
# stitch_beta_frames: Algorithm 4의 border beta(frame 단위). 논문은 값을 정해주지
# 않고 후보 간격 D/N 정도를 출발점으로 제안함 - D/N = 8 frame(0.186초).
parser.add_argument('--stitch_beta_frames', type=int, default=8)
# 논문 9.2절: 후보 feature들끼리 self-attention을 한 번 돌려서 classification 분기에만
# 먹임 (t_hat은 여전히 z_j에서만 계산되므로 eq.1의 단조성 보장은 그대로).
# [근거] 학습된 체크포인트 실측: downbeat에 매칭된 후보의 p(DB)=0.854, background는
# 0.029로 멀쩡한데, 진짜 beat에 매칭된 후보가 p(DB)=0.141을 갖는다. beat이 downbeat보다
# ~3배 많으므로 이 꼬리가 false downbeat 전부의 출처이고, 이것이 downbeat precision
# 0.430의 원인이다. timing(중앙값 14ms), threshold(9x9 sweep에서 Joint +0.001),
# DP 배정(6.7%), class weighting(gradient 비 0.81) 모두 측정으로 배제됨 - 남은 것은
# "이 박이 마디의 몇 번째인가"가 후보 하나의 국소 feature로는 표현되지 않는다는
# 9.1절의 진단뿐이다. 0이면 비활성(기존과 동일).
parser.add_argument('--class_attention_layers', type=int, default=0)
parser.add_argument('--class_attention_heads', type=int, default=4)

# 8-fold CV처럼 여러 run을 병렬/순차로 돌릴 때 서로 log.log나 checkpoints/를
# 안 덮어쓰게 fold(run)마다 분리할 수 있는 옵션. 기본값은 기존 동작과 동일.
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
parser.add_argument('--log_file', type=str, default='./log.log')

# 원 논문(BeatFCOS) Section 3: "unlike WaveBeat, we did not clip our gradients."
# 근데 지금까지는 fcos 경로에 0.1로 항상 clip이 걸려 있었음 (hungarian 경로
# 안정화하면서 우연히 발견함). None(기본값)이면 기존처럼 자동 선택
# (hungarian=1.0, fcos=0.1) - 이미 검증된 hungarian 쪽은 그대로 두기 위함.
# 0 이하 값을 주면 clipping을 아예 안 함(논문 방식, fcos용).
parser.add_argument('--grad_clip', type=float, default=None)

# 원 논문 Section 3: "we also made each dataset represent 1000 music excerpts
# per epoch... in order to prevent one dataset from dominating another in
# representation". 0이면(기본값) 기존처럼 그냥 4개 데이터셋 전체를 이어붙여서
# 씀(데이터셋 크기 차이로 인한 불균형 있음) - 이 값을 주면 데이터셋마다 매
# epoch 이 개수만큼만 무작위로 뽑아서 균형을 맞춤.
parser.add_argument('--samples_per_dataset', type=int, default=0)

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

# --train_length를 명시 안 했을 때의 head별 기본값 결정.
# subset head는 FPN 레벨 길이가 각각 num_candidates * (8, 4, 2)로 정확히 나누어
# 떨어져야 함 - stride conv는 나머지를 조용히 버리기 때문에(길이 1281도 후보 160개를
# 멀쩡히 내놓으면서 마지막 frame만 잃음) subset_head가 입력 길이를 직접 검증한다.
# 또 홀수 길이는 FPN top-down upsample에서 크기 불일치로 아예 죽는다(1281 -> C3 321을
# 2배 업샘플하면 642 vs C2 641). 그래서 여기서 정확히 맞춰준다.
SUBSET_FRAMES_PER_CANDIDATE = 8  # P1(가장 fine한 레벨)의 stride
if args.lr is None:
    args.lr = 3e-4 if args.head_type == 'subset' else 1e-3
if args.train_length is None:
    if args.head_type == 'subset':
        args.train_length = args.num_candidates * SUBSET_FRAMES_PER_CANDIDATE * args.audio_downsampling_factor
    else:
        args.train_length = 2097152
if args.head_type == 'subset':
    _frames = args.train_length // args.audio_downsampling_factor
    if _frames != args.num_candidates * SUBSET_FRAMES_PER_CANDIDATE:
        raise SystemExit(
            f"[subset] train_length {args.train_length} gives {_frames} mel frames, but "
            f"num_candidates {args.num_candidates} * {SUBSET_FRAMES_PER_CANDIDATE} = "
            f"{args.num_candidates * SUBSET_FRAMES_PER_CANDIDATE} are required. "
            f"Use --train_length {args.num_candidates * SUBSET_FRAMES_PER_CANDIDATE * args.audio_downsampling_factor}.")
    print(f"[subset] window = {_frames} mel frames "
          f"({args.train_length / args.audio_sample_rate:.2f}s at "
          f"{args.audio_sample_rate / args.audio_downsampling_factor:.2f} fps), "
          f"N = {args.num_candidates} candidates, spacing "
          f"{args.train_length / args.audio_sample_rate / args.num_candidates * 1000:.0f}ms")

# resume 여부에 따라 로그 파일을 이어쓸지("a") 새로 쓸지("w") 미리 결정. 원래는
# 무조건 "w"라서, 크래시 후 재시작할 때마다 그동안의 로그(0~56 에폭 등)가
# 통째로 날아가서 매번 수동으로 백업해야 했음.
_is_resuming = len(glob.glob(os.path.join(args.checkpoint_dir, 'retinanet_*.pt'))) > 0
configure_log(args.log_file, mode="a" if _is_resuming else "w")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# carnatic/harmonix: 박사님이 논문에 언급된 데이터셋 중 이미 확보된 것들을
# 추가해서 학습해보자고 제안하셔서 추가함. 둘 다 8-fold CV용 .folds 파일이
# 없어서, dataloader.py의 fallback 로직(파일 없으면 80/10/10 split)이 자동으로
# 적용됨 - validation_fold를 줘도 이 두 데이터셋만은 8-fold CV가 아니라
# 80/10/10으로 나뉘는 점 유의.
datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles", "carnatic", "harmonix", "smc"]

# set the seed
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
args.default_root_dir = os.path.join("lightning_logs", "full")
print(args.default_root_dir)

state_dicts = glob.glob(os.path.join(args.checkpoint_dir, 'retinanet_*.pt'))
start_epoch = 0
checkpoint_path = None

if len(state_dicts) > 0:
    # [무엇] glob.glob의 순서는 파일시스템/OS에 따라 달라질 수 있어서(정렬 보장
    # 없음), state_dicts[-1]이 항상 가장 높은 epoch 번호의 체크포인트라는 보장이
    # 없었음. [왜 문제] 실제로 체크포인트가 20개 넘게 쌓인 checkpoints_fold0_
    # expanded_datasets 재개 시, 최신(96 에폭)이 아니라 훨씬 이전(58 에폭)
    # 체크포인트를 불러오는 사고가 있었음 - epoch 번호를 명시적으로 파싱해서
    # 숫자 기준 최댓값을 고르도록 수정.
    state_dicts.sort(key=lambda path: int(re.search("retinanet_(.*).pt", path).group(1)))
    checkpoint_path = state_dicts[-1]
    start_epoch = int(re.search("retinanet_(.*).pt", checkpoint_path).group(1)) + 1
    print("loaded:" + checkpoint_path)
else:
    print("no checkpoint found")

# setup the dataloaders
train_datasets = []
val_datasets = []
val_dataset_names = []  # val_datasets와 같은 순서로 데이터셋 이름 추적 (아래 macro-average 평가용)

for dataset in datasets:
    if args.dataset_dir is not None:
        audio_dir = os.path.join(args.dataset_dir, dataset, "data")
        annot_dir = os.path.join(args.dataset_dir, dataset, "label")
    else:
        if dataset == "beatles":
            audio_dir = args.beatles_audio_dir
            annot_dir = args.beatles_annot_dir
        elif dataset == "ballroom":
            audio_dir = args.ballroom_audio_dir
            annot_dir = args.ballroom_annot_dir
        elif dataset == "hainsworth" or dataset == "hains":
            audio_dir = args.hainsworth_audio_dir
            annot_dir = args.hainsworth_annot_dir
        elif dataset == "rwc_popular":
            audio_dir = args.rwc_popular_audio_dir
            annot_dir = args.rwc_popular_annot_dir
        elif dataset == "carnatic":
            audio_dir = args.carnatic_audio_dir
            annot_dir = args.carnatic_annot_dir
        elif dataset == "harmonix":
            audio_dir = args.harmonix_audio_dir
            annot_dir = args.harmonix_annot_dir
        elif dataset == "smc":
            audio_dir = args.smc_audio_dir
            annot_dir = args.smc_annot_dir

    if not audio_dir or not annot_dir:
        continue

    if args.backbone_type == "tcn2019":
        # Only if using spectrograms, use the hop length to calculate the audio downsampling factor
        args.audio_downsampling_factor = math.floor(args.hop_length_in_seconds * args.audio_sample_rate)

    train_dataset = BeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    audio_downsampling_factor=args.audio_downsampling_factor,
                                    subset="train",
                                    fraction=args.train_fraction,
                                    augment=args.augment,
                                    half=True,
                                    preload=args.preload,
                                    length=args.train_length,
                                    dry_run=args.dry_run,
                                    spectral=True,  # "True if args.backbone_type == "tcn2019" else False" 제거
                                    validation_fold=args.validation_fold)
    train_datasets.append(train_dataset)

    val_dataset = BeatDataset(audio_dir,
                                 annot_dir,
                                 dataset=dataset,
                                 audio_sample_rate=args.audio_sample_rate,
                                 audio_downsampling_factor=args.audio_downsampling_factor,
                                 subset="val",
                                 augment=False,
                                 half=True,
                                 preload=args.preload,
                                 length=args.eval_length,
                                 dry_run=args.dry_run,
                                 spectral=True,
                                 validation_fold=args.validation_fold)
    val_datasets.append(val_dataset)
    val_dataset_names.append(dataset)

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)

class PerDatasetBalancedSampler(torch.utils.data.Sampler):
    """원 논문 Section 3: "we also made each dataset represent 1000 music
    excerpts per epoch... in order to prevent one dataset from dominating
    another in representation". ConcatDataset을 그냥 이어붙여서 shuffle하면
    큰 데이터셋(ballroom 등)이 작은 데이터셋(hainsworth 등)보다 한 epoch 안에
    훨씬 더 많이 등장해서 대표성이 커짐 - 이 Sampler는 데이터셋마다 정확히
    samples_per_dataset개를 (모자라면 중복 허용해서) 무작위로 뽑아 균등하게
    섞어서, 매 epoch마다 데이터셋 개수 x samples_per_dataset개씩만 나오게 함.
    """
    def __init__(self, datasets, samples_per_dataset):
        self.samples_per_dataset = samples_per_dataset
        # ConcatDataset 안에서 각 sub-dataset이 차지하는 [start, end) 오프셋
        self.offsets = []
        start = 0
        for d in datasets:
            self.offsets.append((start, start + len(d)))
            start += len(d)

    def __iter__(self):
        indices = []
        for start, end in self.offsets:
            n = end - start
            replacement = n < self.samples_per_dataset
            local_indices = torch.randint(0, n, (self.samples_per_dataset,)) if replacement \
                else torch.randperm(n)[:self.samples_per_dataset]
            indices.append(local_indices + start)
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())

    def __len__(self):
        return self.samples_per_dataset * len(self.offsets)

if args.samples_per_dataset > 0:
    train_sampler = PerDatasetBalancedSampler(train_datasets, args.samples_per_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                                    sampler=train_sampler,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    collate_fn=collater)
else:
    train_dataloader = torch.utils.data.DataLoader(train_dataset_list,
                                                    shuffle=args.shuffle,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    collate_fn=collater)
# 데이터셋별로 따로 평가하기 위한 per-dataset val_dataloader들. 기존에는
# 전체를 ConcatDataset으로 합친 pooled val_dataloader 하나만 써서 best
# epoch를 판단했는데, 데이터셋 크기 차이가 크면(harmonix 912곡 vs
# rwc_popular 13곡 등) 큰 데이터셋 성능이 합산 평균을 지배해버려서, 작은
# 데이터셋들이 실제로 나빠지고 있어도 pooled Joint score는 계속 오르는
# 것처럼 보이는 문제가 있었음(실측 확인됨: 원래 4개 데이터셋 downbeat F가
# 큰 폭으로 떨어졌는데 pooled score는 계속 상승). 이제 macro-average
# (데이터셋별 평균의 평균)로 best epoch를 판단함.
per_dataset_val_dataloaders = [
    (name, torch.utils.data.DataLoader(ds, shuffle=False, batch_size=1,
                                        num_workers=args.num_workers,
                                        pin_memory=False, collate_fn=collater))
    for name, ds in zip(val_dataset_names, val_datasets)
]

def evaluate_macro_joint_f_measure(model, label):
    """per_dataset_val_dataloaders를 데이터셋별로 돌려서 macro-average Beat/
    Downbeat/Joint F-measure를 계산. 매 에폭 평가와, resume 직후 불러온
    체크포인트의 진짜 점수를 다시 재는 데 공통으로 씀(중복 제거)."""
    per_dataset_beat_f, per_dataset_downbeat_f = [], []
    for name, loader in per_dataset_val_dataloaders:
        if args.head_type == 'subset':
            # subset head는 고정 길이 window만 받으므로 곡을 타일링해서 디코딩하고
            # Algorithm 4로 이어붙여야 함 (beat_eval.evaluate_beat_f_measure_subset).
            # eval_length는 기존과 같은 2097152(4096 frame)로 두어서, fcos_lite 등과
            # 정확히 같은 구간을 평가하게 함 - 숫자 비교가 가능해야 하므로.
            ds_beat_f, ds_downbeat_f, _ = evaluate_beat_f_measure_subset(
                loader, model, args.audio_downsampling_factor, args.audio_sample_rate,
                window_frames=args.num_candidates * SUBSET_FRAMES_PER_CANDIDATE,
                border_frames=args.stitch_beta_frames,
                threshold_beat=args.tau_beat, threshold_downbeat=args.tau_downbeat)
        else:
            ds_beat_f, ds_downbeat_f, _ = evaluate_beat_f_measure(
                loader, model, args.audio_downsampling_factor, args.audio_sample_rate, score_threshold=0.20)
        per_dataset_beat_f.append(ds_beat_f)
        # [beat-only 데이터셋은 downbeat macro에서 제외] SMC에는 downbeat ground
        # truth가 아예 없어서 downbeat F가 구조적으로 항상 0이다. 이걸 평균에 넣으면
        # "측정 불가능한 값"이 macro를 1/N만큼 상수로 끌어내리고, 그 macro가 바로
        # 체크포인트 저장 기준과 ReduceLROnPlateau의 입력이라 모델 성능과 무관한
        # 상수 페널티로 학습 스케줄이 결정된다(실측 epoch 2: downbeat macro가
        # 0.417 -> 0.348). beat F는 SMC에서도 의미가 있으므로 그대로 포함한다.
        if name in BEAT_ONLY_DATASETS:
            print(f"{label} | [{name}] Beat: {ds_beat_f:0.3f} | Downbeat: n/a (beat-only)")
        else:
            per_dataset_downbeat_f.append(ds_downbeat_f)
            print(f"{label} | [{name}] Beat: {ds_beat_f:0.3f} | Downbeat: {ds_downbeat_f:0.3f}")

    beat_mean_f_measure = float(np.mean(per_dataset_beat_f))
    downbeat_mean_f_measure = float(np.mean(per_dataset_downbeat_f)) if per_dataset_downbeat_f else 0.0
    joint_f_measure = (beat_mean_f_measure + downbeat_mean_f_measure) / 2
    print(f"{label} | Beat score: {beat_mean_f_measure:0.3f} | Downbeat score: {downbeat_mean_f_measure:0.3f} | Joint score: {joint_f_measure:0.3f}")
    return beat_mean_f_measure, downbeat_mean_f_measure, joint_f_measure

def get_training_data_clusters():
    all_beat_lengths = torch.tensor([])
    all_downbeat_lengths = torch.tensor([])

    for data in train_dataset_list:
        audio, annotations = data

        downbeat_annotations = annotations[annotations[:, 2] == 0]
        beat_annotations = annotations[annotations[:, 2] == 1]

        downbeat_lengths = downbeat_annotations[:, 1] - downbeat_annotations[:, 0]
        beat_lengths = beat_annotations[:, 1] - beat_annotations[:, 0]

        all_downbeat_lengths = torch.cat((all_downbeat_lengths, downbeat_lengths))
        all_beat_lengths = torch.cat((all_beat_lengths, beat_lengths))
    
    all_downbeat_lengths_in_secs = all_downbeat_lengths * args.audio_downsampling_factor / args.audio_sample_rate
    all_beat_lengths_in_secs = all_beat_lengths * args.audio_downsampling_factor / args.audio_sample_rate

    _, beat_cluster_centers = kmeans(X=all_beat_lengths_in_secs[:, None], num_clusters=2, device=torch.device('cuda:0'))
    _, downbeat_cluster_centers = kmeans(X=all_downbeat_lengths_in_secs[:, None], num_clusters=3, device=torch.device('cuda:0'))

    all_cluster_centers_Cx1 = torch.cat((beat_cluster_centers, downbeat_cluster_centers), dim=0)
    all_cluster_centers_C = all_cluster_centers_Cx1[:, 0]

    sorted_cluster_centers, _ = torch.sort(all_cluster_centers_C, dim=0)

    return sorted_cluster_centers

dict_args = vars(args)

if __name__ == '__main__':
    # Create the model
    #training_data_clusters = get_training_data_clusters()
    # anchors.py의 pyramid_levels=[0,1,2]가 실제 FPN 레벨 3개(P1/P2/P3)와 일치하는데,
    # 예전 5개 클러스터(beat 2개+downbeat 3개, 원래 5-level FPN을 염두에 두고 설계된
    # 값)를 그대로 쓰면 get_fcos_positives가 레벨 0~2만 순회하기 때문에
    # interval_length_ranges의 뒤 2구간(다운비트 전용, 1.59s~ 이상)이 죽은 코드가
    # 됨 - 실측 결과 실제 downbeat GT의 74.5%가 이 때문에 positive anchor를 하나도
    # 못 받고 있었음(verify_level_assignment_bug.py). beat 클러스터 2개는 그대로
    # 두고 downbeat 3개를 대표값 1개로 합쳐 클러스터 개수(3개)를 실제 레벨 개수와
    # 맞춤 - clusters_to_interval_length_ranges가 마지막 구간을 항상 무한대로
    # 열어두므로, 이렇게 하면 모든 downbeat 길이가 반드시 어떤 레벨에는 걸리게 됨.
    if args.head_type == "fcos_lite":
        # fcos_lite는 P1(레벨0)을 빼고 레벨[1,2] 2개만 쓰므로 클러스터도 2개여야
        # 함 - 기존 3개 중 가장 작은 값(원래 레벨0/1 경계였던 0.42574675)을 빼서
        # 남은 2개 경계가 레벨1/2 범위를 그대로 커버하게 함(model_module.py의
        # head_type="fcos_lite" 관련 주석 참고).
        training_data_clusters = torch.tensor([0.66719675, 1.93286828])
    elif args.head_type == "fcos_no_fpn":
        # FPN ablation(레벨 1개만 사용) - 클러스터가 1개면
        # clusters_to_interval_length_ranges가 전체 구간([-1,1000])을 그대로
        # 반환하도록 이미 손봐놔서, 값 자체(anchor 초기 크기 prior)는 beat와
        # downbeat 사이 중간값 정도면 충분함.
        training_data_clusters = torch.tensor([0.66719675])
    else:
        training_data_clusters = torch.tensor([0.42574675, 0.66719675, 1.93286828])

    beatfcos = model_module.create_beatfcos_model(num_classes=2, clusters=training_data_clusters, args=args, **dict_args)

    if torch.cuda.is_available():
        beatfcos = beatfcos.cuda()
        beatfcos = torch.nn.DataParallel(beatfcos).cuda()
    else:
        beatfcos = torch.nn.DataParallel(beatfcos)

    device = next(beatfcos.module.parameters()).device

    if checkpoint_path:
        beatfcos.load_state_dict(torch.load(checkpoint_path, device))

    beatfcos.training = True
    print(f'[MEM] after model init: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')

    optimizer = torch.optim.Adam(beatfcos.parameters(), lr=args.lr, weight_decay=1e-4) # Default weight decay is 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=args.patience, verbose=True)

    # checkpoint_path와 짝이 되는 optim_{epoch}.pt가 있으면 optimizer/scheduler
    # state도 이어서 복원함 - 없으면(옛날 체크포인트 등) 그냥 새 optimizer로
    # 시작(기존 동작과 동일, 하위호환).
    if checkpoint_path:
        optim_path = checkpoint_path.replace('retinanet_', 'optim_')
        if os.path.exists(optim_path):
            optim_state = torch.load(optim_path, map_location=device)
            optimizer.load_state_dict(optim_state['optimizer'])
            scheduler.load_state_dict(optim_state['scheduler'])
            print(f"optimizer/scheduler state도 복원함: {optim_path}")
        else:
            print(f"optimizer/scheduler state 파일 없음({optim_path}) - 새로 시작")

    loss_hist = collections.deque(maxlen=500)

    beatfcos.train()

    print('Num training images: {}'.format(len(train_dataset_list)))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    highest_joint_f_measure = 0
    if checkpoint_path:
        # 재시작할 때마다 highest_joint_f_measure가 0으로 리셋되면, 불러온
        # 체크포인트의 진짜 점수(예: 56 에폭 Joint 0.864)보다 한참 낮은 점수도
        # "새 기록"으로 착각해서 계속 저장하게 됨 - 불러온 체크포인트를 한 번
        # 재평가해서 진짜 기준점을 세워둠.
        beatfcos.eval()
        _, _, highest_joint_f_measure = evaluate_macro_joint_f_measure(beatfcos, label="Resume check")
        print(f"resume 기준점(highest_joint_f_measure) = {highest_joint_f_measure:0.3f}")

    for epoch_num in range(start_epoch, args.epochs):
        beatfcos.train()

        epoch_loss = []
        cls_losses = []
        reg_losses = []
        lft_losses = []
        adj_losses = []
        pha_losses = []

        print(f'[MEM] epoch {epoch_num} start: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')

        for iter_num, data in enumerate(train_dataloader): #target[:,:,0:2]=interval, target[:,:,2]=class
            audio, target = data  #MJ: audio:shape =(16,1,3000,81); target:shape=(16,128,3)
            if torch.cuda.is_available():
                audio = audio.cuda()
                target = target.cuda()

            try:
                optimizer.zero_grad()

                classification_loss, regression_loss,\
                leftness_loss, adjacency_constraint_loss,\
                phase_loss =\
                    beatfcos((audio, target))

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                leftness_loss = leftness_loss.mean()
                adjacency_constraint_loss = torch.zeros(1).to(adjacency_constraint_loss.device) if args.no_adj else adjacency_constraint_loss.mean()
                phase_loss = phase_loss.mean()

                cls_losses.append(classification_loss.item())
                reg_losses.append(regression_loss.item())
                lft_losses.append(leftness_loss.item())
                adj_losses.append(adjacency_constraint_loss.item())
                pha_losses.append(phase_loss.item())

                loss = classification_loss + regression_loss + leftness_loss + adjacency_constraint_loss + phase_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                # --grad_clip을 명시적으로 안 주면 기존처럼 자동 선택
                # (hungarian=1.0, fcos=0.1). 0 이하로 주면 원 논문 방식대로
                # clipping을 아예 안 함 (자세한 이유는 위 argparse 주석 참고).
                if args.grad_clip is None:
                    clip_norm = 1.0 if args.head_type in ("hungarian", "subset") else 0.1
                else:
                    clip_norm = args.grad_clip

                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(beatfcos.parameters(), clip_norm)

                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # head_type="hungarian"에는 leftness/adjacency 개념 자체가 없음 (anchor
                # 기반 FCOS 전용 개념). REG/LFT 자리는 실제로 bbox L1 / GIoU loss이고
                # ADJ는 아예 안 쓰이므로(model_module.py의 _forward_hungarian 참고),
                # 헷갈리지 않게 라벨을 다르게 출력한다.
                if args.head_type == "subset":
                    # subset head는 anchor가 없어 leftness/adjacency/phase 개념이
                    # 없음. 5-tuple 슬롯의 실제 의미: CLS<-클래스 항, TIME<-시간(L1)
                    # 항, BG<-background 항 (eq. 8), 나머지 둘은 미사용(0).
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | TIME: {:1.5f} | BG: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), np.mean(loss_hist))
                    )
                elif args.head_type == "hungarian":
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | BBOX(L1): {:1.5f} | GIOU: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), np.mean(loss_hist))
                    )
                else:
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | REG: {:1.5f} | LFT: {:1.5f} | ADJ: {:1.5f} | PHA: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), float(adjacency_constraint_loss),
                            float(phase_loss), np.mean(loss_hist))
                    )

                if iter_num % 10 == 0:
                    print(f'[MEM] iter {iter_num}: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB, audio={audio.shape}')

                del classification_loss
                del regression_loss
                del leftness_loss
                del adjacency_constraint_loss
                del phase_loss
                del loss
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                print(e)
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue

        # End of: for iter_num, data in enumerate(train_dataloader)

        # Evaluate the evaluation dataset in each epoch
        # [무성 정지 방지] 학습 iteration이 예외로 전부 건너뛰어지면 optimizer가 한 번도
        # 안 돌아가는데도 epoch 루프와 검증은 계속 돌아서, 얼어붙은 모델의 동일한 점수가
        # 100 epoch까지 찍힌다(실측: subset run이 epoch 74에서 조용히 멈춘 뒤 9,749번
        # 연속 실패, 로그만 보면 정상 종료처럼 보였음). 한 epoch이 통째로 실패하면 즉시
        # 세운다 - 몇 시간을 허비하는 것보다 낫다.
        if len(epoch_loss) == 0:
            raise RuntimeError(
                f"epoch {epoch_num}: every training iteration was skipped (0 optimizer "
                f"steps). Training has stopped; aborting instead of validating a frozen "
                f"model for the remaining epochs.")
        if len(epoch_loss) < 0.5 * len(train_dataloader):
            print(f"[warn] epoch {epoch_num}: only {len(epoch_loss)}/{len(train_dataloader)} "
                  f"iterations succeeded - training is degrading", flush=True)

        print(f'[MEM] before eval: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')
        print('Evaluating dataset')
        beat_mean_f_measure, downbeat_mean_f_measure, joint_f_measure = evaluate_macro_joint_f_measure(
            beatfcos, label=f"Epoch = {epoch_num}")

        if args.head_type == "hungarian":
            print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} | BBOX(L1): {np.mean(reg_losses):0.3f} | GIOU: {np.mean(lft_losses):0.3f}")
        else:
            print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} | REG: {np.mean(reg_losses):0.3f} | LFT: {np.mean(lft_losses):0.3f} | ADJ: {np.mean(adj_losses):0.3f} | PHA: {np.mean(pha_losses):0.3f}")
        scheduler.step(joint_f_measure)

        should_save_checkpoint = False
        if joint_f_measure > highest_joint_f_measure:
            should_save_checkpoint = True
            print(f"Joint score of {joint_f_measure:0.3f} exceeded previous best at {highest_joint_f_measure:0.3f}")
            highest_joint_f_measure = joint_f_measure

        #should_save_checkpoint = True # FOR DEBUGGING
        if should_save_checkpoint:
            new_checkpoint_path = os.path.join(args.checkpoint_dir, 'retinanet_{}.pt'.format(epoch_num))
            print(f"Saving checkpoint at {new_checkpoint_path}")
            torch.save(beatfcos.state_dict(), new_checkpoint_path)
            # 모델 가중치는 evaluate_all_datasets.py 등 다른 스크립트들이 raw
            # state_dict를 그대로 기대해서 형식을 안 바꾸고, optimizer/scheduler
            # state는 별도 파일로 같이 저장함 - resume 시 이 파일이 있으면 이어서
            # 불러와서 LR/momentum이 처음부터 다시 시작되지 않게 함(실제로
            # nhead=8 학습이 eval 중 NaN으로 크래시났을 때 이게 없어서 재시작 후
            # 20 에폭 넘게 이전 best를 못 넘는 정체가 있었음).
            new_optim_path = os.path.join(args.checkpoint_dir, 'optim_{}.pt'.format(epoch_num))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, new_optim_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beatfcos.eval()

    torch.save(beatfcos, os.path.join(args.checkpoint_dir, 'model_final.pt'))
