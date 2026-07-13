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
from beatfcos.dataloader import BeatDataset, collater
from beatfcos.beat_eval import evaluate_beat_f_measure

class Logger(object):
    """Log stdout messages."""
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

def configure_log(log_file_name):
    Logger(log_file_name)

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
parser.add_argument('--preload', default=False, action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=22050)
parser.add_argument('--audio_downsampling_factor', type=int, default=512)  # 128 → 512 (hop_length)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=2097152)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=2097152)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', default=True, action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
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
parser.add_argument('--pretrained', default=False, action="store_true")  # True → False
parser.add_argument('--freeze_backbone', default=False, action="store_true")
parser.add_argument('--centerness', default=False, action="store_true")
parser.add_argument('--postprocessing_type', type=str, default='soft_nms')
parser.add_argument('--no_adj', default=False, action="store_true")
parser.add_argument('--validation_fold', type=int, default=None)
parser.add_argument('--backbone_type', type=str, default="wavebeat")
parser.add_argument('--hop_length_in_seconds', type=float, default=0.01) # This is from Spectral TCN
parser.add_argument('--dmodel', type=int, default=128)
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--d_hid', type=int, default=512)
parser.add_argument('--nlayers', type=int, default=9)
parser.add_argument('--attn_len', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.1)
# Soft-NMS 후처리를 없앤 축소판 RT-DETR 헤드(순서 보존 매칭 기반, 자세한 내용은
# beatfcos/hungarian_head.py 참고)를 쓰려면 --head_type hungarian으로 지정.
# 기본값 'fcos'는 기존 anchor+Soft-NMS 파이프라인을 그대로 유지함(비교용).
parser.add_argument('--head_type', type=str, default='fcos', choices=['fcos', 'hungarian'])
parser.add_argument('--num_queries', type=int, default=300)
parser.add_argument('--decoder_layers', type=int, default=3)

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

configure_log(args.log_file)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]
datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]
#MJ: for testing: datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]

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

state_dicts = glob.glob(os.path.join(args.checkpoint_dir, '*.pt'))
start_epoch = 0
checkpoint_path = None

if len(state_dicts) > 0:
    checkpoint_path = state_dicts[-1]
    start_epoch = int(re.search("retinanet_(.*).pt", checkpoint_path).group(1)) + 1
    print("loaded:" + checkpoint_path)
else:
    print("no checkpoint found")

# setup the dataloaders
train_datasets = []
val_datasets = []

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

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)

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
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False,
                                            collate_fn=collater)

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
    loss_hist = collections.deque(maxlen=500)

    beatfcos.train()

    print('Num training images: {}'.format(len(train_dataset_list)))

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    highest_joint_f_measure = 0

    for epoch_num in range(start_epoch, args.epochs):
        beatfcos.train()

        epoch_loss = []
        cls_losses = []
        reg_losses = []
        lft_losses = []
        adj_losses = []

        print(f'[MEM] epoch {epoch_num} start: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')

        for iter_num, data in enumerate(train_dataloader): #target[:,:,0:2]=interval, target[:,:,2]=class
            audio, target = data  #MJ: audio:shape =(16,1,3000,81); target:shape=(16,128,3)
            if torch.cuda.is_available():
                audio = audio.cuda()
                target = target.cuda()

            try:
                optimizer.zero_grad()

                classification_loss, regression_loss,\
                leftness_loss, adjacency_constraint_loss =\
                    beatfcos((audio, target))

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                leftness_loss = leftness_loss.mean()
                adjacency_constraint_loss = torch.zeros(1).to(adjacency_constraint_loss.device) if args.no_adj else adjacency_constraint_loss.mean()

                cls_losses.append(classification_loss.item())
                reg_losses.append(regression_loss.item())
                lft_losses.append(leftness_loss.item())
                adj_losses.append(adjacency_constraint_loss.item())

                loss = classification_loss + regression_loss + leftness_loss + adjacency_constraint_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                # --grad_clip을 명시적으로 안 주면 기존처럼 자동 선택
                # (hungarian=1.0, fcos=0.1). 0 이하로 주면 원 논문 방식대로
                # clipping을 아예 안 함 (자세한 이유는 위 argparse 주석 참고).
                if args.grad_clip is None:
                    clip_norm = 1.0 if args.head_type == "hungarian" else 0.1
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
                if args.head_type == "hungarian":
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | BBOX(L1): {:1.5f} | GIOU: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), np.mean(loss_hist))
                    )
                else:
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | REG: {:1.5f} | LFT: {:1.5f} | ADJ: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), float(adjacency_constraint_loss), np.mean(loss_hist))
                    )

                if iter_num % 10 == 0:
                    print(f'[MEM] iter {iter_num}: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB, audio={audio.shape}')

                del classification_loss
                del regression_loss
                del leftness_loss
                del adjacency_constraint_loss
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
        print(f'[MEM] before eval: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB, reserved={torch.cuda.memory_reserved()/1e9:.3f}GB')
        print('Evaluating dataset')
        beat_mean_f_measure, downbeat_mean_f_measure, _ = evaluate_beat_f_measure(
            val_dataloader, beatfcos, args.audio_downsampling_factor, args.audio_sample_rate, score_threshold=0.20)
        
        joint_f_measure = (beat_mean_f_measure + downbeat_mean_f_measure)/2

        print(f"Epoch = {epoch_num} | Beat score: {beat_mean_f_measure:0.3f} | Downbeat score: {downbeat_mean_f_measure:0.3f} | Joint score: {joint_f_measure:0.3f}")
        if args.head_type == "hungarian":
            print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} | BBOX(L1): {np.mean(reg_losses):0.3f} | GIOU: {np.mean(lft_losses):0.3f}")
        else:
            print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} | REG: {np.mean(reg_losses):0.3f} | LFT: {np.mean(lft_losses):0.3f} | ADJ: {np.mean(adj_losses):0.3f}")
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beatfcos.eval()

    torch.save(beatfcos, os.path.join(args.checkpoint_dir, 'model_final.pt'))
