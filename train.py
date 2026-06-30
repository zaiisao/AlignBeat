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

def configure_log():
    log_file_name = ospj("./", 'log.log')
    Logger(log_file_name)

configure_log()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

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

state_dicts = glob.glob('./checkpoints/*.pt')
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
    training_data_clusters = torch.tensor([0.42574675, 0.66719675, 1.24245649, 1.93286828, 2.78558922])

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

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

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

                torch.nn.utils.clip_grad_norm_(beatfcos.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

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
        print(f"Epoch = {epoch_num} | CLS: {np.mean(cls_losses):0.3f} | REG: {np.mean(reg_losses):0.3f} | LFT: {np.mean(lft_losses):0.3f} | ADJ: {np.mean(adj_losses):0.3f}")
        scheduler.step(joint_f_measure)

        should_save_checkpoint = False
        if joint_f_measure > highest_joint_f_measure:
            should_save_checkpoint = True
            print(f"Joint score of {joint_f_measure:0.3f} exceeded previous best at {highest_joint_f_measure:0.3f}")
            highest_joint_f_measure = joint_f_measure

        #should_save_checkpoint = True # FOR DEBUGGING
        if should_save_checkpoint:
            new_checkpoint_path = './checkpoints/retinanet_{}.pt'.format(epoch_num)
            print(f"Saving checkpoint at {new_checkpoint_path}")
            torch.save(beatfcos.state_dict(), new_checkpoint_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    beatfcos.eval()

    torch.save(beatfcos, './checkpoints/model_final.pt')
