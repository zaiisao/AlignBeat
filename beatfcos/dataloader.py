import os
import glob
import torch 
import julius
import random
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm
# import soxbindings as sox 
import sox

torchaudio.set_audio_backend("sox_io")

def collater(data):
    # data = one batch of [audio, annot(, metadata)]
    audios = [s[0] for s in data]  #MJ: s[0]:  shape = (1, 3000, 81), a single channel 2D tensor for spectrogram input
                                   #MJ: s[0]: shape = = (1,N) = (1,  num of audio samples) for raw audio
    annots = [s[1] for s in data]  #MJ: s[1]:   shape = (M,3)=(num of beat intervals in spectoram frame unit,3)=(57,3) for spectrogram input;
                                   #              shape = (M,3)= (num of beat intervals in target base-level sample unit, 3)
    metadata = None

    if len(data[0]) > 2:
        metadata = [s[2] for s in data]

    new_audios = torch.stack(audios) # new_audios shape: (B, C, W) = (B, 1, num of audio samples) in wavebeat
                                     #  new_audios shape: (B,C,H,W) =(B,1,3000, 81) for spectrogram input (single channel 2D tensor)

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        new_annots = torch.ones((len(annots), max_num_annots, 3)) * -1  # new_annots shape: (B, max_num_annots, 3) = (B, W, C)
                                                                        # in PyTorch, input 2D tensors are written as (B, C, H, W)
                                                                        #             input 1D tensors are written as (B, C, W)
                                                                        # whereas the target or annotations are written as (B, H, W, C)
                                                                        # new_annots[B, j, 2] = -1, which means jth element is  zero padded-element

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    new_annots[idx, :annot.shape[0], :] = annot  #MJ: annot.shape[0] = num of beat locations; new_annots shape: (B, M, 3)
    else:
        new_annots = torch.ones((len(annots), 1, 3)) * -1 # new_annots shape: (B, 1, 3)

    #return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
    if metadata is not None:
        return new_audios, new_annots, metadata

    return new_audios, new_annots

class BeatDataset(torch.utils.data.Dataset):
    """ Downbeat Dataset. """
    def __init__(self, 
                 audio_dir, 
                 annot_dir, 
                 audio_sample_rate=44100, 
                 audio_downsampling_factor=32,
                 dataset="ballroom",
                 subset="train", 
                 length=16384, #MJ: = 3000 frames in the case of spectrogram input
                 preload=False, 
                 half=True, 
                 fraction=1.0,
                 augment=False,
                 dry_run=False,
                 pad_mode='constant',
                 examples_per_epoch=1000,
                 spectral=False,
                 trim_size=(81, 3000),
                 validation_fold=None):
        """
        Args:
            audio_dir (str): Path to the root directory containing the audio (.wav) files.
            annot_dir (str): Path to the root directory containing the annotation (.beats) files.
            audio_sample_rate (float, optional): Sample rate of the audio files. (Default: 44100)
            audio_downsampling_factor (float, optional): The factor by which to downsample the sample rate of the audio to the final output tensor. (Default: 256)
            subset (str, optional): Pull data either from "train", "val", "test", or "full-train", "full-val" subsets. (Default: "train")
            dataset (str, optional): Name of the dataset to be loaded "ballroom", "beatles", "hainsworth", "rwc_popular", "gtzan", "smc". (Default: "ballroom")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            augment (bool, optional): Apply random data augmentations to input audio. (Default: False)
            dry_run (bool, optional): Train on a single example. (Default: False)
            pad_mode (str, optional): Padding type for inputs 'constant', 'reflect', 'replicate' or 'circular'. (Default: 'constant')
            examples_per_epoch (int, optional): Number of examples to sample from the dataset per epoch. (Default: 1000)

        Notes:
            - The SMC dataset contains only beats (no downbeats), so it should be used only for beat evaluation.
        """
        
        #MJ: spectrogram)t,w) = SFFT(t,w)^2
        self.audio_dir = audio_dir
        self.annot_dir = annot_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_downsampling_factor = audio_downsampling_factor
        
        self.target_sample_rate = audio_sample_rate / audio_downsampling_factor
        #Audio_downsamplig_factor is different depending on whether we use spectrogrma or raw audio as the input
        
        self.subset = subset
        self.dataset = dataset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.augment = augment
        self.dry_run = dry_run
        self.pad_mode = pad_mode
        self.dataset = dataset
        self.examples_per_epoch = examples_per_epoch
        self.spectral = spectral
        self.trim_size = trim_size
        self.validation_fold = validation_fold

        self.target_length = int(self.length / self.audio_downsampling_factor)
        #print(f"Audio length: {self.length}")
        #print(f"Target length: {self.target_length}")

        # first get all of the audio files
        #if self.dataset in ["beatles", "rwc_popular"]:
        # if self.dataset in ["rwc_popular"]:
        #     file_ext = "*L+R.wav"
        # elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc", "beatles", "carnatic"]:
        if self.dataset in ["ballroom", "hainsworth", "gtzan", "smc", "beatles", "carnatic", "rwc_popular", "harmonix"]:
            file_ext = "*.wav"
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")

        # 각 annot_dir에 "*_8-fold_cv_*.folds"와 "*_80_10_10.folds"가 둘 다
        # 있어서, 예전처럼 "*.folds"로 다 긁으면 glob 순서에 따라 8-fold CV용
        # 파일 대신 80/10/10용 파일이 잘못 선택될 수 있었음(둘 다 있을 때
        # 어느 게 먼저 나올지 보장이 안 됨). validation_fold를 실제 8-fold CV로
        # 쓰려는 것이므로 "8-fold"가 파일명에 들어간 것만 명시적으로 찾음.
        fold_files = glob.glob(os.path.join(self.annot_dir, "*8-fold*.folds"))  #MJ: /mount/beat-tracking/ballroom/label/???.folds etc
        if self.validation_fold is not None and len(fold_files) > 0 and self.subset in ["train", "val", "test"]:
            fold_file = fold_files[0]  #MJ: len(fold_files) = 1
            self.audio_files = []

            k = 8 # JA: k = 8 is the standard in beat tracking

            with open(fold_file, 'r') as fp:
                lines = fp.readlines()

                for line in lines:
                    line = line.strip('\n')
                    line = line.replace('\t', ' ')
                    audio_filename, fold_number = line.split(' ')
                    audio_filename_start = len(self.dataset) + 1 # Each line in a .folds file starts with "DATASET_"

                    test_fold = (validation_fold + 1) % k
                    is_valid_and_training = \
                        self.subset == "train" \
                        and validation_fold != int(fold_number) \
                        and test_fold != int(fold_number) #MJ: is the file for training

                    is_valid_and_validation = \
                        self.subset == "val" and \
                        validation_fold == int(fold_number) #MJ: is the file for  validation or testing????

                    is_valid_and_test = \
                        self.subset == "test" \
                        and test_fold == int(fold_number)

                    if is_valid_and_training or is_valid_and_validation or is_valid_and_test:
                        audio_file_path = os.path.join(self.audio_dir, audio_filename[audio_filename_start:] + ".wav")
                        if not os.path.isfile(audio_file_path): #MJ: audio_file_path is not a file? If recursive is true, the pattern “**” will match any files and zero or more directories and subdirectories. If the pattern is followed by an os.sep, only directories and subdirectories match.
                            audio_file_paths = glob.glob(os.path.join(self.audio_dir, "**", audio_filename[audio_filename_start:] + ".wav"), recursive=True)
                            if len(audio_file_paths) > 0:
                                audio_file_path = audio_file_paths[0]

                        if os.path.isfile(audio_file_path):
                            self.audio_files.append(audio_file_path)
                        else:
                            print(f"{audio_file_path} not found; skipping")

            #random.shuffle(self.audio_files) # shuffle them: using random()
        else: #MJ: the original version of wavebeat
            self.audio_files = glob.glob(os.path.join(self.audio_dir, "**", file_ext))
            if len(self.audio_files) == 0: # try from the root audio dir
                self.audio_files = glob.glob(os.path.join(self.audio_dir, file_ext))

            self.audio_files = sorted(self.audio_files)

            random1 = random.Random(4)
            random1.shuffle(self.audio_files)
            #random.shuffle(self.audio_files) # shuffle them

            if self.subset in ["train", "train_with_metadata"]:
                start = 0
                stop = int(len(self.audio_files) * 0.8)
            elif self.subset == "val":
                start = int(len(self.audio_files) * 0.8)
                stop = int(len(self.audio_files) * 0.9)
            elif self.subset == "test":
                start = int(len(self.audio_files) * 0.9)
                stop = None
            elif self.subset in ["full-train", "full-val"]:
                start = 0
                stop = None

        # select one file for the dry run
        if self.dry_run: 
            self.audio_files = [self.audio_files[0]] * 50
            print(f"Selected 1 file for dry run.")
        else:
            # now pick out subset of audio files
            if self.validation_fold is None:
                self.audio_files = self.audio_files[start:stop]
                print(f"Selected {len(self.audio_files)} files for {self.subset} set from {self.dataset} dataset.")

            print(self.audio_files)

        self.annot_files = []
        for audio_file in self.audio_files:
            # find the corresponding annot file
            # if self.dataset in ["rwc_popular", "beatles"]:
            #     replace = "_L+R.wav"
            # elif self.dataset in ["ballroom", "hainsworth", "gtzan", "smc", "carnatic"]:
            replace = ".wav"
            
            filename = os.path.basename(audio_file).replace(replace, "")

            if self.dataset == "ballroom":
                self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.beats"))
            elif self.dataset == "hainsworth":
                self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.txt"))
            elif self.dataset == "beatles":
                #album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, f"{filename}.txt")
                self.annot_files.append(annot_file)
            elif self.dataset == "rwc_popular":
                album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, f"{filename}.BEAT.TXT")
                self.annot_files.append(annot_file)
            elif self.dataset == "gtzan":
                # GTZAN dataset audio is named as "NUMBER1_GENRE.NUMBER2.wav"
                # GTZAN dataset annot is named as "gtzan_GENRE_NUMBER2.wav"
                # NUMBER1 always four digits and is only in the audio name, so we remove it
                # (Notice the difference of . and _ so we replace only the first instance of . with _)
                annot_file_name = f"gtzan{filename[4:].replace('.', '_', 1)}.beats"
                annot_file = os.path.join(self.annot_dir, annot_file_name)
                self.annot_files.append(annot_file)
            elif self.dataset == "smc":
                annot_filepath = os.path.join(self.annot_dir, f"{filename}*.txt")
                annot_file = glob.glob(annot_filepath)[0]
                self.annot_files.append(annot_file)
            elif self.dataset == "carnatic":
                self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.beats"))
            elif self.dataset == "harmonix":
                self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.txt"))

        self.data = [] # when preloading store audio data and metadata
        if self.preload:
            for audio_filename, annot_filename in tqdm(zip(self.audio_files, self.annot_files), 
                                                        total=len(self.audio_files), 
                                                        ncols=80):
                    audio, target, metadata = self.load_data(audio_filename, annot_filename)
                    if audio.isnan().sum() > 0: raise ValueError
                    if self.half:
                        if not self.spectral:
                            audio = audio.half()
                        target = target.half()
                    self.data.append((audio, target, metadata))

    def __len__(self):
        if self.spectral:
            if self.subset in ["test", "val", "full-val", "full-test", "train_with_metadata"]:
                return len(self.audio_files)
            else:
                return self.examples_per_epoch

        if self.subset in ["test", "val", "full-val", "full-test", "train_with_metadata"]:
            length = len(self.audio_files)
        else:
            length = self.examples_per_epoch
        return length
        #return len(self.audio_files)

    def __getitem__(self, idx): #MJ: this function does annot = self.make_intervals(target)

        if self.preload:
            audio, target, metadata = self.data[idx % len(self.audio_files)]
        else:
            # get metadata of example
            audio_filename = self.audio_files[idx % len(self.audio_files)]
            annot_filename = self.annot_files[idx % len(self.audio_files)]
            audio, target, metadata = self.load_data(audio_filename, annot_filename)

        # apply augmentations 
        if self.spectral:
            T = audio.shape[0]
            target_len = self.target_length  # = train_length / audio_downsampling_factor = 2097152/512 = 4096

            if T > target_len:
                if self.subset in ['train', 'full-train']:
                    start = np.random.randint(0, T - target_len)
                    audio = audio[start:start + target_len, :]
                    target = target[:, start:start + target_len].float()
                else:
                    audio = audio[:target_len, :]
                    target = target[:, :target_len].float()
            else:
                pad_size = target_len - T
                audio = torch.nn.functional.pad(audio, (0, 0, 0, pad_size))
                target = torch.nn.functional.pad(target.float(), (0, pad_size))


        else:
            audio_old = audio.float()
            target = target.float()

            if self.augment:
                audio, target = self.apply_augmentations(audio_old, target)

            N_audio = audio.shape[-1]
            N_target = target.shape[-1]

            if (N_audio > self.length or N_target > self.target_length) and self.subset not in ['val', 'test', 'full-val']:
                audio_start = np.random.randint(0, N_audio - self.length - 1)
                audio_stop  = audio_start + self.length
                target_start = int(audio_start / self.audio_downsampling_factor)
                target_stop = int(audio_stop / self.audio_downsampling_factor)
                audio = audio[:, audio_start:audio_stop]
                target = target[:, target_start:target_stop]

            if audio.shape[-1] < self.length and self.subset not in ['val', 'test', 'full-val']:
                pad_size = self.length - audio.shape[-1]
                padl = pad_size - (pad_size // 2)
                padr = pad_size // 2
                audio = torch.nn.functional.pad(audio, (padl, padr), mode=self.pad_mode)

            if target.shape[-1] < self.target_length and self.subset not in ['val', 'test', 'full-val']:
                pad_size = self.target_length - target.shape[-1]
                padl = pad_size - (pad_size // 2)
                padr = pad_size // 2
                target = torch.nn.functional.pad(target, (padl, padr), mode=self.pad_mode)
        
        annot = self.make_intervals(target)  ##MJ: # target: shape =(2,3000)2402; annot: shape =(M,3)=(57,3)

        if self.subset in ["train", "full-train"]:
            return audio, annot
        elif self.subset in ["train_with_metadata", "val", "test", "full-val"]:
            # this will only work with batch size = 1
            return audio, annot, metadata
        else:
            raise RuntimeError(f"Invalid subset: `{self.subset}`")
    #END def __getitem__(self, idx)
    
    def load_data(self, audio_filename, annot_filename):
  
       
        # first load the audio file
        #MJ: audio has several roles in computing the target beat locations in DataSet,
        # Override audio with  spectrogram at the end of load_data().
        #         
        audio, sr = torchaudio.load(audio_filename) #MJ: auido: shape=(1,1329330), sr = 44100
        audio = audio.float()

        # resample if needed
        if sr != self.audio_sample_rate:
            audio = julius.resample_frac(audio, sr, self.audio_sample_rate)   

        # convert to mono by averaging the stereo; in_ch becomes 1
        if len(audio) == 2:
            #print("WARNING: Audio is not mono")
            audio = torch.mean(audio, dim=0).unsqueeze(0)  #MJ: audio: shape =(1,N); do we need to get the mono version for gettting spectrogram? No, spectrogram allows stereo
            #MJ: audio: shape =(1,664665)
        # normalize all audio inputs -1 to 1
        audio /= audio.abs().max()
        #END of if self.spectral
        
        # now get the beat location annotation in seconds
        
        annot = self.load_annot(annot_filename)
        beat_samples, downbeat_samples, beat_indices, time_signature = annot #len(beat_samples)=88; len(downbeat_samples)=22

        # get metadata
        genre = os.path.basename(os.path.dirname(audio_filename))

        # convert beat_samples in 22050Hz to beat_seconds
        beat_sec = np.array(beat_samples) / self.audio_sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.audio_sample_rate

        T = audio.shape[-1]/self.audio_sample_rate # audio length in sec: 30.14 sec
        
        N = int(T * self.target_sample_rate) + 1   # target length in samples= samples/sec: 3022
                                                   # MJ: target length in spectrogram frames in the case of spectrogram input, where
                                                   # MJ: target_sample_rate = audio_sample_rate / audio_downsampling_factor = (22050/s) / 200 = 110 /s
                                                   #MJ: in the case of raw audio: target_sample_rate = 22050/s / 128 = 172 /s => Spectrogram downsamples less than wavebeat!
        target = torch.zeros(2,N)

        # now convert from seconds to new sample rate / spectrogram frames
        beat_samples = np.array(beat_sec * self.target_sample_rate)
        downbeat_samples = np.array(downbeat_sec * self.target_sample_rate)

        # check if there are any beats beyond the file end
        beat_samples = beat_samples[beat_samples < N]
        downbeat_samples = downbeat_samples[downbeat_samples < N]

        beat_samples = beat_samples.astype(int)
        downbeat_samples = downbeat_samples.astype(int)

        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        metadata = {
            "Filename" : audio_filename,
            "Genre" : genre,
            "Time signature" : time_signature
        }

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = self.audio_sample_rate,
            n_fft = 2048,
            hop_length = 512,
            n_mels = 128,
            f_min = 30.0,
            f_max = 11000.0
        )

        mel= mel_transform(audio)
        audio = torch.log1p(mel.squeeze(0).T)
        
        return audio, target, metadata  #MJ: audio may be raw audio or its spectrogram

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()

        beat_samples = [] # array of samples containing beats
        downbeat_samples = [] # array of samples containing downbeats (1)
        beat_indices = [] # array of beat type one-hot encoded  
        time_signature = None # estimated time signature (only 3/4 or 4/4)

        for line in lines:
            if self.dataset == "ballroom":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "beatles":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                line = line.replace('  ', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "hainsworth":
                line = line.strip('\n')
                time_sec, beat = line.split(' ')
            elif self.dataset == "rwc_popular":
                line = line.strip('\n')
                line = line.split('\t')
                time_sec = int(line[0]) / 100.0
                beat = 1 if int(line[2]) == 384 else 2
            elif self.dataset == "gtzan":
                line = line.strip('\n')
                time_sec, beat = line.split('\t')
            elif self.dataset == "smc":
                line = line.strip('\n')
                time_sec = line
                beat = 1
            elif self.dataset == "carnatic":
                line = line.strip('\n')
                time_sec, beat = line.split(',')
            elif self.dataset == "harmonix":
                # 형식: "time\tbeat_number\tbar_number" (beat_number가 1일 때
                # 마디 시작=downbeat, 다른 데이터셋과 동일 컨벤션). bar_number는
                # 안 씀.
                line = line.strip('\n')
                time_sec, beat, _bar = line.split('\t')

            # convert beat to one-hot
            beat = int(beat)
            if beat == 1:
                beat_one_hot = [1,0,0,0]
            elif beat == 2:
                beat_one_hot = [0,1,0,0]
            elif beat == 3:
                beat_one_hot = [0,0,1,0]    
            elif beat == 4:
                beat_one_hot = [0,0,0,1]

            # convert seconds to samples
            beat_time_samples = int(float(time_sec) * (self.audio_sample_rate))

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat)

            if beat == 1:
                downbeat_time_samples = int(float(time_sec) * (self.audio_sample_rate))
                downbeat_samples.append(downbeat_time_samples)

        # guess at the time signature
        if np.max(beat_indices) == 2:
            time_signature = "2/4"
        elif np.max(beat_indices) == 3:
            time_signature = "3/4"
        elif np.max(beat_indices) == 4:
            time_signature = "4/4"
        else:
            time_signature = "?"

        return beat_samples, downbeat_samples, beat_indices, time_signature

    def make_intervals(self, target): #MJ: target: shape = (2,N)
        beats = target[0, :]
        downbeats = target[1, :]
        #non_downbeats = beats - downbeats

        beat_locations = torch.nonzero(beats).squeeze()
        downbeat_locations = torch.nonzero(downbeats).squeeze()
        #non_downbeat_locations = torch.nonzero(non_downbeats).squeeze()

        # equivalent code in retinanet "load_annotations" function in dataloader
        annotations = torch.zeros((0, 3))

        # some audio can miss annotations
        # interval을 만드려면 한 리스트에 2개 이상이 있어야 함
        #if downbeat_locations.size(dim=0) < 2 or non_downbeat_locations.size(dim=0) < 2:
        try:
            if downbeat_locations.size(dim=0) < 2 or beat_locations.size(dim=0) < 2:
                return annotations
        except IndexError:
            return annotations
        
        # parse annotations
        def make_interval_subset(samples, class_id):
            
            intervals = torch.zeros((0, 3))
            
            for beat_index, current_beat_location in enumerate(samples[:-1]):
                # next downbeat location 또는 next beat location
                next_beat_location = samples[beat_index + 1]

                interval = torch.zeros((1, 3))
                interval[0, 0] = current_beat_location
                interval[0, 1] = next_beat_location
                interval[0, 2] = class_id

                intervals = torch.cat((intervals, interval), axis=0)

            return intervals  #MJ: shape =(M,3)

        annotations = torch.cat((
            annotations,
            make_interval_subset(downbeat_locations, 0),
            #make_interval_subset(non_downbeat_locations, 1)
            make_interval_subset(beat_locations, 1)
        ), axis=0)

        return annotations

    def apply_augmentations(self, audio, target):

        # random gain from 0dB to -6 dB
        #if np.random.rand() < 0.2:      
        #    #sgn = np.random.choice([-1,1])
        #    audio = audio * (10**((-1 * np.random.rand() * 6)/20))   

        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                              

        # drop continguous frames
        if np.random.rand() < 0.05:     
            zero_size = int(self.length*0.1)
            
            audio_start = np.random.randint(audio.shape[-1] - zero_size - 1)
            audio_stop = audio_start + zero_size
            
            target_start = audio_start // self.audio_downsampling_factor
            target_stop = audio_stop // self.audio_downsampling_factor

            audio[:,audio_start:audio_stop] = 0
            target[:,target_start:target_stop] = 0

        # shift targets forward/back max 70ms
        if np.random.rand() < 0.3:      
            
            # in this method we shift each beat and downbeat by a random amount
            max_shift = int(0.045 * self.target_sample_rate)

            beat_ind = torch.logical_and(target[0,:] == 1, target[1,:] != 1).nonzero(as_tuple=False) # all beats EXCEPT downbeats
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)

            # shift just the downbeats
            dbeat_shifts = torch.normal(0.0, max_shift/2, size=(1,dbeat_ind.shape[-1]))
            dbeat_ind += dbeat_shifts.long()

            # now shift the non-downbeats 
            beat_shifts = torch.normal(0.0, max_shift/2, size=(1,beat_ind.shape[-1]))
            beat_ind += beat_shifts.long()

            # ensure we have no beats beyond max index
            beat_ind = beat_ind[beat_ind < target.shape[-1]]
            dbeat_ind = dbeat_ind[dbeat_ind < target.shape[-1]]  

            # now convert indices back to target vector
            shifted_target = torch.zeros(2,target.shape[-1])
            shifted_target[0,beat_ind] = 1
            shifted_target[0,dbeat_ind] = 1 # set also downbeats on first channel
            shifted_target[1,dbeat_ind] = 1

            target = shifted_target
    
        # apply pitch shifting
        if np.random.rand() < 0.5:
            sgn = np.random.choice([-1,1])
            factor = sgn * np.random.rand() * 8.0     
            tfm = sox.Transformer()        
            tfm.pitch(factor)
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a lowpass filter
        if np.random.rand() < 0.1:
            cutoff = (np.random.rand() * 4000) + 4000
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="lowpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a highpass filter
        if np.random.rand() < 0.1:
            cutoff = (np.random.rand() * 1000) + 20
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="highpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a chorus effect
        if np.random.rand() < 0.05:
            tfm = sox.Transformer()        
            tfm.chorus()
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a compressor effect
        if np.random.rand() < 0.15:
            attack = (np.random.rand() * 0.300) + 0.005
            release = (np.random.rand() * 1.000) + 0.3
            tfm = sox.Transformer()        
            tfm.compand(attack_time=attack, decay_time=release)
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply an EQ effect
        if np.random.rand() < 0.15:
            freq = (np.random.rand() * 8000) + 60
            q = (np.random.rand() * 7.0) + 0.1
            g = np.random.normal(0.0, 6)  
            tfm = sox.Transformer()        
            tfm.equalizer(frequency=freq, width_q=q, gain_db=g)
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # add white noise
        if np.random.rand() < 0.05:
            wn = (torch.rand(audio.shape) * 2) - 1
            g = 10**(-(np.random.rand() * 20) - 12)/20
            audio = audio + (g * wn)

        # apply nonlinear distortion 
        if np.random.rand() < 0.2:   
            g = 10**((np.random.rand() * 12)/20)   
            audio = torch.tanh(audio)    
        
        # normalize the audio
        audio /= audio.float().abs().max()

        return audio, target

    def num_classes():
        return 2
