"""Step through decode on one cached piece: no GPU, no dataloader, no Lightning.

The cache is whole-piece candidate arrays dumped from fold 0, so this is the same input
subset_predict_piece hands the decoder -- but it loads in a second, which makes it the
place to set breakpoints in _elapsed_beats and _offset_transition.

PIECE picks which one. A few worth knowing:
    asap_Bach_Fugue_bwv_862_Song04M_track.npy   ordinary, 4/4, clean detection
    smc_smc_212_track.npy                       3 detections for 26 true beats
    asap_Beethoven_Piano_Sonatas_17-1_no_repeat_WangH03M_track.npy   ties in the Viterbi
"""
import numpy as np
import torch

from alignbeat.constants import CLASS_DOWNBEAT
from alignbeat.inference.decode import decode, _elapsed_beats

CACHE = "/tmp/promote/cache"
PIECE = "asap_Bach_Fugue_bwv_862_Song04M_track.npy"
METER_PRIOR = {2: 0.0992763854571126, 3: 0.09468761030709495, 4: 0.7942993293328627,
               5: 0.0009707024355806565, 6: 0.009265795975997176, 8: 0.0015001764913519238}


def main():
    cached = np.load(f"{CACHE}/{PIECE}.npz", allow_pickle=True)
    probabilities = torch.as_tensor(cached["probs"].astype(np.float64))
    seconds = torch.as_tensor(cached["seconds"].astype(np.float64))

    # decode softmaxes its input, so hand it logs of the cached probabilities.
    class_logits = probabilities.clamp_min(1e-300).log()
    classes, times, _scores = decode(class_logits, seconds, METER_PRIOR,
                                     tau=0.5, read_out="duration")

    downbeats = times[classes == CLASS_DOWNBEAT].numpy()
    truth = cached["truth_downbeat"]
    print(f"{PIECE}  ({cached['dataset']})")
    print(f"  candidates {len(seconds)}, detected {len(times)}, downbeats {len(downbeats)}")
    print(f"  true downbeats {len(truth)}")

    advances = _elapsed_beats(times, 64, 3)
    for advance in advances.unique().tolist():
        count = int((advances == advance).sum())
        print(f"  advance {advance}: {count} steps ({100 * count / len(advances):.1f}%)")

    print(f"  first 8 predicted downbeats {np.round(downbeats[:8], 2)}")
    print(f"  first 8 true downbeats      {np.round(truth[:8], 2)}")


if __name__ == "__main__":
    main()
