"""Batch collation and the shared beat-only annotation convention.

The raw-audio loader that used to live here is gone: training and evaluation both run
on the Beat This spectrogram corpus (alignbeat/bt_dataset.py), which reads pre-computed
50 fps spectrograms. What remains is the collater both paths share and the two
constants that define how a beat-only annotation is represented.
"""
import torch

# Datasets that ship beat annotations only, with no downbeat labels. Every event from
# such a dataset is emitted as class_id=CLASS_BEAT_ONLY, which the head treats as
# "some beat, downbeat status unobserved" -- the B* label of Section 2, routed through
# the beat-only training loss of Section 8 rather than asserted to be a non-downbeat.
# bt_dataset derives the real set from each corpus directory's info.json; this constant
# is the fallback for anything that ships without one.
BEAT_ONLY_DATASETS = {"smc"}
CLASS_BEAT_ONLY = 2


def collater(data):
    # data = one batch of [audio, annot(, metadata)]
    # s[0]: (T, n_mels) spectrogram;  s[1]: (M, 3) intervals as (start, end, class_id),
    # both in frame units;  s[2]: optional per-item metadata.
    audios = [s[0] for s in data]
    annots = [s[1] for s in data]
    metadata = None

    if len(data[0]) > 2:
        metadata = [s[2] for s in data]

    new_audios = torch.stack(audios)   # (B, T, n_mels)

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
                    new_annots[idx, :annot.shape[0], :] = annot   # (B, M_max, 3), padded with -1
    else:
        new_annots = torch.ones((len(annots), 1, 3)) * -1 # new_annots shape: (B, 1, 3)

    #return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
    if metadata is not None:
        return new_audios, new_annots, metadata

    return new_audios, new_annots
