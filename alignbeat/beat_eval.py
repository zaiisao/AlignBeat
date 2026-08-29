"""Evaluation for the order-preserving alignment head.

Decodes a piece by tiling it into fixed-length fragments, stitching the
central keep-regions back together (Section 9.3), and scoring against the
annotation with mir_eval. No NMS: eq. (1) forbids crossings structurally, so
there is nothing to deduplicate.
"""
import mir_eval
import numpy as np
import torch


def evaluate_beat_f_measure_subset(dataloader, model, audio_downsampling_factor, audio_sample_rate,
                                   window_frames=1280, border_frames=8,
                                   threshold_beat=0.2, threshold_downbeat=0.2,
                                   label="", use_amp=False, full_metrics=False,
                                   verbose=False):
    """Evaluation path.

    1) The model only accepts a fixed-length window, so a piece is tiled into
       window_frames fragments, decoded, and joined back with the Section 9.3
       stitching (alignbeat/stitching.py). No NMS - eq. (1) structurally forbids
       crossings, so there is nothing to deduplicate.

    2) Important: the head's classes {DB, B, none} are mutually exclusive, but the
       "beat" definition used by the ground truth and by mir_eval includes downbeats
       (the dataloader marks downbeats as 1 in target channel 0 as well, and
       make_intervals builds beat intervals from that). So the predicted beat list
       must also include the candidates classified as DB. Omitting them turns every
       downbeat into a missed beat and structurally costs Beat F about 1/L
       (roughly 25% in 4/4).
    """
    from alignbeat.stitching import stitch_piece
    from alignbeat.subset_head import BEAT, DOWNBEAT

    model.eval()
    inner = getattr(model, 'module', model)
    to_seconds = audio_downsampling_factor / audio_sample_rate

    results = []
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            audio, target, metadata = data
            metadata = metadata[0]

            mel = audio[0]  # (T, n_mels) - eval assumes batch_size 1
            if torch.cuda.is_available():
                mel = mel.cuda()

            def forward_fn(fragment):
                # Using the DataParallel wrapper directly would be harmless here, since
                # a batch-of-1 fragment lands on a single GPU anyway. Going through
                # inner is unconditionally safe though: it sidesteps forward's
                # len(inputs)==2 convention, which mistakes a batch of 2 for an
                # (audio, target) pair.
                # Match training's precision: without this the eval forward runs in
                # fp32 while training runs in bf16, and validation -- which is the
                # larger half of an epoch here -- pays full price for no benefit.
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    return inner(fragment)

            classes, frames, _scores = stitch_piece(
                mel, forward_fn, window_frames, border_frames,
                threshold_beat=threshold_beat, threshold_downbeat=threshold_downbeat)

            classes = classes.cpu()
            frames = frames.cpu()

            # See note (2) above: predicted beats = those classified B plus those DB
            beat_pred_positions = np.sort(frames[(classes == BEAT) | (classes == DOWNBEAT)].numpy() * to_seconds)
            downbeat_pred_positions = np.sort(frames[classes == DOWNBEAT].numpy() * to_seconds)

            # --- GT: reconstructed from the (M,3) intervals ---
            beat_target_positions, downbeat_target_positions = [], []
            last_target_beat_index, last_target_downbeat_index = None, None
            for beat_interval in target[0]:
                interval_label = int(beat_interval[2])
                if interval_label < 0:
                    continue  # -1 padding from the collater
                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])
                if interval_label == 0:
                    downbeat_target_positions.append(left_position_index * to_seconds)
                    if last_target_downbeat_index is None or right_position_index > last_target_downbeat_index:
                        last_target_downbeat_index = right_position_index
                elif interval_label == 2:
                    # Beat-only datasets (SMC): class_id 2 means "certainly a beat, but
                    # the B/DB distinction is unlabelled" (dataloader.CLASS_BEAT_ONLY).
                    # Count it as beat GT; there is no downbeat GT, so leave that empty.
                    # Without this branch class 2 matches neither case, SMC's GT comes
                    # out as empty beat and downbeat lists, mir_eval returns all zeros,
                    # and the macro-average - which drives checkpoint saving and the LR
                    # scheduler - is quietly dragged down (measured: v3 epoch 0 Beat
                    # 0.563 -> 0.432).
                    beat_target_positions.append(left_position_index * to_seconds)
                    if last_target_beat_index is None or right_position_index > last_target_beat_index:
                        last_target_beat_index = right_position_index
                elif interval_label == 1:
                    beat_target_positions.append(left_position_index * to_seconds)
                    if last_target_beat_index is None or right_position_index > last_target_beat_index:
                        last_target_beat_index = right_position_index

            if last_target_beat_index is not None:
                beat_target_positions.append(last_target_beat_index * to_seconds)
            if last_target_downbeat_index is not None:
                downbeat_target_positions.append(last_target_downbeat_index * to_seconds)

            beat_target_positions = np.sort(np.array(beat_target_positions))
            downbeat_target_positions = np.sort(np.array(downbeat_target_positions))

            # mir_eval.beat.evaluate() runs the whole suite -- continuity, information
            # gain, Goto, P-score -- at ~468 ms per call, and it is called twice per
            # song. f_measure alone is ~13 ms. Over 571 val songs every epoch that is
            # the difference between ~9 minutes of validation and ~15 seconds, and it
            # dominated the epoch: validation was the larger half of an epoch purely
            # because of this. Per-epoch validation therefore computes only what the
            # macro average actually consumes; evaluate_all_datasets.py passes
            # full_metrics=True for the CMLt/AMLt columns it reports at the end.
            # (beat_this does exactly this, for the same reason -- see the comment in
            # its own Metrics class about limiting validation metrics.)
            score = (mir_eval.beat.evaluate if full_metrics
                     else lambda t, p: {'F-measure': mir_eval.beat.f_measure(t, p)})
            beat_scores = score(
                mir_eval.beat.trim_beats(beat_target_positions),
                mir_eval.beat.trim_beats(beat_pred_positions))
            downbeat_scores = score(
                mir_eval.beat.trim_beats(downbeat_target_positions),
                mir_eval.beat.trim_beats(downbeat_pred_positions))

            if verbose:
                print(f"{index}/{len(dataloader)} {metadata['Filename']}")
                print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | "
                      f"DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | "
                      f"pred B/DB: {len(beat_pred_positions)}/{len(downbeat_pred_positions)} | "
                      f"gt B/DB: {len(beat_target_positions)}/{len(downbeat_target_positions)}")

            results.append({
                'image_id': metadata["Filename"],
                'beat_scores': beat_scores,
                'downbeat_scores': downbeat_scores,
                # results schema the evaluation scripts expect.
                'cls_loss': 0.0, 'reg_loss': 0.0, 'lft_loss': 0.0, 'adj_loss': 0.0,
            })

    beat_mean_f_measure = float(np.mean([r['beat_scores']['F-measure'] for r in results])) if results else 0.0
    downbeat_mean_f_measure = float(np.mean([r['downbeat_scores']['F-measure'] for r in results])) if results else 0.0
    print(f"{label}Average beat F-measure: {beat_mean_f_measure:0.3f}")
    print(f"{label}Average downbeat F-measure: {downbeat_mean_f_measure:0.3f}\n")

    # Reported so the numbers line up column-for-column with beat_this Table 2
    # (F1/CMLt/AMLt). mir_eval.beat.evaluate() already computes CMLt/AMLt and puts them
    # in beat_scores/downbeat_scores; only F-measure was being read out until now.
    if results and full_metrics:
        # The actual keys in mir_eval.beat.evaluate() are not "CMLt"/"AMLt" but the
        # full names, "Correct/Any Metric Level Total" (checked against its source).
        beat_cmlt = float(np.mean([r['beat_scores']['Correct Metric Level Total'] for r in results]))
        beat_amlt = float(np.mean([r['beat_scores']['Any Metric Level Total'] for r in results]))
        downbeat_cmlt = float(np.mean([r['downbeat_scores']['Correct Metric Level Total'] for r in results]))
        downbeat_amlt = float(np.mean([r['downbeat_scores']['Any Metric Level Total'] for r in results]))
        print(f"{label}Average beat CMLt: {beat_cmlt:0.3f} | AMLt: {beat_amlt:0.3f}")
        print(f"{label}Average downbeat CMLt: {downbeat_cmlt:0.3f} | AMLt: {downbeat_amlt:0.3f}\n")

    return beat_mean_f_measure, downbeat_mean_f_measure, results
