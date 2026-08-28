#import madmom
import mir_eval
import numpy as np
import scipy.signal
import torch
import os
from beatfcos.utils import calc_iou

def find_beats(t, p, 
                smoothing=127, 
                threshold=0.5, 
                distance=None, 
                sample_rate=44100, 
                beat_type="beat",
                filter_type="none",
                peak_type="simple"):
    # 15, 2

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]

    if filter_type == "savgol":
        # apply smoothing with savgol filter
        p = scipy.signal.savgol_filter(p, smoothing, 2)   
    elif filter_type == "cheby":
        sos = scipy.signal.cheby1(10,
                                  1, 
                                  45, 
                                  btype='lowpass', 
                                  fs=sample_rate,
                                  output='sos')
        p = scipy.signal.sosfilt(sos, p)

    # normalize the smoothed signal between 0.0 and 1.0
    p /= np.max(np.abs(p))                                     

    if peak_type == "simple":
        # by default, we assume that the min distance between beats is fs/4
        # this allows for at max, 4 BPS, which corresponds to 240 BPM 
        # for downbeats, we assume max of 1 downBPS
        if beat_type == "beat":
            if distance is None:
                distance = sample_rate / 4
        elif beat_type == "downbeat":
            if distance is None:
                distance = sample_rate / 2
        else:
            raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")

        # apply simple peak picking
        est_beats, heights = scipy.signal.find_peaks(p, height=threshold, distance=distance)

    elif peak_type == "cwt":
        # use wavelets
        est_beats = scipy.signal.find_peaks_cwt(p, np.arange(1,50))

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p

    return ref_beats, est_beats, est_sm

def evaluate(pred, target, target_sample_rate, use_dbn=False):

    t_beats = target[0,:]
    t_downbeats = target[1,:]
    p_beats = pred[0,:]
    p_downbeats = pred[1,:]

    # ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
    #                                     p_beats.numpy(), 
    #                                     beat_type="beat",
    #                                     sample_rate=target_sample_rate)

    # ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
    #                                             p_downbeats.numpy(), 
    #                                             beat_type="downbeat",
    #                                             sample_rate=target_sample_rate)

    # if use_dbn:
    #     beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=55,
    #         max_bpm=215,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=10,
    #         max_bpm=75,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     beat_pred = pred[0,:].clamp(1e-8, 1-1e-8).view(-1).numpy()
    #     downbeat_pred = pred[1,:].clamp(1e-8, 1-1e-8).view(-1).numpy()

    #     est_beats = beat_dbn.process_offline(beat_pred)
    #     est_downbeats = downbeat_dbn.process_offline(downbeat_pred)

    # evaluate beats - trim beats before 5 seconds.
    ref_beats = mir_eval.beat.trim_beats(ref_beats)
    est_beats = mir_eval.beat.trim_beats(est_beats)
    beat_scores = mir_eval.beat.evaluate(ref_beats, est_beats)

    # evaluate downbeats - trim beats before 5 seconds.
    ref_downbeats = mir_eval.beat.trim_beats(ref_downbeats)
    est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
    downbeat_scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)

    return beat_scores, downbeat_scores

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    #area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area = b[:, 1] - b[:, 0]

    #iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    iw = np.minimum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    #ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    #ih = np.maximum(ih, 0)

    #ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.expand_dims(a[:, 1] - a[:, 0], axis=1) + area - iw

    ua = np.maximum(ua, np.finfo(float).eps)

    #intersection = iw * ih
    intersection = iw

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_results_from_model(audio, target, model, iou_threshold=0.5, score_threshold=0.05, max_thresh=1, downbeat_score_threshold=None, downbeat_sigma=None, downbeat_phase_reweight=False):
    #data = dataset[index]
    #scale = data['scale']
    #audio, target = data

    if torch.cuda.is_available():
        # move data to GPU
        audio = audio.to('cuda')
        target = target.to('cuda')

    # run network
    with torch.no_grad():
        predicted_scores, predicted_labels, predicted_boxes, losses = model( #MJ: shape =(15,) (15,) (15,2)
            (audio, target),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_thresh=max_thresh,
            downbeat_score_threshold=downbeat_score_threshold,
            downbeat_sigma=downbeat_sigma,
            downbeat_phase_reweight=downbeat_phase_reweight
        ) #MJ: The results of model() has been obtained by applying the nms process

    predicted_scores = predicted_scores.cpu()
    predicted_labels = predicted_labels.cpu()
    predicted_boxes  = predicted_boxes.cpu()
    
    return predicted_scores, predicted_labels, predicted_boxes, losses

def get_detections(dataloader, model, num_classes, score_threshold=0.05, max_detections=10000000):
    """ Get the detections from the beatfcos using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the beatfcos.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(num_classes)] for j in range(len(dataloader))]

    model.eval()
    
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            audio, target, metadata = data

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]
        
            predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(audio, target, model, score_threshold=score_threshold)
            # scale = data['scale']

            # # run network
            # if torch.cuda.is_available():
            #     scores, labels, boxes = beatfcos(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            # else:
            #     scores, labels, boxes = beatfcos(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            # scores = scores.cpu().numpy()
            # labels = labels.cpu().numpy()
            # boxes  = boxes.cpu().numpy()

            # # correct boxes for image scale
            # boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(predicted_scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                predicted_scores = predicted_scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-predicted_scores)[:max_detections]

                # select detections
                image_boxes      = predicted_boxes[indices[scores_sort], :]
                image_scores     = predicted_scores[scores_sort]
                image_labels     = predicted_labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(num_classes):
                    #all_detections[index][label] = np.zeros((0, 5))
                    all_detections[index][label] = np.zeros((0, 3))

            print('{}/{}'.format(index + 1, len(dataloader)), end='\r')

    return all_detections


def get_annotations(dataloader, num_classes):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(num_classes)] for j in range(len(dataloader))]

    #for i in range(len(dataloader)):
    for i, data in enumerate(dataloader):
        _, batch_annotations, _ = data
        annotations = batch_annotations[0].cpu().numpy()
        # load the annotations
        #annotations = dataloader.load_annotations(i)

        # copy detections to all_annotations
        for label in range(num_classes):
            #all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            all_annotations[i][label] = annotations[annotations[:, 2] == label, :2].copy()

        print('{}/{}'.format(i + 1, len(dataloader)), end='\r')

    return all_annotations

def evaluate_beat_ap(
    dataloader,
    model,
    num_classes=2,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100
):
    all_detections = get_detections(dataloader, model, num_classes, score_threshold=score_threshold, max_detections=max_detections)
    all_annotations = get_annotations(dataloader, num_classes)
    
    average_precisions = {}

    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(dataloader)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                #scores = np.append(scores, d[4])
                scores = np.append(scores, d[2])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations


    print('\nmAP:')
    for label in range(num_classes):
        #label_name = dataloader.label_to_name(label)
        if label == 0:
            label_name = "Downbeat"
        elif label == 1:
            label_name = "Beat"
        else:
            raise NotImplementedError

        print('{}: {}'.format(label_name, average_precisions[label][0]))
        print("Precision: ",precision[-1])
        print("Recall: ",recall[-1])

        # if save_path!=None:
        #     plt.plot(recall,precision)
        #     # naming the x axis 
        #     plt.xlabel('Recall') 
        #     # naming the y axis 
        #     plt.ylabel('Precision') 

        #     # giving a title to my graph 
        #     plt.title('Precision Recall curve') 

        #     # function to show the plot
        #     plt.savefig(save_path+'/'+label_name+'_precision_recall.jpg')

    return average_precisions

def evaluate_beat_f_measure(dataloader, model, audio_downsampling_factor, audio_sample_rate,
                            score_threshold=0.2, iou_threshold=0.5, max_thresh=1, downbeat_score_threshold=None, downbeat_sigma=None, downbeat_phase_reweight=False):
    model.eval()
    # beat와 downbeat은 confidence 분포가 달라서 최적 threshold도 다름(실측:
    # beat=0.2, downbeat=0.05가 최적). downbeat_score_threshold를 안 주면
    # score_threshold 하나로 기존처럼 beat/downbeat 둘 다에 동일 적용됨.
    effective_downbeat_threshold = downbeat_score_threshold if downbeat_score_threshold is not None else score_threshold
    
    with torch.no_grad():
        # start collecting results
        results = []

        all_beat_ious = []
        all_downbeat_ious = []

        for index, data in enumerate(dataloader):
            audio, target, metadata = data #MJ: audio: shape =(1,1,3000,81) =(B,C,H,W); target: shape=(1,56,3) =(B,L,C)

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0] #MJ: predicted_labels[] = all 0 = all downbeats?
        
            # CombinedLoss.forward()가 jth_classification_loss에 NaN이 뜨면 (특정
            # 곡에서 모델 예측이 극단적으로 나빠질 때 발생 가능) 무조건 raise
            # ValueError를 던져서, 이 곡 하나 때문에 eval 전체가 죽어버리는 문제가
            # 있었음(harmonix 학습 중 실제로 발생, 프로세스 통째로 크래시). 학습
            # 루프(train.py)는 이미 이런 NaN을 skip-and-continue로 처리하는데 eval
            # 쪽엔 그 방어가 없었던 것 - 곡 하나를 스킵하고 나머지 val set은 계속
            # 평가하도록 함.
            try:
                predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(audio, target, model, score_threshold=score_threshold, iou_threshold=iou_threshold, max_thresh=max_thresh, downbeat_score_threshold=downbeat_score_threshold, downbeat_sigma=downbeat_sigma, downbeat_phase_reweight=downbeat_phase_reweight)
            except ValueError:
                print(f"[eval] NaN loss로 스킵됨: index={index}, metadata={metadata}", flush=True)
                continue
            #MJ: Note that the results predicted_scores, predicted_labels, predicted_boxes have been obtained by applying the nms process

            beat_pred_left_positions = []
            downbeat_pred_left_positions = []

            beat_pred_right_positions = []
            downbeat_pred_right_positions = []

            first_pred_beat_index, first_pred_downbeat_index = None, None
            last_pred_beat_index, last_pred_downbeat_index = None, None
            last_target_beat_index, last_target_downbeat_index = None, None

            # construct pred tensor
            for box_id in range(predicted_boxes.shape[0]):
                predicted_score = float(predicted_scores[box_id]) # predicted_scores: (number of anchor positions,)
                predicted_label = int(predicted_labels[box_id]) # predicted_labels: (number of anchor positions,)
                predicted_box = predicted_boxes[box_id, :] # The shape of predicted_boxes is (number of anchor positions, 2)

                # scores are sorted, so we can break
                # but this filtering is redundant, as the filtering is done within the evaluation part of the model
                effective_threshold = effective_downbeat_threshold if predicted_label == 0 else score_threshold
                if predicted_score < effective_threshold:
                    continue

                # if beat (label 1), first row (index 0)
                # if downbeat (label 0), second row (index 1)
                # row = 1 - label
                left_position_index = int(predicted_box[0])
                right_position_index = int(predicted_box[1])

                # We obtained the base level audio from the original audio sampled at 22050 Hz by downsampling audio_downsampling_factor
                # and on that base level audio the gt beat intervals are defined.
                #
                # left_position_index * audio_downsampling_factor is the position of the beat location on the base level audio.
                # In order to get the time of that location, we need to divide it by 22050 Hz

                if predicted_label == 0:
                    downbeat_pred_left_positions.append(left_position_index * audio_downsampling_factor / audio_sample_rate)
                    downbeat_pred_right_positions.append(right_position_index * audio_downsampling_factor / audio_sample_rate)
                elif predicted_label == 1:
                    beat_pred_left_positions.append(left_position_index * audio_downsampling_factor / audio_sample_rate)
                    beat_pred_right_positions.append(right_position_index * audio_downsampling_factor / audio_sample_rate)

                # wavebeat_format_pred_left[row, min(left_position_index, length - 1)] = 1
                # wavebeat_format_pred_right[row, min(right_position_index, length - 1)] = 1

                # box_scores_left[row, min(left_position_index, length - 1)] = score
                # box_scores_right[row, min(right_position_index, length - 1)] = score

                if predicted_label == 0 and (first_pred_downbeat_index is None or left_position_index < first_pred_downbeat_index):
                    first_pred_downbeat_index = left_position_index
                elif predicted_label == 1 and (first_pred_beat_index is None or left_position_index < first_pred_beat_index):
                    first_pred_beat_index = left_position_index

                if predicted_label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif predicted_label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index
            #End  for box_id in range(predicted_boxes.shape[0]):
            
            beat_scores = predicted_scores[predicted_labels == 1]
            beat_intervals = predicted_boxes[predicted_labels == 1][beat_scores >= score_threshold]
            if beat_scores.size(dim=0) == 0:
                sorted_beat_intervals = beat_intervals
            else:
                sorted_beat_intervals = beat_intervals[beat_intervals[:, 0].sort()[1]]
                #MJ: beat_intervals[:, 0] accesses the first column of the beat_intervals array, which corresponds to the start time of each predicted beat interval.
                #beat_intervals[:, 0].sort()[1]: This part selects the indices that would reorder the beat_intervals in ascending order of the start time.
                #beat_intervals[beat_intervals[:, 0].sort()[1]]
                
            beat_ious = torch.zeros(1, 0).to(sorted_beat_intervals.device) #MJ: Shape (1, 0) means it is a tensor with one row and zero columns.

            downbeat_scores = predicted_scores[predicted_labels == 0]
            downbeat_intervals = predicted_boxes[predicted_labels == 0][downbeat_scores >= effective_downbeat_threshold]
            if downbeat_scores.size(dim=0) == 0:
                sorted_downbeat_intervals = downbeat_intervals
            else:
                sorted_downbeat_intervals = downbeat_intervals[downbeat_intervals[:, 0].sort()[1]]
            downbeat_ious = torch.zeros(1, 0).to(downbeat_intervals.device)

            for beat_index, beat_interval in enumerate(sorted_beat_intervals[:-1]):
                next_beat_interval = sorted_beat_intervals[beat_index + 1]

                beat_iou = calc_iou(beat_interval[None], next_beat_interval[None])
                beat_ious = torch.cat((beat_ious, beat_iou), dim=1)
                #MJ: torch.cat((beat_ious, beat_iou), dim=1) adds the newly computed IoU value to the beat_ious tensor, 
                # building the collection of IoU values over time.
                #MJ: The result is that beat_ious will store the IoU values between all consecutive predicted beat intervals. 
                # These values are accumulated over time, 
                # giving you a collection of IoUs for all adjacent pairs of predicted beat intervals.


            for downbeat_index, downbeat_interval in enumerate(sorted_downbeat_intervals[:-1]):
                next_downbeat_interval = sorted_downbeat_intervals[downbeat_index + 1]

                downbeat_iou = calc_iou(downbeat_interval[None], next_downbeat_interval[None])
                downbeat_ious = torch.cat((downbeat_ious, downbeat_iou), dim=1)

            all_beat_ious += beat_ious.tolist()[0]
            all_downbeat_ious += downbeat_ious.tolist()[0]

            if last_pred_beat_index is not None:
                beat_pred_left_positions.append(last_pred_beat_index * audio_downsampling_factor / audio_sample_rate)

            if last_pred_downbeat_index is not None:
                downbeat_pred_left_positions.append(last_pred_downbeat_index * audio_downsampling_factor / audio_sample_rate)

            beat_target_left_positions = []
            downbeat_target_left_positions = []
            # construct target tensor
            for beat_interval in target[0]:
                label = int(beat_interval[2])
                # row = 1 - label

                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])

                # wavebeat_format_target[row, min(left_position_index, length - 1)] = 1
                # label 2 = beat-only 데이터셋(SMC, dataloader.CLASS_BEAT_ONLY):
                # beat GT로만 세고 downbeat GT는 비워 둔다. 이 분기가 없으면 SMC의
                # GT가 통째로 비어 mir_eval이 0을 돌려준다.
                if label == 0:
                    downbeat_target_left_positions.append(left_position_index * audio_downsampling_factor / audio_sample_rate)
                elif label in (1, 2):
                    beat_target_left_positions.append(left_position_index * audio_downsampling_factor / audio_sample_rate)

                if label == 0 and (last_target_downbeat_index is None or right_position_index > last_target_downbeat_index):
                    last_target_downbeat_index = right_position_index
                elif label in (1, 2) and (last_target_beat_index is None or right_position_index > last_target_beat_index):
                    last_target_beat_index = right_position_index
            #End for beat_interval in target[0]:
            
            #wavebeat_format_target[0, min(last_target_beat_index, length - 1)] = 1
            #wavebeat_format_target[1, min(last_target_downbeat_index, length - 1)] = 1
            # beat-only 데이터셋(SMC)에는 downbeat interval이 하나도 없어서
            # last_target_downbeat_index가 None으로 남는다. 원래는 무조건 곱셈을
            # 해서 TypeError로 프로세스가 통째로 죽었음 - 실제로 fcos_lite 학습 두
            # 개가 epoch 0 검증에서 harmonix까지 마치고 SMC에 들어가는 순간 조용히
            # 사라졌다(로그에 traceback도 안 남음, 이 지점이 try/except 밖이라).
            if last_target_beat_index is not None:
                beat_target_left_positions.append(last_target_beat_index * audio_downsampling_factor / audio_sample_rate)
            if last_target_downbeat_index is not None:
                downbeat_target_left_positions.append(last_target_downbeat_index * audio_downsampling_factor / audio_sample_rate)

            predicted_scores = predicted_scores.cpu()
            predicted_labels = predicted_labels.cpu()
            predicted_boxes  = predicted_boxes.cpu()
            
            # beat_scores_left, downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)
            beat_target_left_positions = np.array(beat_target_left_positions)
            beat_pred_left_positions = np.array(beat_pred_left_positions)
            downbeat_target_left_positions = np.array(downbeat_target_left_positions)
            downbeat_pred_left_positions = np.array(downbeat_pred_left_positions)

            # beat_pred_right_positions = np.array(beat_pred_right_positions)
            # downbeat_pred_right_positions = np.array(downbeat_pred_right_positions)

            # beat_left_and_length = np.stack((
            #     beat_pred_left_positions,
            #     beat_pred_right_positions - beat_pred_left_positions
            # ), axis=1)
            # beat_left_and_length = beat_left_and_length[beat_left_and_length[:, 0].argsort()]
            #print(f"Beat time and length\n{beat_left_and_length}")

            beat_target_left_positions.sort()
            beat_pred_left_positions.sort()
            downbeat_target_left_positions.sort()
            downbeat_pred_left_positions.sort()

            # print(f"beat_pred_left_positions: {beat_pred_left_positions}")
            # print(f"beat_target_left_positions: {beat_target_left_positions}")
            # print(f"downbeat_pred_left_positions: {downbeat_pred_left_positions}")
            # print(f"downbeat_target_left_positions: {downbeat_target_left_positions}")

            beat_target_left_positions = mir_eval.beat.trim_beats(beat_target_left_positions)
            beat_pred_left_positions = mir_eval.beat.trim_beats(beat_pred_left_positions)
            beat_scores = mir_eval.beat.evaluate(beat_target_left_positions, beat_pred_left_positions)

            # evaluate downbeats - trim beats before 5 seconds.
            downbeat_target_left_positions = mir_eval.beat.trim_beats(downbeat_target_left_positions)
            downbeat_pred_left_positions = mir_eval.beat.trim_beats(downbeat_pred_left_positions)
            downbeat_scores = mir_eval.beat.evaluate(downbeat_target_left_positions, downbeat_pred_left_positions)

            # try:
            #     dbn_beat_scores_left, dbn_downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_left = { 'F-measure': 0 }
            #     dbn_downbeat_scores_left = { 'F-measure': 0 }

            # beat_scores_right, downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_right, dbn_downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_right = { 'F-measure': 0 }
            #     dbn_downbeat_scores_right = { 'F-measure': 0 }

            # beat_scores_average, downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_average, dbn_downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_average = { 'F-measure': 0 }
            #     dbn_downbeat_scores_average = { 'F-measure': 0 }

            # beat_scores_weighted, downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_weighted, dbn_downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_weighted = { 'F-measure': 0 }
            #     dbn_downbeat_scores_weighted = { 'F-measure': 0 }


            print(f"{index}/{len(dataloader)} {metadata['Filename']}")
            # head_type="hungarian"에는 leftness/adjacency 개념이 없음(anchor 기반
            # FCOS 전용) - losses[1]/losses[2]는 실제로 bbox L1 / GIoU, losses[3](ADJ)은
            # 항상 0이라 헷갈리지 않게 라벨을 다르게 출력한다.
            if getattr(model, 'module', model).head_type == "hungarian":
                print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | CLS: {losses[0]:0.3f} | BBOX(L1): {losses[1]:0.3f} | GIOU: {losses[2]:0.3f}")
            else:
                print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | CLS: {losses[0]:0.3f} | REG: {losses[1]:0.3f} | LFT: {losses[2]:0.3f} | ADJ: {losses[3]:0.3f}")
            #print("LEFT")
            # print(f"BEAT (F-measure): {beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_left['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_left['F-measure']:0.3f}")
            #print("RIGHT")
            #print(f"BEAT (F-measure): {beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_right['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_right['F-measure']:0.3f}")
            # print("AVERAGE")
            # print(f"BEAT (F-measure): {beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_average['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_average['F-measure']:0.3f}")
            # print("WEIGHTED")
            # print(f"BEAT (F-measure): {beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_weighted['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_weighted['F-measure']:0.3f}\n")

            # beat_scores = beat_scores_left
            # downbeat_scores = downbeat_scores_left
            # dbn_beat_scores = dbn_beat_scores_left
            # dbn_downbeat_scores = dbn_downbeat_scores_left

            # if predicted_boxes.shape[0] > 0:
            #     # change to (x, y, w, h) (MS COCO standard)
            #     #boxes[:, 2] -= boxes[:, 0]
            #     #boxes[:, 3] -= boxes[:, 1]

            #     # compute predicted labels and scores
            #     #for box, score, label in zip(boxes[0], scores[0], labels[0]):
            #     for box_id in range(predicted_boxes.shape[0]):
            #         predicted_score = float(predicted_scores[box_id])
            #         predicted_label = int(predicted_labels[box_id])
            #         predicted_box = predicted_boxes[box_id, :]

            #         # scores are sorted, so we can break
            #         if predicted_score < score_threshold:
            #             # break
            #             continue

            # append detection for each positively labeled class
            image_result = {
                'image_id': metadata["Filename"],
                #'category_id': dataset.label_to_coco_label(label),
                #'score': float(predicted_score),
                #'bbox': predicted_box.tolist(),
                'beat_scores': beat_scores,
                'downbeat_scores': downbeat_scores,
                'cls_loss': losses[0],
                'reg_loss': losses[1],
                'lft_loss': losses[2],
                'adj_loss': losses[3],

                # 'beat_scores_left': beat_scores_left,
                # 'downbeat_scores_left': downbeat_scores_left,
                # 'beat_scores_right': beat_scores_right,
                # 'downbeat_scores_right': downbeat_scores_right,
                # 'beat_scores_average': beat_scores_average,
                # 'downbeat_scores_average': downbeat_scores_average,
                # 'beat_scores_weighted': beat_scores_weighted,
                # 'downbeat_scores_weighted': downbeat_scores_weighted,
                # 'dbn_beat_scores': dbn_beat_scores,
                # 'dbn_downbeat_scores': dbn_downbeat_scores
            }

            # append detection to results
            results.append(image_result)
        # END for index, data in enumerate(dataset)

        # if not len(results):
        #     return

        beat_mean_f_measure = np.mean([result['beat_scores']['F-measure'] for result in results])  #MJ: results = detection results of the batch of the test-dataset
        downbeat_mean_f_measure = np.mean([result['downbeat_scores']['F-measure'] for result in results])
        cls_loss_mean = np.mean([result['cls_loss'] for result in results])
        reg_loss_mean = np.mean([result['reg_loss'] for result in results])
        lft_loss_mean = np.mean([result['lft_loss'] for result in results])
        adj_loss_mean = np.mean([result['adj_loss'] for result in results])
        # left_beat_mean_f_measure = np.mean([result['beat_scores_left']['F-measure'] for result in results])
        # left_downbeat_mean_f_measure = np.mean([result['downbeat_scores_left']['F-measure'] for result in results])
        #right_beat_mean_f_measure = np.mean([result['beat_scores_right']['F-measure'] for result in results])
        #right_downbeat_mean_f_measure = np.mean([result['downbeat_scores_right']['F-measure'] for result in results])
        # average_beat_mean_f_measure = np.mean([result['beat_scores_average']['F-measure'] for result in results])
        # average_downbeat_mean_f_measure = np.mean([result['downbeat_scores_average']['F-measure'] for result in results])
        # weighted_beat_mean_f_measure = np.mean([result['beat_scores_weighted']['F-measure'] for result in results])
        # weighted_downbeat_mean_f_measure = np.mean([result['downbeat_scores_weighted']['F-measure'] for result in results])
        #dbn_beat_mean_f_measure = np.mean([result['dbn_beat_scores']['F-measure'] for result in results])
        #dbn_downbeat_mean_f_measure = np.mean([result['dbn_downbeat_scores']['F-measure'] for result in results])

        print(f"Average beat F-measure: {beat_mean_f_measure:0.3f}")
        print(f"Average downbeat F-measure: {downbeat_mean_f_measure:0.3f}")
        if getattr(model, 'module', model).head_type == "hungarian":
            print(f"Average losses | CLS: {cls_loss_mean:0.3f} | BBOX(L1): {reg_loss_mean:0.3f} | GIOU: {lft_loss_mean:0.3f}")
        else:
            print(f"Average losses | CLS: {cls_loss_mean:0.3f} | REG: {reg_loss_mean:0.3f} | LFT: {lft_loss_mean:0.3f} | ADJ: {adj_loss_mean:0.3f}")
        # print(f"Average left beat F-measure: {left_beat_mean_f_measure:0.3f}")
        # print(f"Average left downbeat F-measure: {left_downbeat_mean_f_measure:0.3f}")
        #print(f"Average right beat F-measure: {right_beat_mean_f_measure:0.3f}")
        #print(f"Average right downbeat F-measure: {right_downbeat_mean_f_measure:0.3f}")
        # print(f"Average average beat F-measure: {average_beat_mean_f_measure:0.3f}")
        # print(f"Average average downbeat F-measure: {average_downbeat_mean_f_measure:0.3f}")
        # print(f"Average weighted beat F-measure: {weighted_beat_mean_f_measure:0.3f}")
        # print(f"Average weighted downbeat F-measure: {weighted_downbeat_mean_f_measure:0.3f}")
        print()
        # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

        import matplotlib.pyplot as plt

        def create_histogram(data, path, title):
            total = len(data)

            # Create histogram and capture the counts, bin edges, and patches
            counts, bins, patches = plt.hist(data, bins=10, edgecolor='black')

            # Calculate bin centers for annotation
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            # Annotate each bin with the count and percentage
            for i, count in enumerate(counts):
                if count > 0:  # Only annotate bins with at least one entry
                    percentage = (count / total) * 100
                    # Place text at the top of each bar
                    plt.text(
                        bin_centers[i],        # x position: center of bin
                        count,                 # y position: bar height (count)
                        f'{int(count)}\n({percentage:.1f}%)',
                        ha='center',           # horizontally center text
                        va='bottom'            # place text above the bar
                    )

            # Add labels and title
            plt.xlabel('IoU')
            plt.ylabel('Frequency')
            plt.title(title)

            # Save the plot as a PNG file
            plt.savefig(path)

            # Close the plot to free up memory
            plt.close()
        #End def create_histogram(data, path, title):
        

        create_histogram(
            all_beat_ious, f"beat_histogram_{score_threshold}.png",
            f"Beat Histogram Before NMS (threshold = {score_threshold}~{max_thresh})"
        )

        create_histogram(
            all_downbeat_ious, f"downbeat_histogram_{score_threshold}.png",
            f"Downbeat Histogram Before NMS (threshold = {score_threshold}~{max_thresh})"
        )

        model.train()

        # beat_mean_f_measure = left_beat_mean_f_measure#average_beat_mean_f_measure
        # downbeat_mean_f_measure = left_downbeat_mean_f_measure#average_downbeat_mean_f_measure
        return beat_mean_f_measure, downbeat_mean_f_measure, results#dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure


def evaluate_beat_f_measure_subset(dataloader, model, audio_downsampling_factor, audio_sample_rate,
                                   window_frames=1280, border_frames=8,
                                   threshold_beat=0.2, threshold_downbeat=0.2,
                                   label=""):
    """평가 경로: order-constrained subset selection head (beat_dp_matching-1.pdf).

    FCOS 경로(evaluate_beat_f_measure)와 반환 형식/GT 추출/mir_eval 호출을 똑같이
    맞춰서 fcos_lite 등과 숫자를 직접 비교할 수 있게 함. 다른 점은 두 가지뿐:

    1) 모델이 고정 길이 window만 받으므로 곡을 window_frames짜리 조각으로 타일링해서
       디코딩하고 Algorithm 4로 이어붙임 (beatfcos/stitching.py). Soft-NMS 없음 -
       eq. (1)이 교차를 구조적으로 금지하므로 중복 제거가 필요 없다.

    2) [중요] 이 헤드의 클래스는 {DB, B, none} 배타적(exclusive)이지만, GT와 mir_eval
       쪽 "beat" 정의는 downbeat을 포함함 (dataloader의 target 채널 0에 downbeat도
       전부 1로 찍히고, make_intervals가 그걸로 beat interval을 만듦). 따라서
       predicted beat 목록에는 DB로 분류된 후보도 반드시 합쳐 넣어야 한다. 이걸
       빠뜨리면 모든 downbeat이 beat miss로 잡혀서 Beat F가 구조적으로 ~1/L만큼
       깎인다 (4/4면 약 25%).
    """
    from beatfcos.stitching import stitch_piece
    from beatfcos.subset_head import BEAT, DOWNBEAT

    model.eval()
    inner = getattr(model, 'module', model)
    to_seconds = audio_downsampling_factor / audio_sample_rate

    results = []
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            audio, target, metadata = data
            metadata = metadata[0]

            mel = audio[0]  # (T, n_mels) - eval은 batch_size 1 가정 (FCOS 경로와 동일)
            if torch.cuda.is_available():
                mel = mel.cuda()

            def forward_fn(fragment):
                # DataParallel wrapper를 그대로 쓰면 batch 1짜리 조각이 GPU 하나로만
                # 가므로 굳이 우회하지 않아도 되지만, inner를 쓰면 forward의
                # len(inputs)==2 관례(배치 크기 2인 텐서를 (audio, target)으로
                # 오인하는 기존 함정)와 무관하게 항상 안전함.
                return inner(fragment)

            classes, frames, _scores = stitch_piece(
                mel, forward_fn, window_frames, border_frames,
                threshold_beat=threshold_beat, threshold_downbeat=threshold_downbeat)

            classes = classes.cpu()
            frames = frames.cpu()

            # (2) 위 주석 참고: beat 예측 = B로 분류된 것 + DB로 분류된 것
            beat_pred_positions = np.sort(frames[(classes == BEAT) | (classes == DOWNBEAT)].numpy() * to_seconds)
            downbeat_pred_positions = np.sort(frames[classes == DOWNBEAT].numpy() * to_seconds)

            # --- GT: FCOS 경로와 완전히 동일한 방식으로 (M,3) interval에서 복원 ---
            beat_target_positions, downbeat_target_positions = [], []
            last_target_beat_index, last_target_downbeat_index = None, None
            for beat_interval in target[0]:
                interval_label = int(beat_interval[2])
                if interval_label < 0:
                    continue  # collater의 -1 패딩
                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])
                if interval_label == 0:
                    downbeat_target_positions.append(left_position_index * to_seconds)
                    if last_target_downbeat_index is None or right_position_index > last_target_downbeat_index:
                        last_target_downbeat_index = right_position_index
                elif interval_label == 2:
                    # beat-only 데이터셋(SMC): class_id 2는 "beat인 건 확실하나 B/DB
                    # 구분은 라벨이 없음"을 뜻함(dataloader.CLASS_BEAT_ONLY). beat GT로는
                    # 그대로 세고, downbeat GT는 존재하지 않으므로 비워 둔다.
                    # 이 분기가 없으면 class 2가 어느 쪽에도 안 걸려서 SMC의 GT가
                    # beat/downbeat 모두 빈 리스트가 되고, mir_eval이 전부 0을 돌려줘서
                    # macro-average(=체크포인트 저장과 LR 스케줄러의 기준)를 조용히
                    # 끌어내린다 (실측: v3 epoch 0의 Beat 0.563 -> 0.432).
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

            beat_scores = mir_eval.beat.evaluate(
                mir_eval.beat.trim_beats(beat_target_positions),
                mir_eval.beat.trim_beats(beat_pred_positions))
            downbeat_scores = mir_eval.beat.evaluate(
                mir_eval.beat.trim_beats(downbeat_target_positions),
                mir_eval.beat.trim_beats(downbeat_pred_positions))

            print(f"{index}/{len(dataloader)} {metadata['Filename']}")
            print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | "
                  f"DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f} | "
                  f"pred B/DB: {len(beat_pred_positions)}/{len(downbeat_pred_positions)} | "
                  f"gt B/DB: {len(beat_target_positions)}/{len(downbeat_target_positions)}")

            results.append({
                'image_id': metadata["Filename"],
                'beat_scores': beat_scores,
                'downbeat_scores': downbeat_scores,
                # FCOS 경로의 results 스키마와 키를 맞춰둠(평가 스크립트 호환).
                # subset head에는 leftness/adjacency 개념이 없어 0.
                'cls_loss': 0.0, 'reg_loss': 0.0, 'lft_loss': 0.0, 'adj_loss': 0.0,
            })

    beat_mean_f_measure = float(np.mean([r['beat_scores']['F-measure'] for r in results])) if results else 0.0
    downbeat_mean_f_measure = float(np.mean([r['downbeat_scores']['F-measure'] for r in results])) if results else 0.0
    print(f"{label}Average beat F-measure: {beat_mean_f_measure:0.3f}")
    print(f"{label}Average downbeat F-measure: {downbeat_mean_f_measure:0.3f}\n")

    # beat_this Table 2와 같은 열(F1/CMLt/AMLt)로 바로 대조할 수 있도록 추가.
    # mir_eval.beat.evaluate()가 애초에 CMLt/AMLt까지 다 계산해서 beat_scores/
    # downbeat_scores 안에 넣어주고 있었는데, 지금까지는 F-measure만 뽑아 썼음.
    if results:
        # mir_eval.beat.evaluate()의 실제 키는 "CMLt"/"AMLt"가 아니라 풀네임
        # ("Correct/Any Metric Level Total")이다 (mir_eval.beat.evaluate 소스로 확인).
        beat_cmlt = float(np.mean([r['beat_scores']['Correct Metric Level Total'] for r in results]))
        beat_amlt = float(np.mean([r['beat_scores']['Any Metric Level Total'] for r in results]))
        downbeat_cmlt = float(np.mean([r['downbeat_scores']['Correct Metric Level Total'] for r in results]))
        downbeat_amlt = float(np.mean([r['downbeat_scores']['Any Metric Level Total'] for r in results]))
        print(f"{label}Average beat CMLt: {beat_cmlt:0.3f} | AMLt: {beat_amlt:0.3f}")
        print(f"{label}Average downbeat CMLt: {downbeat_cmlt:0.3f} | AMLt: {downbeat_amlt:0.3f}\n")

    return beat_mean_f_measure, downbeat_mean_f_measure, results
