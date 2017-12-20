# -*- coding:utf-8 -*-
import sys

sys.path.append('/home/prmct/workspace/py-RFCN-priv/caffe-priv/python')
sys.path.append('/home/prmct/workspace/py-RFCN-priv/lib')

import caffe
import cv2
import os
import numpy as np
import datetime
import cPickle
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
from nms.cpu_nms import cpu_soft_nms

gpu_mode = True gpu_id = 1
val_file = '../../test-dev2017.txt'  # for voc2007 test
image_root = '/home/prmct/Database/MSCOCO2017/test2017/'
save_root = './predict_test/'
rpn_deploy = './rpn_rcnn_deploys/rpn_deploy_rfcn_coco_air101-merge.prototxt'
rcnn_deploy = './rpn_rcnn_deploys/rcnn_deploy_rfcn_coco_air101-merge.prototxt'
model_weights = './output/rfcn_end2end/coco81_trainval/' \
                'rfcn_coco_air101_ms-ohem-multigrid-deformpsroi_iter_1100000.caffemodel'
NUM_CLS = 81

# cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# cfg.PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])
cfg.PIXEL_MEANS = np.array([[[103.52, 116.28, 123.675]]])
cfg.PIXEL_STDS = np.array([[[57.375, 57.12, 58.395]]])
# cfg.PIXEL_MEANS = np.array([[[128.0, 128.0, 128.0]]])
# cfg.PIXEL_STDS = np.array([[[128.0, 128.0, 128.0]]])

# ----- common setting -----
cfg.TEST.AGNOSTIC = True
cfg.TEST.HAS_RPN = True
cfg.TEST.BBOX_REG = True
cfg.TEST.RCNN_BATCH_SIZE = 2000  # batch_size of rcnn network
ROI_FEATURE = ['conv_new_1', 'rfcn_cls', 'rfcn_bbox']
# --------------------------

# ----- nms and box-voting setting -----
cfg.TEST.BBOX_VOTE = False
cfg.TEST.SOFT_NMS = 0  # 0 for standard NMS, 1 for linear weighting, 2 for gaussian weighting
cfg.TEST.NMS = 0.3
cfg.TEST.RPN_PRE_NMS_TOP_N = 2000
cfg.TEST.RPN_POST_NMS_TOP_N = 100
cfg.TEST.CONF_THRESH = 0.05
# cfg.TEST.RPN_MIN_SIZE = 0
cfg.USE_GPU_NMS = False
# ---------------------------------------

# ----- multi scale and flipping test setting -----
cfg.TEST.IMAGE_FLIP = False
cfg.TEST.SCALES = [600]
cfg.TEST.MAX_SIZE = [1000]
# cfg.TEST.SCALES = [480, 576, 688, 864, 1200, 1400]
# cfg.TEST.MAX_SIZE = [800, 1000, 1200, 1500, 1800, 1900]
cfg.TEST.ROI_POLICY = 'assign'  # str: normal\merge\assign
cfg.TEST.ROI_NMS_THRESH = 0.8  # when ROI_POLICY=merge
RPN_SCORES = 'rpn-scores'  # when ROI_POLICY=merge
FEAT_STRIDE = 16.0  # when ROI_POLICY=assign
BASE_SIZE = 224.0  # when ROI_POLICY=assign
# --------------------------------------------------

# ----- multi crop setting -----
cfg.TEST.MULTI_CROP = True
cfg.TEST.CROP_SIZE_RATIO = 0.75
# cfg.TEST.CROP_REMAIN_THRESH = 0.75
CROP_RPN_PRE_NMS_TOP_N = int(cfg.TEST.RPN_PRE_NMS_TOP_N * cfg.TEST.CROP_SIZE_RATIO * 0.5)
CROP_RPN_POST_NMS_TOP_N = int(cfg.TEST.RPN_POST_NMS_TOP_N * cfg.TEST.CROP_SIZE_RATIO * 0.5)
# ------------------------------

# ----- iter box average setting -----
cfg.TEST.ITER_BBOX_AVG = False
cfg.TEST.ITER_NUM = 2  # iter_num should >= 2
# ------------------------------------

_classes = ('__background__',  # always index 0
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush')

if gpu_mode:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()

t1 = datetime.datetime.now()
net_rpn = caffe.Net(rpn_deploy, model_weights, caffe.TEST)
net_rcnn = caffe.Net(rcnn_deploy, model_weights, caffe.TEST)
t2 = datetime.datetime.now()
print 'load model:', t2 - t1


def eval_batch(max_per_image=100):
    eval_images = []
    f = open(val_file, 'r')
    for i in f:
        eval_images.append(i.strip())

    skip_num = 0
    eval_len = len(eval_images)

    all_boxes = [[[] for _ in xrange(eval_len - skip_num)]
                 for _ in xrange(NUM_CLS)]

    start_time = datetime.datetime.now()
    for i in xrange(eval_len - skip_num):
        test_img = cv2.imread(image_root + eval_images[i + skip_num])
        timer_pt1 = datetime.datetime.now()
        im_boxes = eval_img(test_img)
        timer_pt2 = datetime.datetime.now()

        for j in xrange(1, NUM_CLS):
            all_boxes[j][i] = im_boxes[j]

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, NUM_CLS)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, NUM_CLS):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        print 'Testing image: {}/{} {} {}s' \
            .format(str(i + 1), str(eval_len - skip_num), str(eval_images[i + skip_num]),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds))

    det_file = './detections.pkl'
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    for cls_ind, cls in enumerate(_classes):
        if cls == '__background__':
            continue
        print 'Writing {} VOC results file'.format(cls)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        filename = save_root + 'comp4' + '_det' + '_test_' + cls + '.txt'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(eval_images[skip_num:]):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))
    end_time = datetime.datetime.now()
    print '\nEvaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))


def eval_img(_img):
    roi_info = get_rois(_img)
    scores, pred_boxes = rois_clsloc(roi_info, _img)

    if len(scores) == 0:
        print scores

    if cfg.TEST.ITER_BBOX_AVG:
        im_boxes = det_post_process(scores, pred_boxes, vote=cfg.TEST.BBOX_VOTE, nms_type=cfg.TEST.SOFT_NMS,
                                    nms_thresh=cfg.TEST.NMS, conf_thresh=cfg.TEST.CONF_THRESH)
    else:
        im_boxes = det_post_process(scores, pred_boxes, vote=cfg.TEST.BBOX_VOTE, nms_type=cfg.TEST.SOFT_NMS,
                                    nms_thresh=cfg.TEST.NMS, conf_thresh=cfg.TEST.CONF_THRESH)

    return im_boxes


def get_rois(_img):
    original_im = _img.astype(np.float32, copy=True)
    original_im -= cfg.PIXEL_MEANS
    original_im /= cfg.PIXEL_STDS

    scaled_ims, scaled_ratios = multi_scale_operator(original_im)
    # multi scale testing with flipping
    roi_info = []
    for i in xrange(len(scaled_ims)):
        im = scaled_ims[i]
        # rpn
        roi_feature_blob, roi_blob, roi_scores = rpn_process(net_rpn, im)

        # multi crop testing
        if cfg.TEST.MULTI_CROP:
            keep_pre = cfg.TEST.RPN_PRE_NMS_TOP_N
            keep_post = cfg.TEST.RPN_POST_NMS_TOP_N
            cfg.TEST.RPN_PRE_NMS_TOP_N = CROP_RPN_PRE_NMS_TOP_N
            cfg.TEST.RPN_POST_NMS_TOP_N = CROP_RPN_POST_NMS_TOP_N
            crop_ims, crop_offsets = multi_crop_operator(im)  # 9 crops from single scale
            for j in xrange(len(crop_ims)):
                crop_roi_feature_blob, crop_roi_blob, crop_roi_scores = rpn_process(net_rpn, crop_ims[j])
                for k in xrange(len(crop_roi_blob)):
                    tmp_roi = crop_roi_blob[k][1:5] * cfg.TEST.CROP_SIZE_RATIO
                    tmp_roi = np.array(tmp_roi)
                    tmp_roi += np.array([crop_offsets[j][0], crop_offsets[j][1],
                                         crop_offsets[j][0], crop_offsets[j][1]])
                    roi_scores = np.concatenate((roi_scores, [crop_roi_scores[k]]), axis=0)
                    roi_blob = np.concatenate((roi_blob, [[0, tmp_roi[0], tmp_roi[1], tmp_roi[2], tmp_roi[3]]]), axis=0)
                pass
            cfg.TEST.RPN_PRE_NMS_TOP_N = keep_pre
            cfg.TEST.RPN_POST_NMS_TOP_N = keep_post
        roi_info.append(dict(im_shape=im.shape, roi_feature_blob=roi_feature_blob, roi_blob=roi_blob.copy(),
                             roi_scores=roi_scores.copy(), scaled_ratio=scaled_ratios[i]))
        print roi_blob.shape,
    return roi_info


def rois_clsloc(roi_info, _img):
    scores_list = []
    pred_boxes_list = []
    # merge all rois
    if cfg.TEST.ROI_POLICY == 'normal':
        pass
    elif cfg.TEST.ROI_POLICY == 'merge':
        all_rois = []
        all_scores = []
        for i in xrange(len(roi_info)):
            im_width = roi_info[i]['im_shape'][1]
            if roi_info[i]['scaled_ratio'] < 0:
                old_xmin = roi_info[i]['roi_blob'][:, 1].copy()
                old_xmax = roi_info[i]['roi_blob'][:, 3].copy()
                roi_info[i]['roi_blob'][:, 1] = im_width - old_xmax - 1
                roi_info[i]['roi_blob'][:, 3] = im_width - old_xmin - 1
            all_rois.extend(roi_info[i]['roi_blob'] / abs(roi_info[i]['scaled_ratio']))
            all_scores.extend(roi_info[i]['roi_scores'])
        all_rois = np.asarray(all_rois)
        all_scores = np.asarray(all_scores)

        all_rpn_dets = np.zeros((len(all_scores), 5), dtype=np.float)
        all_rpn_dets[:, :4] = all_rois[:, 1:5]
        all_rpn_dets[:, 4] = all_scores[:, 0]
        if len(roi_info) == 1:
            _keep = soft_nms(all_rpn_dets, Nt=1.01, threshold=0.0, method=0)
        else:
            _keep = soft_nms(all_rpn_dets, Nt=cfg.TEST.ROI_NMS_THRESH, method=0)
        keep_rois = all_rois[_keep, :]
        print '({})'.format(str(keep_rois.shape)),

        for i in xrange(len(roi_info)):  # replace old rois
            temp_rois = keep_rois.copy()
            if roi_info[i]['scaled_ratio'] < 0:
                old_xmin = temp_rois[:, 1].copy()
                old_xmax = temp_rois[:, 3].copy()
                temp_rois[:, 1] = _img.shape[1] - old_xmax - 1
                temp_rois[:, 3] = _img.shape[1] - old_xmin - 1
            roi_info[i]['roi_blob'] = temp_rois * abs(roi_info[i]['scaled_ratio'])
    elif cfg.TEST.ROI_POLICY == 'assign':
        all_rois = []
        all_scores = []
        for i in xrange(len(roi_info)):
            im_width = roi_info[i]['im_shape'][1]
            if roi_info[i]['scaled_ratio'] < 0:
                old_xmin = roi_info[i]['roi_blob'][:, 1].copy()
                old_xmax = roi_info[i]['roi_blob'][:, 3].copy()
                roi_info[i]['roi_blob'][:, 1] = im_width - old_xmax - 1
                roi_info[i]['roi_blob'][:, 3] = im_width - old_xmin - 1
            all_rois.extend(roi_info[i]['roi_blob'] / abs(roi_info[i]['scaled_ratio']))
            all_scores.extend(roi_info[i]['roi_scores'])
        all_rois = np.asarray(all_rois)
        all_scores = np.asarray(all_scores)

        scaled_ratios = [roi_info[_]['scaled_ratio'] for _ in xrange(len(roi_info))]

        if cfg.TEST.IMAGE_FLIP:
            assigned_rois = [[] for _ in xrange(len(roi_info) / 2)]
            assigned_scores = [[] for _ in xrange(len(roi_info) / 2)]
            target_scale = [BASE_SIZE / abs(x) for x in scaled_ratios[::2]]
        else:
            assigned_rois = [[] for _ in xrange(len(roi_info))]
            assigned_scores = [[] for _ in xrange(len(roi_info))]
            target_scale = [BASE_SIZE / abs(x) for x in scaled_ratios]
        for i in xrange(len(all_rois)):
            roi_scale = np.sqrt(float((all_rois[i][3] - all_rois[i][1]) * (all_rois[i][4] - all_rois[i][2])))
            k = np.argsort([abs(x - roi_scale) for x in target_scale])[0]
            assigned_rois[k].append(all_rois[i])
            assigned_scores[k].append(all_scores[i])
        for i in xrange(len(roi_info)):  # replace old rois
            if cfg.TEST.IMAGE_FLIP:
                temp_rois = np.asarray(assigned_rois[int(i / 2)])
                roi_info[i]['roi_scores'] = np.asarray(assigned_scores[int(i / 2)])
            else:
                temp_rois = np.asarray(assigned_rois[i])
                roi_info[i]['roi_scores'] = np.asarray(assigned_scores[i])
            if roi_info[i]['scaled_ratio'] < 0 and len(temp_rois) != 0:
                old_xmin = temp_rois[:, 1].copy()
                old_xmax = temp_rois[:, 3].copy()
                temp_rois[:, 1] = _img.shape[1] - old_xmax - 1
                temp_rois[:, 3] = _img.shape[1] - old_xmin - 1
            roi_info[i]['roi_blob'] = temp_rois * abs(roi_info[i]['scaled_ratio'])
            print '({})'.format(str(temp_rois.shape)),
    else:
        print 'ROI_POLICY error!'
        exit(0)

    for i in xrange(len(roi_info)):
        im_shape = roi_info[i]['im_shape']
        # rcnn
        if len(roi_info[i]['roi_blob']) == 0:
            continue
        scores, boxes, box_deltas = batch_rcnn_process(net_rcnn, roi_info[i]['roi_feature_blob'],
                                                       roi_info[i]['roi_blob'])
        if cfg.TEST.BBOX_REG:
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_shape)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        scores_list.extend(scores)
        if roi_info[i]['scaled_ratio'] > 0:
            pred_boxes_list.extend(pred_boxes / abs(roi_info[i]['scaled_ratio']))
        if roi_info[i]['scaled_ratio'] < 0:
            temp_boxes = pred_boxes.reshape((len(pred_boxes), -1, 4))
            temp_xmins = temp_boxes[:, :, 0].copy()
            temp_xmaxs = temp_boxes[:, :, 2].copy()
            temp_boxes[:, :, 0] = im_shape[1] - temp_xmaxs - 1
            temp_boxes[:, :, 2] = im_shape[1] - temp_xmins - 1
            pred_boxes_list.extend(temp_boxes.reshape((len(pred_boxes), -1)) / abs(roi_info[i]['scaled_ratio']))
    scores_list = np.asarray(scores_list)
    pred_boxes_list = np.asarray(pred_boxes_list)
    if cfg.TEST.ROI_POLICY == 'merge':
        det_scores = np.mean(scores_list.reshape((len(roi_info), -1, NUM_CLS)), axis=0)
        if cfg.TEST.AGNOSTIC:
            det_pred_boxes = np.mean(pred_boxes_list.reshape((len(roi_info), -1, 2 * 4)), axis=0)
        else:
            det_pred_boxes = np.mean(pred_boxes_list.reshape((len(roi_info), -1, NUM_CLS * 4)), axis=0)
    elif cfg.TEST.ROI_POLICY == 'assign':
        if cfg.TEST.IMAGE_FLIP:
            tmp_scores = []
            tmp_pred_boxes = []
            margin = 0
            for i in xrange(len(roi_info) / 2):
                tmp_scores.extend(
                    np.mean(scores_list[margin:margin + 2 * len(roi_info[2 * i]['roi_blob'])].
                            reshape((2, -1, NUM_CLS)), axis=0))
                if cfg.TEST.AGNOSTIC:
                    tmp_pred_boxes.extend(np.mean(pred_boxes_list[margin:margin + 2 * len(roi_info[2 * i]['roi_blob'])].
                                                  reshape((2, -1, 2 * 4)), axis=0))
                else:
                    tmp_pred_boxes.extend(np.mean(pred_boxes_list[margin:margin + 2 * len(roi_info[2 * i]['roi_blob'])].
                                                  reshape((2, -1, NUM_CLS * 4)), axis=0))
                margin += 2 * len(roi_info[2 * i]['roi_blob'])
            det_scores = np.asarray(tmp_scores)
            det_pred_boxes = np.asarray(tmp_pred_boxes)
        else:
            det_scores = scores_list
            det_pred_boxes = pred_boxes_list
    else:
        det_scores = scores_list
        det_pred_boxes = pred_boxes_list

    return det_scores, det_pred_boxes


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)  # x1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)  # y1 >= 0
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)  # x2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)  # y2 < im_shape[0]
    return boxes


def bbox_vote(dets_NMS, dets_all, thresh=0.5):
    dets_voted = np.zeros_like(dets_NMS)  # Empty matrix with the same shape and type

    _overlaps = bbox_overlaps(
        np.ascontiguousarray(dets_NMS[:, 0:4], dtype=np.float),
        np.ascontiguousarray(dets_all[:, 0:4], dtype=np.float))

    # for each survived box
    for i, det in enumerate(dets_NMS):
        dets_overlapped = dets_all[np.where(_overlaps[i, :] >= thresh)[0]]
        assert (len(dets_overlapped) > 0)

        boxes = dets_overlapped[:, 0:4]
        scores = dets_overlapped[:, 4]

        out_box = np.dot(scores, boxes)

        dets_voted[i][0:4] = out_box / sum(scores)  # Weighted bounding boxes
        dets_voted[i][4] = det[4]  # Keep the original score

        # Weighted scores (if enabled)
        if cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE > 1:
            n_agreement = cfg.TEST.BBOX_VOTE_N_WEIGHTED_SCORE
            w_empty = cfg.TEST.BBOX_VOTE_WEIGHT_EMPTY

            n_detected = len(scores)

            if n_detected >= n_agreement:
                top_scores = -np.sort(-scores)[:n_agreement]
                new_score = np.average(top_scores)
            else:
                new_score = np.average(scores) * (n_detected * 1.0 + (n_agreement - n_detected) * w_empty) / n_agreement

            dets_voted[i][4] = min(new_score, dets_voted[i][4])

    return dets_voted


def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep


def rpn_process(_rpn, img, scale=1.0):
    im_h, im_w = img.shape[:2]
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)
    _rpn.blobs['data'].reshape(*img.shape)
    rpn_input_blob = {'data': img, 'rois': None}

    if cfg.TEST.HAS_RPN:
        rpn_input_blob['im_info'] = np.array([[im_h, im_w, scale]], dtype=np.float32)
        _rpn.blobs['im_info'].reshape(*rpn_input_blob['im_info'].shape)

    # do forward
    rpn_forward_kwargs = {'data': rpn_input_blob['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        rpn_forward_kwargs['im_info'] = rpn_input_blob['im_info'].astype(np.float32, copy=False)
    _rpn.forward(**rpn_forward_kwargs)

    roi_feature_blob = []
    for _ in ROI_FEATURE:
        roi_feature_blob.append(_rpn.blobs[_].data[...].copy())
    rois_blob = _rpn.blobs['rois'].data[...]
    roi_scores = _rpn.blobs[RPN_SCORES].data[...]

    return roi_feature_blob, rois_blob, roi_scores


def batch_rcnn_process(_rcnn, roi_feature_blob, rois_blob, scale=1.0):
    scores = []
    boxes = []
    box_deltas = []
    roi_num = rois_blob.shape[0]
    batch_num = int(roi_num / cfg.TEST.RCNN_BATCH_SIZE)
    batch_array = np.asarray(np.append(int(cfg.TEST.RCNN_BATCH_SIZE) * np.arange(batch_num + 1),
                                       [] if not int(roi_num) % int(cfg.TEST.RCNN_BATCH_SIZE) else [roi_num],
                                       axis=0), dtype=np.int)

    for i in xrange(len(batch_array) - 1):
        for _ in xrange(len(ROI_FEATURE)):
            _rcnn.blobs[ROI_FEATURE[_]].reshape(*(roi_feature_blob[_].shape))
            _rcnn.blobs[ROI_FEATURE[_]].data[...] = roi_feature_blob[_]  # bug when multi-scale testing
        _rcnn.blobs['rois'].reshape(*(rois_blob[batch_array[i]:batch_array[i + 1], :].shape))
        _rcnn.blobs['rois'].data[...] = rois_blob[batch_array[i]:batch_array[i + 1], :]
        rcnn_output_blob = _rcnn.forward()

        if cfg.TEST.HAS_RPN:
            rois = _rcnn.blobs['rois'].data.copy()
            boxes.extend(rois[:, 1:5] / scale)
        scores.extend(rcnn_output_blob['cls_prob'].copy())
        box_deltas.extend(rcnn_output_blob['bbox_pred'].copy())

    scores = np.asarray(scores)
    scores = scores.reshape(*scores.shape[:2])
    boxes = np.asarray(boxes)
    box_deltas = np.asarray(box_deltas)
    box_deltas = box_deltas.reshape(*box_deltas.shape[:2])

    return scores, boxes, box_deltas


def multi_scale_operator(_img):
    im_shape = _img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    scaled_ims = []
    scaled_ratios = []

    for i in xrange(len(cfg.TEST.SCALES)):
        scale_ratio = float(cfg.TEST.SCALES[i]) / float(im_size_min)
        if np.round(scale_ratio * im_size_max) > float(cfg.TEST.MAX_SIZE[i]):
            scale_ratio = float(cfg.TEST.MAX_SIZE[i]) / float(im_size_max)
        im = cv2.resize(_img, None, None, fx=scale_ratio, fy=scale_ratio,
                        interpolation=cv2.INTER_LINEAR)
        scaled_ims.append(im)
        scaled_ratios.append(scale_ratio)
        if cfg.TEST.IMAGE_FLIP:
            scaled_ims.append(cv2.resize(_img[:, ::-1], None, None, fx=scale_ratio, fy=scale_ratio,
                                         interpolation=cv2.INTER_LINEAR))
            scaled_ratios.append(-scale_ratio)

    return scaled_ims, scaled_ratios


def multi_crop_operator(_img):
    """

    :param _img: image
    :return: nine crop images and offsets
    """
    im_shape = _img.shape
    im_h, im_w = im_shape[0:2]

    crop_ims = []

    crop_h = int(cfg.TEST.CROP_SIZE_RATIO * im_h)
    crop_w = int(cfg.TEST.CROP_SIZE_RATIO * im_w)

    yy = int((im_h - crop_h) / 2.0)
    xx = int((im_w - crop_w) / 2.0)

    crop_offsets = [[0, 0, crop_w, crop_h], [xx, 0, xx + crop_w, crop_h], [im_w - crop_w, 0, im_w, crop_h],
                    [0, yy, crop_w, yy + crop_h], [xx, yy, xx + crop_w, yy + crop_h],
                    [im_w - crop_w, yy, im_w, yy + crop_h],
                    [0, im_h - crop_h, crop_w, im_h], [xx, im_h - crop_h, xx + crop_w, im_h],
                    [im_w - crop_w, im_h - crop_h, im_w, im_h]]

    for i in crop_offsets:
        crop_ims.append(
            cv2.resize(_img[i[1]:i[3], i[0]:i[2], :], (im_w, im_h), interpolation=cv2.INTER_LINEAR))

    return crop_ims, crop_offsets


def det_post_process(scores, boxes, vote=False, nms_type=0, nms_thresh=0.4, conf_thresh=0.05):
    """

    :param scores: numpy.ndarray, (N*NUM_CLS)
    :param boxes: numpy.ndarray, (N*(4*NUM_CLS))
    :param vote: bool, default False
    :param nms_type: int, (0 for common nms, 1 for linear nms, 2 for gaussian nms)
    :param nms_thresh:  float, (nms threshold, default is 0.3, 0.4 is better)
    :param conf_thresh: float, default is 0.05
    :return: im_boxes: numpy.ndarray, (NUM_CLS*M*5, M is uncertainty)
    """
    im_boxes = [[] for _ in xrange(NUM_CLS)]
    for i in xrange(1, NUM_CLS):
        inds = np.where(scores[:, i] > conf_thresh)[0]
        cls_scores = scores[inds, i]
        if cfg.TEST.AGNOSTIC:
            cls_boxes = boxes[inds, 4:8]
        else:
            cls_boxes = boxes[inds, i * 4:(i + 1) * 4]
        # cls_boxes = boxes[inds, i * 4:(i + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

        nms_dets = cls_dets.copy()
        nms_keep = soft_nms(nms_dets, Nt=nms_thresh, method=nms_type)
        NMSed = nms_dets[nms_keep, :]
        dets_diff = [_ for _ in NMSed.tolist() if _ not in cls_dets.tolist()]

        keep = soft_nms(cls_dets, Nt=nms_thresh, method=0)
        dets_NMSed = cls_dets[keep, :]
        if vote:
            VOTEed = bbox_vote(dets_NMSed, cls_dets)
        else:
            VOTEed = dets_NMSed
        dets_diff.extend(VOTEed.tolist())

        im_boxes[i] = np.asarray(dets_diff).reshape((-1, 5))

    return im_boxes


if __name__ == '__main__':
    eval_batch(max_per_image=100)
