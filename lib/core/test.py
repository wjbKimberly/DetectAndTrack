##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Test a Fast R-CNN network on an image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from collections import defaultdict

from caffe2.python import core, workspace
import pycocotools.mask as mask_util

from core.config import cfg
import utils.boxes as box_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
from utils.timer import Timer
from core.nms_wrapper import nms, soft_nms
import utils.blob as blob_utils
import modeling.FPN as fpn

import logging
logger = logging.getLogger(__name__)

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations.
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (list of ndarray): a list of color images in BGR order. In case of
        video it is a list of frames, else is is a list with len = 1.

    Returns:
        blob (ndarray): a data blob holding an image pyramid (or video pyramid)
        im_scale_factors (ndarray): array of image scales (relative to im) used
            in the image pyramid
    """
    all_processed_ims = []  # contains a a list for each frame, for each scale
    all_im_scale_factors = []
    for frame in im:
        processed_ims, im_scale_factors = blob_utils.prep_im_for_blob(
            frame, cfg.PIXEL_MEANS, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE)
        all_processed_ims.append(processed_ims)
        all_im_scale_factors.append(im_scale_factors)
    # All the im_scale_factors will be the same, so just take the first one
    for el in all_im_scale_factors:
        assert(all_im_scale_factors[0] == el)
    im_scale_factors = all_im_scale_factors[0]
    # Now get all frames with corresponding scale next to each other
    processed_ims = []
    for i in range(len(all_processed_ims[0])):
        for frames_at_specific_scale in all_processed_ims:
            processed_ims.append(frames_at_specific_scale[i])
    # Now processed_ims contains
    # [frame1_scale1, frame2_scale1..., frame1_scale2, frame2_scale2...] etc
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        # Works for tubes as well, as it uses the first box's area -- which is
        # a reasonable approx for the tube area
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max)


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if cfg.MODEL.FASTER_RCNN and rois is None:
        blobs['im_info'] = np.array(
            [[blobs['data'].shape[-2], blobs['data'].shape[-1],
              im_scale_factors[0]]],
            dtype=np.float32)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect_bbox(model, im, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        raise NotImplementedError('Can not handle tubes, need to extend dedup')
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True)
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)

    # dump workspace blobs (debugging)
    # if 0:
    #    from utils.io import robust_pickle_dump
    #    import os, sys
    #    saved_blobs = {}
    #    ws_blobs = workspace.Blobs()
    #    for dst_name in ws_blobs:
    #        ws_blob = workspace.FetchBlob(dst_name)
    #        saved_blobs[dst_name] = ws_blob
    #    det_file = os.path.join('/tmp/output/data_dump_inflT1.pkl')
    #    robust_pickle_dump(saved_blobs, det_file)
    #    logger.info("DUMPED BLOBS")
    #    sys.exit(0)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        assert len(im_scales) == 1, \
            'Only single-image / single-scale batch implemented'
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they were
        # trained as linear SVMs
        scores = workspace.FetchBlob(core.ScopedName('cls_score')).squeeze()
    else:
        # use softmax estimated probabilities
        scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])
    time_dim = boxes.shape[-1] // 4

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4 * time_dim:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im[0].shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    return scores, pred_boxes, im_scales


def im_detect_bbox_hflip(model, im, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    # im is a list now, to be compat with video case
    im_hf = [e[:, ::-1, :] for e in im]
    # Since all frames would be same shape, just take values from 1st
    im_width = im[0].shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scales = im_detect_bbox(
        model, im_hf, box_proposals_hf)

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scales


def im_detect_bbox_scale(
        model, im, scale, max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, box_proposals)
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(
            model, im, box_proposals)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = [image_utils.aspect_ratio_rel(el, aspect_ratio) for el in im]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model, im_ar, box_proposals_ar)
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(
            model, im_ar, box_proposals_ar)

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _im_scales_hf = im_detect_bbox_hflip(
            model, im, box_proposals)
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals)
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True)
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals)
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True)
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scales_i = im_detect_bbox(model, im, box_proposals)
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR))

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR))

    return scores_c, boxes_c, im_scales_i


def im_detect_mask(model, im_scales, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scales)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_hflip(model, im, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    masks_hf = im_detect_mask(model, im_scales, boxes_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes masks at the given scale."""

    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform mask detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        masks_scl = im_detect_mask_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        masks_scl = im_detect_mask(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        masks_ar = im_detect_mask(model, im_scales, boxes_ar)

    return masks_ar


def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scales_i = im_conv_body_only(model, im)
    masks_i = im_detect_mask(model, im_scales_i, boxes)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(model, im, boxes)
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True)
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True)
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':
        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))
        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR))

    return masks_c


def im_detect_keypoints(model, im_scales, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'
    time_dim = boxes.shape[-1] // 4

    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros(
            (0, time_dim * cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scales)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_hflip(model, im, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    heatmaps_hf = im_detect_keypoints(model, im_scales, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""

    # Store the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        heatmaps_scl = im_detect_keypoints(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
        model, im, aspect_ratio, boxes, hflip=False):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        heatmaps_ar = im_detect_keypoints(model, im_scales, boxes_ar)

    return heatmaps_ar


def im_detect_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """
    assert not cfg.TEST.KPS_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []

    # Compute the heatmaps for the original image (identity transform)
    im_scales = im_conv_body_only(model, im)
    heatmaps_i = im_detect_keypoints(model, im_scales, boxes)
    heatmaps_ts.append(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(model, im, boxes)
        heatmaps_ts.append(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        max_size = cfg.TEST.KPS_AUG.MAX_SIZE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, max_size, boxes)
        heatmaps_ts.append(heatmaps_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, max_size, boxes, hflip=True)
            heatmaps_ts.append(heatmaps_scl_hf)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes)
        heatmaps_ts.append(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True)
            heatmaps_ts.append(heatmaps_ar_hf)

    # Combine the predicted heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        heatmaps_c = np.mean(heatmaps_ts, axis=0)
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        heatmaps_c = np.amax(heatmaps_ts, axis=0)
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR))

    return heatmaps_c


def box_results_with_nms_and_limit(is_opt_flag,scores, boxes,nms_iou):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    time_dim = boxes.shape[-1] // (num_classes * 4)
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4 * time_dim:(j + 1) * 4 * time_dim]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            # Not implemented for time_dim > 1
            nms_dets = soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=nms_iou,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD)
        else:
            keep = nms(dets_j, nms_iou)
            is_opt_flag=is_opt_flag[keep,:]
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets, dets_j, cfg.TEST.BBOX_VOTE.VOTE_TH)
        cls_boxes[j] = nms_dets
        
    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(
                image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]
                is_opt_flag=is_opt_flag[keep,:]
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return is_opt_flag,scores, boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms

def keypoint_results(is_opt_flag,cls_boxes, pred_heatmaps, ref_boxes,scores):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()

    # handle the tubes
    assert pred_heatmaps.shape[1] % cfg.KRCNN.NUM_KEYPOINTS == 0, \
        'Heatmaps must be 17xT'
    time_dim = pred_heatmaps.shape[1] // cfg.KRCNN.NUM_KEYPOINTS
    assert time_dim == ref_boxes.shape[-1] // 4, 'Same T for boxes and keypoints'
    all_xy_preds = []
    
#     for t in range(time_dim):
#         all_xy_preds.append(keypoint_utils.heatmaps_to_keypoints(
#             pred_heatmaps[:, t * cfg.KRCNN.NUM_KEYPOINTS:
#                           (t + 1) * cfg.KRCNN.NUM_KEYPOINTS, ...],
#             ref_boxes[:, t * 4: (t + 1) * 4]))

    ################ jianbo add ###############
    #attention that I set the value t equal to zero !!!!!
    t=0
    kp_predictions= keypoint_utils.heatmaps_to_keypoints(
            pred_heatmaps[:, t * cfg.KRCNN.NUM_KEYPOINTS:
                          (t + 1) * cfg.KRCNN.NUM_KEYPOINTS, ...],
            ref_boxes[:, t * 4: (t + 1) * 4])
    
    keep=delete_optical_by_oks(is_opt_flag,kp_predictions,ref_boxes[:,t * 4: (t + 1) * 4])
    is_opt_flag=is_opt_flag[keep,:]
    kp_predictions=kp_predictions[keep,:]
    cls_boxes[person_idx]=cls_boxes[person_idx][keep,:]
    ref_boxes=ref_boxes[keep,:]
    scores=scores[keep]
    
    all_xy_preds.append(kp_predictions)
    
    ################ jianbo add ###############
    xy_preds = np.concatenate(all_xy_preds, axis=-1)
    
    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        raise NotImplementedError('Handle tubes')
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    
    del_num=0
    for ki in keep:
        if ki==False:
            del_num+=1
    
    return cls_keyps,all_xy_preds,is_opt_flag,cls_boxes,ref_boxes,scores,del_num

####################### optical flow bbox propagation begin#######################
import cv2
import numpy as np
from convert.box import compute_boxes_from_pose,expand_boxes
import utils.image as image_utils
import copy

#use cv2 version, expected to add flownet2_version
# when use opencv, pay attention to transfer NCHW to NHWC for use with OpenCV

def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """ Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)
    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[0, :] - src_keypoints[0, :]
    dy = dst_keypoints[1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e)) / e.shape[0]

    return e

# attention that I does not do distance normalize
def compute_our_kps_dis(src_keypoints, src_roi, dst_keypoints, dst_roi):
    #kps shape is [4,17]
    
    src_keypoints_tmp=src_keypoints.transpose([1,0])
    dst_keypoints_tmp=dst_keypoints.transpose([1,0])
    #kps_tmp shape is [17,4]
    
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)
    dst_area = (dst_roi[2] - dst_roi[0] + 1) * (dst_roi[3] - dst_roi[1] + 1)
    norm_area = src_area+dst_area
    
    norm_area=1
    
    judge_src=src_keypoints_tmp[:,2]>=cfg.EVAL.EVAL_MPII_KPT_THRESHOLD*2
    judge_dst=dst_keypoints_tmp[:,2]>=cfg.EVAL.EVAL_MPII_KPT_THRESHOLD*2
    index=np.where(np.logical_and.reduce((judge_src,judge_dst) ))[0]
    
#     print("heatamp:",src_keypoints_tmp[:,2],dst_keypoints_tmp[:,2])
#     print(judge_src,judge_dst,index)
    dx = dst_keypoints_tmp[index,0] - src_keypoints_tmp[index,0]
    dy = dst_keypoints_tmp[index,1] - src_keypoints_tmp[index,1]
    
    num=len(dx)
    
    e=(np.sqrt( (dx**2 + dy**2) )/norm_area).sum()/num
    
    if np.isnan(e):
        e=cfg.TEST.DEL_OPT_BBOX_OKS+10
    
    return e
    
    

def _prune_bad_detections(is_opt_flag,scores, boxes, image_height,image_width):
    from core.tracking_engine import _get_high_conf_boxes,_get_big_inside_image_boxes
    """
    Keep only the boxes/poses that correspond to confidence > conf (float),
    and are big enough inside the image.
    """
    conf=cfg.TRACKING.CONF_FILTER_INITIAL_DETS
    json_data={
        'height':image_height,
        "width":image_width
    }
    sel = np.where(np.logical_and.reduce((
            _get_high_conf_boxes(np.hstack((boxes,scores)), conf),
            _get_big_inside_image_boxes(boxes, json_data),
        )))[0]
    if len(boxes)-len(sel)>0:
        print("prune_bad_detections:",len(boxes)-len(sel))
    is_opt_flag=is_opt_flag[sel]
    boxes = boxes[sel]
    scores=scores[sel]
    
    return is_opt_flag,scores, boxes

def get_cls_boxes(boxes,scores):
    cls_tmp = np.hstack((boxes, scores)).astype(np.float32, copy=False)
    
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    
    person_idx = keypoint_utils.get_person_class_index()    
    cls_boxes[person_idx]=cls_tmp 
        
    return cls_boxes

    

def delete_optical_by_oks(is_opt_flag,kp_predictions, rois):
    
    """Nms based on kp predictions."""
    thresh=cfg.TEST.DEL_OPT_BBOX_OKS
    keep=np.full((kp_predictions.shape[0]),fill_value=True)
    if thresh==-1.0:
        return keep
    for i,(opt_kps_i,opt_box_i) in enumerate( zip(kp_predictions,rois) ):
        if is_opt_flag[i]:
            for j, (det_kps_i,det_box_i) in enumerate( zip(kp_predictions,rois) ):
                if not is_opt_flag[j]:
                    sim_i_j=compute_our_kps_dis(
                            opt_kps_i, opt_box_i, det_kps_i,det_box_i)
                    
                    print("sim:",i,j,sim_i_j)
                    if sim_i_j<=thresh:
                        keep[i]=False
                        break
#                     sim_i_j=compute_oks(
#                             opt_kps_i, opt_box_i, det_kps_i,det_box_i)
#                     if sim_i_j>thresh:
#                         keep[i]=False
#                         break
    
    return keep

# Assume that the remaining boxes all been passed by nms operation, the scores of them are highly closed to 0.999
# we use iou to filter the bboxes
def get_area(dets):
    dets_x1 = dets[:, 0]
    dets_y1 = dets[:, 1]
    dets_x2 = dets[:, 2]
    dets_y2 = dets[:, 3]
    areas = (dets_x2 - dets_x1 + 1) * (dets_y2 - dets_y1 + 1)
    return areas 
def get_inter(opt_i,det_i):
    ix1,iy1,ix2,iy2 = det_i[0],det_i[1],det_i[2],det_i[3]
    jx1,jy1,jx2,jy2 = opt_i[0],opt_i[1],opt_i[2],opt_i[3]
    xx1 = max(ix1,jx1)
    yy1 = max(iy1,jy1)
    xx2 = min(ix2,jx2)
    yy2 = min(iy2,jy2)
    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    return inter
def iou_drive_nms(opts,dets):
    # get iou
    iou_metrics=np.zeros((opts.shape[0],dets.shape[0]), dtype=np.float32)
    areas_dets= get_area(dets)
    areas_opts= get_area(opts)
    suppressed_opt =np.zeros((opts.shape[0]), dtype=np.int)
    suppressed_det =np.zeros((dets.shape[0]), dtype=np.int)
    suppressed_iou =np.zeros((opts.shape[0],dets.shape[0]), dtype=np.int)
    
    for i,det_i in enumerate(opts):
        for j,opt_i in enumerate(dets):
            inter =get_inter(opt_i,det_i)
            ovr = inter / (areas_opts[i]+areas_dets[j] - inter)
            iou_metrics[i][j]=ovr
    abandon_cnt=0
    add_opt_bbox_num=int(cfg.TEST.FORCE_ADD_OPTICAL*dets.shape[0])
    
    pos_list=np.argsort(-iou_metrics)
    for i in range(opts.shape[0]-add_opt_bbox_num):
        pos=pos_list[i]
        w=opts.shape[0]
        x_int = pos % w
        y_int = (pos - x_int) // w
        suppressed_opt[x_int]=1
        
        #######################!!!!!!!!!! under
                    
    return np.where(suppressed_opt == 0)[0],np.where(suppressed_det == 0)[0]

def optical_detect_nms(opt_boxes,opt_scores,det_boxes,det_scores):
    num_classes = cfg.MODEL.NUM_CLASSES
    time_dim = 1
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        
        opt_scores_j = opt_scores[:, j]
        opt_boxes_j = opt_boxes[:, j * 4 * time_dim:(j + 1) * 4 * time_dim]
        
        
        det_scores_j = det_scores[:, j]
        det_boxes_j = det_boxes[:, j * 4 * time_dim:(j + 1) * 4 * time_dim]
        
        keep_opt,keep_det =iou_drive_nms(opt_boxes_j,det_boxes_j)
        
        #expand bboxes propagated by optical flow 
        if cfg.TEST.EXTEND_OPT_BBOX>1.0:
            opt_boxes_j=expand_boxes(opt_boxes_j, cfg.TEST.EXTEND_OPT_BBOX)
        
        opt_j = np.hstack((opt_boxes_j, opt_scores_j[:, np.newaxis])).astype(
            np.float32, copy=False)
        det_j = np.hstack((det_boxes_j, det_scores_j[:, np.newaxis])).astype(
            np.float32, copy=False)
        
        nms_dets = det_j[keep_det, :]
        nms_opts = opt_j[keep_opt, :]
        
        cls_boxes[j] = np.vstack((nms_dets,nms_opts))
        
        
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes

def get_padding_flownet2(pim1_path, pim2_path):  
    
    import torch
    import sys
    import os.path as osp
    import os
    sys.path.insert(0, osp.join(os.getcwd(), '../flownet2-pytorch-qiu/'))
    sys.path.insert(0, osp.join(os.getcwd(), '../flownet2-pytorch-qiu/utils'))
    from models import FlowNet2,FlowNet2C,FlowNet2S #the path is depended on where you create this module
    from frame_utils import read_gen#the path is depended on where you create this module 
    from torch.autograd import Variable
    import argparse
    
    args = argparse.Namespace(fp16=False, rgb_max=255.)
    args.grads = {}
    model_path=cfg.TEST.OPTICAL_MODEL_PATH
    #initial a Net
    net = FlowNet2(args).cuda()
#     net = FlowNet2S(args).cuda()
    #load the state_dict
    dict = torch.load(model_path)
    net.load_state_dict(dict["state_dict"])
    
    #load the image pair, you can find this operation in dataset.py
    pim1 = read_gen(pim1_path)
    pim2 = read_gen(pim2_path)
    
    images = [pim1, pim2]
    
    image_size = pim1.shape[:2]

    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    
    #process the image pair to obtian the flow 
    im=Variable(im)
    result = net(im).squeeze()

    data = result.data.cpu().numpy().transpose(1, 2, 0)
    return data

def get_optical_flow(entry,pre_entry):
    # optical_choice=0 means using cv2 version, =1 means flownet2
    pim1_path=pre_entry["image"][0]
    pim2_path=entry["image"][0]
    if cfg.TEST.OPTICAL_CHOICE==0:
        pim1 = cv2.imread(pim1_path)
        pim2 = cv2.imread(pim2_path)

        pim1_gray = cv2.cvtColor(pim1,cv2.COLOR_BGR2GRAY)
        pim2_gray = cv2.cvtColor(pim1,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(pim1_gray, pim2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
    elif cfg.TEST.OPTICAL_CHOICE==1:
        flow=get_padding_flownet2(pim1_path, pim2_path)
        
    elif cfg.TEST.OPTICAL_CHOICE==2:
        flow=np.zeros((entry["height"],entry["width"],2))
    return flow

def _shift_using_flow(poses, flow):
    # too large flow => must be a shot boundary. So don't transfer poses over
    # shot boundaries
    res = []
    for pose in poses:
        res_pose = copy.deepcopy(pose)
        x = np.round(copy.deepcopy(pose[0]))
        y = np.round(copy.deepcopy(pose[1]))
        if x<0 or y<0 or x>=flow.shape[1] or y>=flow.shape[0]:
        # Since I wasn't able to move the points for which a valid flow vector
        # did not exist, set confidence to 0
            res_pose[3] = 0
            res_pose[2] = 0
            res_pose[0] = 0
            res_pose[1] = 0
        else:
            delta_x = flow[y.astype('int'), x.astype('int'), 0]
            delta_y = flow[y.astype('int'), x.astype('int'), 1]
            res_pose[0] += delta_x
            res_pose[1] += delta_y
        res.append(res_pose)
    return res

def warp_keypoints_from_flow(flow,pre_keypoints):
    import copy
    flow=copy.deepcopy(flow)
    #transpose keypoints from (4,17) to (17,4) and we just use [0,1,3] indices as [x,y,prob]
    pre_keypoints_process=pre_keypoints.transpose([1,0])
    optical_keypoints=copy.deepcopy(pre_keypoints)
    optical_keypoints_process=_shift_using_flow(pre_keypoints_process,flow)
    ##transpose keypoints from (17,3) to (3,17) and heatmap line on the row2
    optical_keypoints=np.asarray(optical_keypoints_process).transpose([1,0])
    
    return optical_keypoints

def get_optical_bbox(entry,pre_entry,pre_keypoints,pre_bbox,pre_scores):
    im = image_utils.read_image_video(entry)
    pre_im=image_utils.read_image_video(pre_entry)[0]
    
    optical_keypoints=[i for i in pre_keypoints]
    optical_bbox_scores=copy.deepcopy(pre_scores)
    optical_bbox=copy.deepcopy(pre_bbox)
    cv2_flow= get_optical_flow(entry,pre_entry)
    # pre_keypoints[0] is background, ignore it
    for i,obj_kps_i in enumerate(pre_keypoints[1]):
        #pre_keypoints[1][i].shape=(4,17)
        optical_keypoints[1][i]= warp_keypoints_from_flow(cv2_flow,pre_keypoints[1][i])
    #for compute_boxes_from_pose ,the input should be (frames,17,3) as COCO style, 
    #The third dimension means the visible label. We only consider the points that are marked "2", i.e. labeled and visible
    #Here I transpose keypoints from (num,4,17) to (num,17,4) and we just use [0,1,3] indices as [x,y,prob]
    use_keypoints=np.asarray(optical_keypoints[1]).transpose([0,2,1])[:,:,[0,1,3]]
    # TODO: Here I regard all keypoints as visible
    # if prob >cfg.TEST.DROP_KPS_SCORE assume this point is visible
#     use_keypoints[:,:,2]=2
    for obj_ind,obj_i in enumerate(use_keypoints):
        for k_ind,k_i in enumerate(obj_i):
            if k_i[2]>cfg.TEST.DROP_OPTICAL_KPS_SCORE:
                use_keypoints[obj_ind][k_ind][2]=2
    # compute_boxes_from_pose get multiple frames as input,also the output
    optical_bbox=compute_boxes_from_pose([use_keypoints])[0]
    # convert COCO style[x,y,w,h] into our style[x1,y1,x2,y2]
    for i,bbox_i in enumerate(optical_bbox):
        x1,y1,w,h=bbox_i[0],bbox_i[1],bbox_i[2],bbox_i[3]
        x2,y2=bbox_i[0]+bbox_i[2],bbox_i[1]+bbox_i[3]
#         x1=np.maximum(np.minimum(x1, pre_im.shape[1] - 1), 0) 
#         x2=np.maximum(np.minimum(x2, pre_im.shape[1] - 1), 0) 
#         y1=np.maximum(np.minimum(y1, pre_im.shape[0] - 1), 0) 
#         y2=np.maximum(np.minimum(y2, pre_im.shape[0] - 1), 0) 
        optical_bbox[i]=[x1,y1,x2,y2]
    optical_bbox=np.asarray(optical_bbox)
    
    return optical_bbox_scores,optical_bbox

####################### optical flow bbox propagation end #######################

def im_detect_all(model, entry, box_proposals, timers=None,pre_entry=None,pre_keypoints=None,pre_bbox=None,pre_scores=None):
    im = image_utils.read_image_video(entry)
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.COMPETITION_MODE:
        scores, boxes, im_scales = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scales = im_detect_bbox(model, im, box_proposals)
    timers['im_detect_bbox'].toc()
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    
    is_opt_flag=np.full((scores.shape[0],1),fill_value=False)
    is_opt_flag,scores, boxes= box_results_with_nms_and_limit(is_opt_flag,scores, boxes,cfg.TEST.NMS)
    scores=scores.reshape((len(scores),1))
    is_opt_flag,scores, boxes=_prune_bad_detections(is_opt_flag,scores, boxes, entry["height"],entry["width"])
    ####################### optical flow bbox propagation begin#######################
    #of course pre_keypoints!=None and pre_bbox!=None and pre_scores!=None
    # pay attention that boxes=[R* 4*K ] ,every bbox is in the format of [x1, y1, x2, y2]
    # pay attention that scores=[R* K ]
    tmp_dic={
            "image_name":entry["image"][0],
            "size":(entry["width"],entry["height"]),
            "frame_id":entry["frame_id"],
            "detect":boxes.tolist(),
            "det_scores":scores.tolist()        
        }
    
    import json
    if cfg.TEST.OPTICAL_BBOX and pre_entry!=None and pre_keypoints!=None:
        pre_im=image_utils.read_image_video(pre_entry)
        optical_bbox_scores,optical_bbox=get_optical_bbox(entry,pre_entry,pre_keypoints,pre_bbox,pre_scores)
        optical_bbox_scores=optical_bbox_scores.reshape((len(optical_bbox_scores),1))
        
        is_opt_flag=np.full((optical_bbox_scores.shape[0],1),fill_value=True)
        is_opt_flag,optical_bbox_scores,optical_bbox=_prune_bad_detections(is_opt_flag,optical_bbox_scores,optical_bbox,entry["height"],entry["width"])
        tmp_dic["optical_bbox"]=optical_bbox.tolist()
        tmp_dic["optical_bbox_scores"]=optical_bbox_scores.tolist()
        
#         if cfg.TEST.FORCE_ADD_OPTICAL>0.0:
#             scores=np.hstack(( np.zeros_like(scores),scores))
#             boxes=np.hstack(( np.zeros_like(boxes),boxes ))
#             optical_bbox_scores=np.hstack(( np.zeros_like(optical_bbox_scores),optical_bbox_scores))
#             optical_bbox=np.hstack(( np.zeros_like(optical_bbox),optical_bbox ))
#             scores, boxes, cls_boxes =optical_detect_nms(optical_bbox,optical_bbox_scores,boxes,scores)
#         else:
        is_opt_flag=np.vstack( ( np.full((scores.shape[0],1),fill_value=False),
                                    np.full((optical_bbox_scores.shape[0],1),fill_value=True) ) )
        boxes=np.vstack(( boxes,optical_bbox ))
        scores=np.vstack(( scores,optical_bbox_scores ))
            # add 1 dimension for background
        
        if cfg.TEST.OPT_OKS_NMS:
            pass
        else:
            scores=np.hstack(( np.zeros_like(scores),scores))
            boxes=np.hstack(( np.zeros_like(boxes),boxes ))
#             print("1 detected:",boxes.shape,boxes[np.where(~is_opt_flag)].shape)
            is_opt_flag,scores, boxes= box_results_with_nms_and_limit(is_opt_flag,scores, boxes,cfg.TEST.NMS_OPTICAL)
#             print("2 detected:",boxes.shape,boxes[np.where(~is_opt_flag)].shape)
            
            
            scores=scores.reshape((len(scores),1))
            is_opt_flag,scores, boxes=_prune_bad_detections(is_opt_flag,scores, boxes, entry["height"],entry["width"])
            
    
        optical_num=boxes[np.where(is_opt_flag)].shape[0]
        
        tmp_dic["pre_bbox"]=pre_bbox.tolist()
        tmp_dic["pre_keypoints"]=[i.tolist() for i in pre_keypoints[1]]
        tmp_dic["nms_ans"]=boxes.tolist()
        tmp_dic["ans_scores"]=scores.tolist()       
        
    scores=scores.reshape((len(scores),1))
    cls_boxes=get_cls_boxes(boxes,scores)
    ####################### optical flow bbox propagation end#######################
    timers['misc_bbox'].toc()
    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        raise NotImplementedError('Handle tubes..')
        timers['im_detect_mask'].tic()
        if cfg.TEST.COMPETITION_MODE:
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            masks = im_detect_mask(model, im_scales, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.COMPETITION_MODE:
            heatmaps = im_detect_keypoints_aug(model, im, boxes)
        else:
            heatmaps = im_detect_keypoints(model, im_scales, boxes)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        
#         print("3 detected:",boxes.shape,boxes[np.where(~is_opt_flag)].shape)
        cls_keyps,keypoints,is_opt_flag,cls_boxes,boxes,scores,del_num = keypoint_results(is_opt_flag,cls_boxes, heatmaps, boxes,scores)

#         print("4 detected:",boxes.shape,boxes[np.where(~is_opt_flag)].shape)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None
        
    ####################### save tmp_dic #######################
    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        tmp_dic["keypoints"]=[]
        for person_i in keypoints[0]:
            xy_i=person_i.transpose([1,0]).tolist()
            for joint_index,ki in enumerate(xy_i):
                if xy_i[joint_index][2]< cfg.EVAL.EVAL_MPII_KPT_THRESHOLD*2:
                    xy_i[joint_index].append(-1)
                else:
                    xy_i[joint_index].append(joint_index)
            xy_i=np.asarray(xy_i).transpose([1,0]).tolist()
            tmp_dic["keypoints"].append(xy_i)
        
        if cfg.TEST.OPTICAL_BBOX and pre_entry!=None and pre_keypoints!=None:
            print("optical_num",optical_num,",del_num",del_num)
            
            tmp_dic["final_boxes"]=boxes.tolist()
            tmp_dic["final_scores"]=scores.tolist()
            
            tmp_dic["optical_num"]=optical_num
            tmp_dic["del_num"]=del_num
    #############debug####################
#     if "23754" in entry["image"][0] and int(entry["image"][0][-12:-4])==52:
#         assert(False and "Find it!")
    #############debug####################
    import re
    import os
    from core.config import get_log_dir_path
    dir_path=get_log_dir_path()
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if "PoseTrack" in tmp_dic["image_name"]:
        patt=r"data/PoseTrack/(.+\.jpg)"
    else:
        patt=r"/home/data/DetectAndTrack-wjb/lib/datasets/data/mens/(.+\.jpg)"
    image_name=re.findall(patt,tmp_dic["image_name"])[0]
    f=open(dir_path+"/%s.json"%str(image_name).replace('/','\\'),"w")
    f.write(json.dumps(tmp_dic))
    f.flush()
    f.close()
    ####################### save tmp_dic #######################
    
    # Debugging
    # from utils import vis
    # T = vis.vis_one_image_opencv(im[0], cls_boxes[1], keypoints=cls_keyps[1])
    # time_dim = (cls_boxes[1].shape[1] - 1) // 4
    # for t in range(time_dim):
    #     T = vis.vis_one_image_opencv(
    #         im[t],
    #         cls_boxes[1][:, range(t * 4, (t + 1) * 4) + [-1]],
    #         keypoints=[el[:, 17 * t: (t + 1) * 17] for el in cls_keyps[1]])
    #     cv2.imwrite('/tmp/{}.jpg'.format(t + 1), T)
    # import pdb; pdb.set_trace()

    return boxes,scores,cls_boxes, cls_segms, cls_keyps


def im_conv_body_only(model, im):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale_factors = _get_image_blob(im)
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale_factors
