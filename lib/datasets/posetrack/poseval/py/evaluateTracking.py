import numpy as np
import json
import os
import sys
import multiprocessing as mp
from functools import partial

import eval_helpers
from eval_helpers import Joint
sys.path.insert(
    0, 'lib/datasets/posetrack/poseval/py-motmetrics')
import motmetrics as mm

def process_frame(si, nJoints, seqidxs, seqidxsUniq, motAll, metricsMidNames):
    # create metrics
    mh = mm.metrics.create()
    print("seqidx: %d" % (si+1))

    # init per-joint metrics accumulator
    accAll = {}
    for i in range(nJoints):
        accAll[i] = mm.MOTAccumulator(auto_id=True)

    # extract frames IDs for the sequence
    imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
    # DEBUG: remove the last frame of each sequence from evaluation due to buggy annotations
    print("DEBUG: remove last frame from eval until annotations are fixed")
    imgidxs = imgidxs[:-1].copy()
    # create an accumulator that will be updated during each frame
    # iterate over frames
    for j in range(len(imgidxs)):
        imgidx = imgidxs[j,0]
        # iterate over joints
        FP=np.zeros((nJoints+1))
        FN=np.zeros((nJoints+1))
        FN_unmatch_curr=np.zeros((nJoints+1))
        Match_exce_thre=np.zeros((nJoints+1))
        IDSW=np.zeros((nJoints+1))
        gt_num=0
        for i in range(nJoints):
            FP_i,FN_i,FN_unmatch_curr_i,Match_exce_thre_i,IDSW_i=[0],[0],[0],[0],[0]
            # GT tracking ID
            trackidxGT = motAll[imgidx][i]["trackidxGT"]
            # prediction tracking ID
            trackidxPr = motAll[imgidx][i]["trackidxPr"]
            # distance GT <-> pred part to compute MOT metrics
            # 'NaN' means force no match
            dist = motAll[imgidx][i]["dist"]
            # Call update once per frame
            accAll[i].update(
                trackidxGT,  # Ground truth objects in this frame
                trackidxPr,  # Detector hypotheses in this frame
                dist,  # Distances from objects to hypotheses
                FP_i,
                FN_i,
                FN_unmatch_curr_i,
                Match_exce_thre_i,
                IDSW_i
            )
            FP[i]=FP_i[0]
            FN[i]=FN_i[0]
            FN_unmatch_curr_i=FN_unmatch_curr_i[0]
            Match_exce_thre_i=Match_exce_thre_i[0]
            IDSW[i]=IDSW_i[0]
            gt_num+=len(trackidxGT)
        FP[nJoints]=FP.sum()
        FN[nJoints]=FN.sum()
        FN_unmatch_curr[nJoints]=FN_unmatch_curr.sum()
        Match_exce_thre[nJoints]=Match_exce_thre.sum()
        IDSW[nJoints]=IDSW.sum()
        ######################## save fp,fn,idsw values #################
        import re,os,json
        from core.config import get_log_dir_path
#         print(motAll[imgidx][0])
        tmp_dic={
                    "FP":FP.tolist(),
                    "FN":FN.tolist(),
                    "FN_unmatch_curr":FN_unmatch_curr.tolist(),
                    "Match_exce_thre":Match_exce_thre.tolist(),
                    "IDSW":IDSW.tolist(),
                    "gt_num":gt_num,
                    "image_name":motAll[imgidx]["image_name"]
#                     "image_name":motAll[imgidx]["image_name"]
        }
        gt_num_npy=np.full((nJoints+1),tmp_dic["gt_num"])
        tmp_dic["mota_i"]=100*(  1-(FP+FN+IDSW)/  gt_num_npy).mean()
        if True or tmp_dic["mota_i"]<0:
            dir_path=get_log_dir_path()
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            dir_path+="/mid_mota"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            f=open(dir_path+"/mid_mota_%s.json"%(tmp_dic["image_name"]).replace('/','\\'),"w")
            f.write(json.dumps(tmp_dic))
            f.flush()
            f.close()
        ######################## save fp,fn,idsw values #################
    # compute intermediate metrics per joint per sequence
    all_metricsMid = []
    for i in range(nJoints):
        all_metricsMid.append(
            mh.compute(accAll[i], metrics=metricsMidNames,
                       return_dataframe=False, name='acc'))
    return all_metricsMid, accAll


def computeMetrics(gtFramesAll, motAll):

    assert(len(gtFramesAll) == len(motAll))

    pool = mp.Pool(min(12, len(gtFramesAll) + 1))

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    # intermediate metrics
    metricsMidNames = ['num_misses', 'num_switches', 'num_false_positives',
                       'num_objects', 'num_detections']

    # final metrics computed from intermediate metrics
    metricsFinNames = ['mota', 'motp', 'pre', 'rec']

    # initialize intermediate metrics
    metricsMidAll = {}
    for name in metricsMidNames:
        metricsMidAll[name] = np.zeros([1, nJoints])
    metricsMidAll['sumD'] = np.zeros([1, nJoints])

    # initialize final metrics
    metricsFinAll = {}
    for name in metricsFinNames:
        metricsFinAll[name] = np.zeros([1, nJoints + 1])

    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    nSeq = len(seqidxsUniq)

    res_all_metricsMid = pool.map(partial(
        process_frame,
        nJoints=nJoints, seqidxs=seqidxs, seqidxsUniq=seqidxsUniq,
        motAll=motAll, metricsMidNames=metricsMidNames), range(nSeq))
    for si in range(nSeq):
        # compute intermediate metrics per joint per sequence
        all_metricsMid, accAll = res_all_metricsMid[si]
        for i in range(nJoints):
            metricsMid = all_metricsMid[i]
            for name in metricsMidNames:
                metricsMidAll[name][0, i] += metricsMid[name]
            metricsMidAll['sumD'][0, i] += accAll[i].events['D'].sum()

    # compute final metrics per joint for all sequences
    for i in range(nJoints):
        metricsFinAll['mota'][0, i] = 100 * (1. - (
            metricsMidAll['num_misses'][0, i] +
            metricsMidAll['num_switches'][0, i] +
            metricsMidAll['num_false_positives'][0, i]) /
            metricsMidAll['num_objects'][0, i])
        numDet = metricsMidAll['num_detections'][0, i]
        s = metricsMidAll['sumD'][0, i]
        if (numDet == 0 or np.isnan(s)):
            metricsFinAll['motp'][0, i] = 0.0
        else:
            metricsFinAll['motp'][0, i] = 100*(1. - (s / numDet))
        metricsFinAll['pre'][0, i] = 100 * (
            metricsMidAll['num_detections'][0, i] / (
                metricsMidAll['num_detections'][0, i] +
                metricsMidAll['num_false_positives'][0, i]))
        metricsFinAll['rec'][0, i] = 100 * (
            metricsMidAll['num_detections'][0, i] /
            metricsMidAll['num_objects'][0, i])

    # average metrics over all joints over all sequences
    metricsFinAll['mota'][0, nJoints] = metricsFinAll['mota'][0, :nJoints].mean()
    metricsFinAll['motp'][0, nJoints] = metricsFinAll['motp'][0, :nJoints].mean()
    metricsFinAll['pre'][0, nJoints] = metricsFinAll['pre'][0, :nJoints].mean()
    metricsFinAll['rec'][0, nJoints] = metricsFinAll['rec'][0, :nJoints].mean()

    return metricsFinAll


def evaluateTracking(gtFramesAll, prFramesAll, trackUpperBound):

    distThresh = 0.5
    # assign predicted poses to GT poses
    _, _, _, motAll = eval_helpers.assignGTmulti(
        gtFramesAll, prFramesAll, distThresh, trackUpperBound)

    # compute MOT metrics per part
    ###########jianbo comment this following line and add debug line
    metricsAll = computeMetrics(gtFramesAll, motAll)
#     metricsAll =show_MOTA_each_frame(gtFramesAll, motAll)
    ###########jianbo 
    return metricsAll
