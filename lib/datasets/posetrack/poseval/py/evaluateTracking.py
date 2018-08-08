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
global flag
def get_kps_dic(FramesAll,imgidx,FN_kps,FN_kps_indx_i,joint_indx):
    for person_id in range(len(FramesAll[imgidx]["annorect"])):
        person_i=FramesAll[imgidx]["annorect"][person_id]
        person_i_kps=person_i["annopoints"][0]["point"]
        for FN_pi in FN_kps_indx_i:
            if FN_pi==person_i["track_id"][0]:
                
                if not (FN_pi in FN_kps.keys() ):
                    FN_kps[FN_pi]=[]
                for kps_i in person_i_kps:
                    if kps_i["id"][0]==joint_indx:
                        FN_kps[FN_pi].append([kps_i["x"][0],kps_i["y"][0],joint_indx])
                        break
                        
def get_kps_anno(gtFramesAll,imgidx,person_index_list,gt_person_dic,joint_indx,flag0=False):
    
    for person_id in range(len(gtFramesAll[imgidx]["annorect"])):
        person_i=gtFramesAll[imgidx]["annorect"][person_id]
        person_i_kps=person_i["annopoints"][0]["point"]
        person_id=person_i["track_id"][0]
#         if flag0 and "20880" in gtFramesAll[imgidx]["image"][0]["name"] and "0001.jpg" in gtFramesAll[imgidx]["image"][0]["name"]:
#             print(len(gtFramesAll[imgidx]["annorect"]),person_id,person_index_list)
#         if (not flag0) and "20880" in gtFramesAll[imgidx]["image"] and "0001.jpg" in gtFramesAll[imgidx]["image"]:
#             print(person_id,person_index_list)
        if person_id in person_index_list:
            if not (person_id in gt_person_dic.keys() ):
                gt_person_dic[person_id]=[]
            for kps_i in person_i_kps:
                if kps_i["id"][0]==joint_indx:
                    gt_person_dic[person_id].append([kps_i["x"][0],kps_i["y"][0],joint_indx])
                    break                  
def get_gt_pre_dics(gtFramesAll,prFramesAll,imgidx,match_gp_pair):
    
    gt_person_dic={}
    pre_person_dic={}
    for joint_indx in range(len(match_gp_pair)):
        pre_index=[]
        for gt_i in match_gp_pair[joint_indx].keys():
            pre_index.append(match_gp_pair[joint_indx][gt_i])
        get_kps_anno(gtFramesAll,imgidx,match_gp_pair[joint_indx].keys(),gt_person_dic,joint_indx,True)
        get_kps_anno(prFramesAll,imgidx,pre_index,pre_person_dic,joint_indx,False)
    return gt_person_dic,pre_person_dic

# given kps_index, find its coordinates                       
def get_kps_i(FramesAll,imgidx,kps_indx_i,joint_indx,gt_flag=False):
    
    for person_id in range(len(FramesAll[imgidx]["annorect"])):
        person_i=FramesAll[imgidx]["annorect"][person_id]
        print("personid:",person_id)
        person_i_kps=person_i["annopoints"][0]["point"]
        if kps_indx_i==person_i["track_id"][0]:
            for kps_i in person_i_kps:
                if kps_i["id"][0]==joint_indx:
                    if gt_flag:
                        head_coordinate=[person_i["x1"][0],person_i["y1"][0],person_i["x2"][0],person_i["y2"][0]]
                        return [kps_i["x"][0],kps_i["y"][0]],head_coordinate
                    else:
                        return [kps_i["x"][0],kps_i["y"][0]]

def get_dis_i_j(gtFramesAll,prFramesAll,imgidx,joint_indx,FN_match_exc_thre_i):
    from eval_helpers import getHeadSize 
    for i,pair_i in enumerate(FN_match_exc_thre_i):
        gt_i=pair_i["gt"]
        pred_i=pair_i["predict"]
        print("in get_dis_i_j",gtFramesAll[0])
        pointGT,head_coordinate=get_kps_i(gtFramesAll,imgidx,gt_i,joint_indx,gt_flag=True)
        pointPr=get_kps_i(prFramesAll,imgidx,pred_i,joint_indx) 
        headSize = getHeadSize(head_coordinate[0],head_coordinate[1],head_coordinate[2],head_coordinate[3])
        dist_i_j = np.linalg.norm(np.subtract(pointGT, pointPr)) / headSize
        FN_match_exc_thre_i[i]["gt"]={"x":pointGT[0],"y":pointGT[1],"id":joint_indx,"head_size":headSize }
        FN_match_exc_thre_i[i]["predict"]={"x":pointPr[0],"y":pointPr[1],"id":joint_indx}
        FN_match_exc_thre_i[i]["dis"]=dist_i_j 
    return FN_match_exc_thre_i

def process_frame(si, nJoints, seqidxs, seqidxsUniq, motAll, metricsMidNames,gtFramesAll,prFramesAll):
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
        
#         if j==0 or "0001.jpg" in motAll[imgidx]["image_name"]:
#             continue
        # iterate over joints
        FP=np.zeros((nJoints+1))
        FN=np.zeros((nJoints+1))
        FN_unmatch_curr=np.zeros((nJoints+1))
        FN_match_exc_thre=[]
        FN_kps={}
        FP_kps={}
        match_gp_pair=[]
        IDSW=np.zeros((nJoints+1))
        gt_num=np.zeros((nJoints+1))
        
        for i in range(nJoints):
            
            # GT tracking ID
            trackidxGT = motAll[imgidx][i]["trackidxGT"]
            # prediction tracking ID
            trackidxPr = motAll[imgidx][i]["trackidxPr"]
            # distance GT <-> pred part to compute MOT metrics
            # 'NaN' means force no match
            dist = motAll[imgidx][i]["dist"]
            # Call update once per frame
            _,cache=accAll[i].update(
                trackidxGT,  # Ground truth objects in this frame
                trackidxPr,  # Detector hypotheses in this frame
                dist,  # Distances from objects to hypotheses
            )
            FP[i]=cache["FP_i"]
            FN[i]=cache["FN_i"]
            FN_unmatch_curr[i]=cache["FN_unmatch_curr_i"]
            FN_match_exc_thre.append( get_dis_i_j(gtFramesAll,prFramesAll,imgidx,i,cache["FN_match_exc_thre_i"] ))
            IDSW[i]=cache["IDSW_i"]
            gt_num[i]=len(trackidxGT)
            
            ############ for FN
            get_kps_dic(gtFramesAll,imgidx,FN_kps,cache["FN_kps_indx_i"],i)
            get_kps_dic(prFramesAll,imgidx,FP_kps,cache["FP_kps_indx_i"],i)
            
            match_gp_pair.append(cache["match_gp_pair_i"])
        FP[nJoints]=FP.sum()
        FN[nJoints]=FN.sum()
        FN_unmatch_curr[nJoints]=FN_unmatch_curr.sum()
        FN_match_exc_thre.append(0)
        for i in range(nJoints):
            FN_match_exc_thre[nJoints]+=len(FN_match_exc_thre[i])
        IDSW[nJoints]=IDSW.sum()
        gt_num[nJoints]=gt_num.sum()
        gt_person_dic,pre_person_dic=get_gt_pre_dics(gtFramesAll,prFramesAll,imgidx,match_gp_pair)
        
        name_=motAll[imgidx]["image_name"]
        if "145.jpg" in name_ and "00522" in name_:
            flag=True
            print("process_frame%d Catch!!!!!!"%si)
            
        ######################## save fp,fn,idsw values #################
        import re,os,json
        from core.config import get_log_dir_path
        
        tmp_dic={
                    "FP":FP.tolist(),
                    "FN":FN.tolist(),
                    "FN_unmatch_curr":FN_unmatch_curr.tolist(),
                    "FN_match_exc_thre":FN_match_exc_thre,
                    "FN_kps":FN_kps,
                    "FP_kps":FP_kps,
                    "match_gp_pair":match_gp_pair,
                    "gt_person_dic":gt_person_dic,
                    "pre_person_dic":pre_person_dic,
                    "IDSW":IDSW.tolist(),
                    "gt_num":gt_num.tolist(),
                    "image_name":motAll[imgidx]["image_name"]
        }
        
        tmp_dic["mota_i"]=(100*(  1-(FP+FN+IDSW)/  gt_num) ).tolist()
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


def computeMetrics(gtFramesAll, prFramesAll,motAll):
    
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
    
    flag=False
    res_all_metricsMid=[]
    for si in range(nSeq):
        res_all_metricsMid.append( process_frame(si, nJoints, seqidxs, seqidxsUniq, motAll, metricsMidNames,gtFramesAll,prFramesAll) )
    #multi process
#     res_all_metricsMid = pool.map(partial(
#         process_frame,
#         nJoints=nJoints, seqidxs=seqidxs, seqidxsUniq=seqidxsUniq,
#         motAll=motAll, metricsMidNames=metricsMidNames,gtFramesAll=gtFramesAll,prFramesAll=prFramesAll), range(nSeq))
                
    if not flag:
        print("process_frame Catch bad!!!!!!")
        assert(False)
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
    
#     #############debug
#     flag=False
#     for imgidx_ in range(len(gtFramesAll)):
#         name_=gtFramesAll[imgidx_]["image_name"]
#         if "145.jpg" in name_ and "00522" in name_:
#             flag=True
#             print("assign before Catch!!!!!!")
#             break
#     if not flag:
#         print("assign before Catch bad!!!!!!")
#         assert(False)
#     ##################
    
    _, _, _, motAll = eval_helpers.assignGTmulti(
        gtFramesAll, prFramesAll, distThresh, trackUpperBound)
#     _, _, _, motAll = eval_helpers.assignGT_pr_sub_one_multi(
#         gtFramesAll, prFramesAll, distThresh, trackUpperBound)
    
#      #############debug
#     flag=False
#     for imgidx_ in range(len(motAll)):
#         name_=motAll[imgidx_]["image_name"]
#         if "145.jpg" in name_ and "00522" in name_:
#             flag=True
#             print("assign after Catch!!!!!!")
#             break
#     if not flag:
#         print("assign after Catch bad!!!!!!")
#         assert(False)
#     ##################
    
    # compute MOT metrics per part
    ###########jianbo 
    metricsAll = computeMetrics(gtFramesAll, prFramesAll,motAll)
    ###########jianbo 
    return metricsAll
