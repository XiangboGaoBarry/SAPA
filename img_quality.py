import argparse
import os
import sys
import pdb

import cv2
from joblib import load

from utils.quality_assessment.brisque import brisque
from utils.quality_assessment.niqe import niqe
from utils.quality_assessment.piqe import piqe

parser = argparse.ArgumentParser('Test an image')
parser.add_argument(
    '--mode', choices=['brisque', 'niqe', 'piqe'], help='iqa algorithoms,brisque or niqe or piqe')
parser.add_argument('--path', required=True, help='image path')
args = parser.parse_args()

# if __name__ == "__main__":
#     '''
#     test conventional blindly image quality assessment methods(brisque/niqe/piqe)
#     '''
#     mode = args.mode
#     path = args.path
#     im = cv2.imread(path)
#     if im is None:
#         print("please input correct image path!")
#         sys.exit(0)
#     if mode == "piqe":
#         score, _, _, _ = piqe(im)
#     elif mode == "niqe":
#         score = niqe(im)
#     elif mode == "brisque":
#         feature = brisque(im)
#         feature = feature.reshape(1, -1)
#         clf = load('utils/quality_assessment/svr_brisque.joblib')
#         score = clf.predict(feature)[0]
#     print("{}-----{} score:{}".format(path, mode, score))

from brisque import BRISQUE
import numpy as np


import random




if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''
    mode = args.mode
    path = args.path
    p_score_sum = n_score_sum = b_score_sum = b_score_2_sum = 0
    
    from pathlib import Path

    file_path_list = [f for f in Path(path).rglob('*.*')]
    random.shuffle(file_path_list)
    
    for idx, file_path in enumerate(file_path_list):
        im = cv2.imread(str(file_path))
        im = cv2.resize(im, (256, 256))
        p_score, _, _, _ = piqe(im)
        p_score_sum += p_score
        n_score = niqe(im)
        n_score_sum += n_score
        feature = brisque(im)
        feature = feature.reshape(1, -1)
        clf = load('utils/quality_assessment/svr_brisque.joblib')
        b_score = clf.predict(feature)[0]
        b_score_sum += b_score
        

        obj = BRISQUE(url=False)
        b_score_2 = obj.score(np.array(im))
        b_score_2_sum += b_score_2
        
        print("idx:{} Path: {}----- piqe:{}, niqe:{}, brisque:{}, b_score_2:{}".format(idx, file_path, p_score, n_score, b_score, b_score_2))
    print("{} avg:  piqe:{}, niqe:{}, brisque:{}, b_score_2:{}".format(path, p_score_sum/(idx+1), n_score_sum/(idx+1), b_score_sum/(idx+1), b_score_2_sum/(idx+1)))

